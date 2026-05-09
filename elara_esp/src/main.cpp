/*
 * elara_esp/src/main.cpp
 *
 * Pan servo controlled by two sources with clear priority:
 *
 *   1. Active sound (INMP441 mic array, cross-correlation)
 *        → pan toward sound direction
 *        → wins even when face is visible
 *
 *   2. No sound + Pi sending face position (camera tracking)
 *        → pan follows face smoothly
 *        → Pi already selects registered user when multiple faces present
 *
 *   3. No sound + no face
 *        → hold current position
 *
 * Jitter fixes:
 *   - Sound hysteresis: separate ON/OFF thresholds stop threshold flickering
 *   - Per-source smoothing: mic uses lower alpha (noisier), camera uses higher
 *   - Output deadzone: servo only commanded when pan changes >= PAN_DEADZONE_DEG
 *
 * Protocol from Pi: "P<pan_deg> T<tilt_deg>\n"  (tilt received but ignored)
 *
 * INMP441: 24-bit left-justified in 32-bit I2S frame → shift right 8
 *
 * Wiring:
 *   Pan servo → GPIO 13
 *   I2S SD    → GPIO 32
 *   I2S WS    → GPIO 15
 *   I2S SCK   → GPIO 14
 *   TFT pins  → platformio.ini build_flags
 */

#include <Arduino.h>
#include "driver/i2s.h"
#include <ESP32Servo.h>
#include <TFT_eSPI.h>
#include <math.h>

// ── Pins ──────────────────────────────────────────────────────────────────────
#define I2S_SD   32
#define I2S_WS   15
#define I2S_SCK  14
#define PAN_PIN  13

// ── Servo ─────────────────────────────────────────────────────────────────────
#define PW_MIN      500
#define PW_MAX     2500
#define PAN_MIN     30.0f
#define PAN_MAX    150.0f
#define PAN_CENTER  90.0f

// ── Audio ─────────────────────────────────────────────────────────────────────
#define SAMPLE_RATE  16000
#define NUM_SAMPLES  512
#define MIC_DIST_M   0.08f
#define MAX_SHIFT    12

// Hysteresis: sound must exceed ON to activate, fall below OFF to release
#define SOUND_ON_THRESHOLD   600000.0f
#define SOUND_OFF_THRESHOLD  380000.0f

// ── Camera timeout ────────────────────────────────────────────────────────────
#define CAM_TIMEOUT_MS  600

// ── Smoothing (exponential moving average) ────────────────────────────────────
// Mic is noisier → lower alpha = heavier smoothing
#define ALPHA_MIC   0.12f
#define ALPHA_CAM   0.22f
#define ALPHA_IDLE  0.00f   // zero = hold exactly, no drift

// ── Output deadzone ───────────────────────────────────────────────────────────
// Only send a new PWM command when pan moved more than this many degrees.
// Stops the servo buzzing on floating-point micro-changes.
#define PAN_DEADZONE_DEG  0.8f

// ── Display ───────────────────────────────────────────────────────────────────
#define ROBOT_BLUE TFT_CYAN
#define BG_COLOR   TFT_BLACK

// ── Globals ───────────────────────────────────────────────────────────────────
static int32_t raw_buf[NUM_SAMPLES * 2];
static float   left_ch[NUM_SAMPLES];
static float   right_ch[NUM_SAMPLES];

static Servo    pan_servo;
static TFT_eSPI tft = TFT_eSPI();

static float cur_pan  = PAN_CENTER;
static float sent_pan = PAN_CENTER;

static float         cam_pan     = PAN_CENTER;
static unsigned long last_cam_ms = 0;

static bool sound_active = false;

// TFT eye state
static const int EYE_Y          = 30;
static const int LEFT_EYE_BASE  = 38;
static const int RIGHT_EYE_BASE = 103;
static int eye_offset      = 0;
static int prev_eye_offset = -999;

static String serial_buf;

// ── Helpers ───────────────────────────────────────────────────────────────────

static int angle_to_us(float deg) {
    deg = constrain(deg, PAN_MIN, PAN_MAX);
    return (int)(PW_MIN + (deg / 180.0f) * (PW_MAX - PW_MIN));
}

static void update_eye(float pan_deg) {
    float dev = pan_deg - PAN_CENTER;
    int   off = (dev < -15.0f) ? -16 : (dev > 15.0f) ? 16 : 0;
    if (off == eye_offset) return;
    eye_offset = off;
    tft.fillRect(LEFT_EYE_BASE - 20, EYE_Y, 130, 60, BG_COLOR);
    tft.fillRoundRect(LEFT_EYE_BASE  + eye_offset, EYE_Y, 26, 50, 12, ROBOT_BLUE);
    tft.fillRoundRect(RIGHT_EYE_BASE + eye_offset, EYE_Y, 26, 50, 12, ROBOT_BLUE);
}

// ── Serial reader (non-blocking) ──────────────────────────────────────────────

static void read_serial() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n') {
            serial_buf.trim();
            int p = serial_buf.indexOf('P');
            int t = serial_buf.indexOf('T');
            if (p >= 0 && t > p) {
                cam_pan     = serial_buf.substring(p + 1, t).toFloat();
                last_cam_ms = millis();
            }
            serial_buf = "";
        } else {
            serial_buf += c;
        }
    }
}

// ── I2S init ──────────────────────────────────────────────────────────────────

static void init_i2s() {
    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format       = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .dma_buf_count        = 4,
        .dma_buf_len          = NUM_SAMPLES,
        .use_apll             = false
    };
    i2s_pin_config_t pins = {
        .bck_io_num   = I2S_SCK,
        .ws_io_num    = I2S_WS,
        .data_out_num = -1,
        .data_in_num  = I2S_SD
    };
    i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pins);
}

// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);

    tft.init();
    tft.setRotation(1);
    tft.fillScreen(BG_COLOR);

    ESP32PWM::allocateTimer(0);
    pan_servo.setPeriodHertz(50);
    pan_servo.attach(PAN_PIN, PW_MIN, PW_MAX);
    pan_servo.writeMicroseconds(angle_to_us(PAN_CENTER));

    init_i2s();
    update_eye(PAN_CENTER);

    Serial.println("READY");
}

// ── Main loop ─────────────────────────────────────────────────────────────────

void loop() {
    read_serial();

    size_t bytes_read;
    i2s_read(I2S_NUM_0, raw_buf, sizeof(raw_buf), &bytes_read, portMAX_DELAY);

    read_serial();

    // INMP441: 24-bit left-justified → shift right 8
    for (int i = 0; i < NUM_SAMPLES; i++) {
        left_ch[i]  = (float)(raw_buf[i * 2]     >> 8);
        right_ch[i] = (float)(raw_buf[i * 2 + 1] >> 8);
    }

    // Cross-correlation
    float max_corr  = -1e10f;
    int   best_shift = 0;
    for (int shift = -MAX_SHIFT; shift <= MAX_SHIFT; shift++) {
        float corr = 0;
        for (int i = MAX_SHIFT; i < NUM_SAMPLES - MAX_SHIFT; i++)
            corr += left_ch[i] * right_ch[i + shift];
        if (corr > max_corr) { max_corr = corr; best_shift = shift; }
    }

    float sin_theta = constrain(
        ((float)best_shift / SAMPLE_RATE * 343.0f) / MIC_DIST_M, -1.0f, 1.0f);
    float mic_angle = asin(sin_theta) * 180.0f / PI;

    // Sound hysteresis
    if (!sound_active && max_corr > SOUND_ON_THRESHOLD)  sound_active = true;
    if ( sound_active && max_corr < SOUND_OFF_THRESHOLD) sound_active = false;

    bool cam_active = ((millis() - last_cam_ms) < CAM_TIMEOUT_MS);

    // Priority + smoothing
    float target_pan, alpha;
    if (sound_active) {
        target_pan = PAN_CENTER - mic_angle;
        alpha      = ALPHA_MIC;
    } else if (cam_active) {
        target_pan = cam_pan;
        alpha      = ALPHA_CAM;
    } else {
        target_pan = cur_pan;   // hold — no movement
        alpha      = ALPHA_IDLE;
    }

    cur_pan = constrain(
        cur_pan * (1.0f - alpha) + target_pan * alpha,
        PAN_MIN, PAN_MAX
    );

    // Output deadzone — only write to servo when position meaningfully changed
    if (fabsf(cur_pan - sent_pan) >= PAN_DEADZONE_DEG) {
        pan_servo.writeMicroseconds(angle_to_us(cur_pan));
        sent_pan = cur_pan;
    }

    update_eye(cur_pan);
}
