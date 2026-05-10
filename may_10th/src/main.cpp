#include <Arduino.h>
#include "driver/i2s.h"
#include <ESP32Servo.h>
#include <TFT_eSPI.h>
#include <math.h>

// ── Pins ──────────────────────────────────────────────────────────────────────
#define I2S_SD 32
#define I2S_WS 15
#define I2S_SCK 14
#define SERVO_PIN 13

// ── Audio ─────────────────────────────────────────────────────────────────────
#define SAMPLE_RATE 16000
#define NUM_SAMPLES 512
#define MIC_SPACING_M 0.08f

// Thresholds derived from quiet log statistics (172 samples, >> 8 scaling):
//   quiet mean = 181B, quiet std = 100B
//   SOUND_ON  = mean + 3σ = 481B → rejects 99.7% of quiet noise
//   SOUND_OFF = mean + 1.5σ = 331B → holds through word pauses, clears on real silence
#define SOUND_ON_THRESH  220000000000.0f   
#define SOUND_OFF_THRESH 143000000000.0f   

static int32_t dmaBuf[NUM_SAMPLES * 2];
static float lCh[NUM_SAMPLES];
static float rCh[NUM_SAMPLES];

// ── NEW: Sound Filtering Variables ───────────────────────────────────────────
float noise_floor = 18000.0f;         // measured quiet-room RMS average (>> 8 scaling)
const float NOISE_MULTIPLIER = 1.7f;  // trigger at ~45000, sits in the quiet/speech gap
const unsigned long SOUND_CONFIRM_MS = 400;    // 400ms of continuous speech to confirm direction

unsigned long sound_confirm_start = 0;
float last_sound_angle = 90.0f;
bool sound_direction_confirmed = false;
// ─────────────────────────────────────────────────────────────────────────────

// ── TFT ───────────────────────────────────────────────────────────────────────
TFT_eSPI tft = TFT_eSPI();
#define EYE_COLOR TFT_CYAN
#define BG_COLOR TFT_BLACK
static const int EYE_Y = 30;
static const int L_EYE_X = 38;
static const int R_EYE_X = 103;
static int prevEyeOff = 999;

// ── Servo ─────────────────────────────────────────────────────────────────────
Servo panServo;
static float liveAngle = 90.0f;
static float wantAngle = 90.0f;
static const float EMA_ALPHA = 0.12f;
static const float RATE_DEG = 2.5f;
static const float DEADBAND = 1.2f;

// Minimum angular difference (degrees) between sound and face before we hand off
#define SOUND_OVERRIDE_DEG 25.0f

// ── Mode ──────────────────────────────────────────────────────────────────────
enum Mode { FACE, SOUND, IDLE } mode = IDLE;
static volatile float gSoundAngle = 90.0f;
static volatile bool gSoundActive = false;
static SemaphoreHandle_t sMutex;

// ── Serial / face ─────────────────────────────────────────────────────────────
static String rxBuf;
static float faceAngle = 90.0f;
static uint32_t lastFaceMs = 0;
#define FACE_TIMEOUT_MS 2000u

// ── Helper Functions ──────────────────────────────────────────────────────────
float calculateRMS(float* left, float* right, int samples) {
    float sum = 0;
    for (int i = 0; i < samples; i++) {
        float avg = (left[i] + right[i]) / 2.0f;
        sum += avg * avg;
    }
    return sqrt(sum / samples);
}

// Stable first-order high-pass: removes DC and low-frequency rumble (<~13 Hz).
// Each channel gets its own state so they don't bleed into each other.
// Pole at alpha=0.995, always inside the unit circle → always stable.
static float hp_x1_L = 0, hp_y1_L = 0;
static float hp_x1_R = 0, hp_y1_R = 0;

void applyHighPass(float* buffer, int len, float* x1_st, float* y1_st) {
    const float alpha = 0.995f;
    for (int i = 0; i < len; i++) {
        float x0 = buffer[i];
        float y0 = alpha * (*y1_st + x0 - *x1_st);
        *x1_st = x0;
        *y1_st = y0;
        buffer[i] = y0;
    }
}

// ── I2S task — Core 0 ─────────────────────────────────────────────────────────
void taskI2S(void*) {
    bool soundOn = false;
    for (;;) {
        size_t bytes;
        i2s_read(I2S_NUM_0, dmaBuf, sizeof(dmaBuf), &bytes, portMAX_DELAY);
        int n = (int)(bytes / (2 * sizeof(int32_t)));
        if (n < 1) continue;

        // >> 8: correct shift for INMP441 (24-bit data left-justified in 32-bit frame)
        // matches the diagnostic tool so measured thresholds apply directly
        for (int i = 0; i < n; i++) {
            lCh[i] = (float)(dmaBuf[i * 2]     >> 8);
            rCh[i] = (float)(dmaBuf[i * 2 + 1] >> 8);
        }

        // Remove DC/rumble separately per channel so states don't bleed
        applyHighPass(lCh, n, &hp_x1_L, &hp_y1_L);
        applyHighPass(rCh, n, &hp_x1_R, &hp_y1_R);

        float current_rms = calculateRMS(lCh, rCh, n);

        // Update RMS noise floor when quiet (same EMA pattern for both floors)
        if (current_rms < noise_floor * 2.5f) {
            noise_floor = noise_floor * 0.97f + current_rms * 0.03f;
        }

        // Cross-correlation
        float maxCorr = -1e10f;
        int best = 0;
        const int MS = 12;
        for (int s = -MS; s <= MS; s++) {
            float c = 0;
            for (int i = MS; i < n - MS; i++) {
                c += lCh[i] * rCh[i + s];
            }
            if (c > maxCorr) {
                maxCorr = c;
                best = s;
            }
        }

        if (!soundOn && maxCorr > SOUND_ON_THRESH)  soundOn = true;
        if ( soundOn && maxCorr < SOUND_OFF_THRESH) soundOn = false;

        float angle = 90.0f;
        if (soundOn) {
            float td = (float)best / SAMPLE_RATE;
            float sv = constrain(td * 343.0f / MIC_SPACING_M, -1.0f, 1.0f);
            angle = 90.0f - asinf(sv) * (180.0f / (float)M_PI);
        }

        // === Dynamic Confirmation Logic ===
        float trigger_threshold = noise_floor * NOISE_MULTIPLIER;

        if (current_rms > trigger_threshold && soundOn) {
            if (!sound_direction_confirmed) {
                // Start timer only once per detection window — never reset while waiting
                if (sound_confirm_start == 0) {
                    sound_confirm_start = millis();
                }
                if (millis() - sound_confirm_start >= SOUND_CONFIRM_MS) {
                    sound_direction_confirmed = true;
                    last_sound_angle = angle;
                }
            } else {
                last_sound_angle = angle;
                // Keep the "silence" timer fresh while sound is still confirmed+active
                sound_confirm_start = millis();
            }
        } else {
            if (sound_confirm_start != 0 && millis() - sound_confirm_start > 800) {
                sound_direction_confirmed = false;
                sound_confirm_start = 0;   // fully reset so next detection starts clean
            }
        }

        // Final decision
        bool finalSoundActive = soundOn && sound_direction_confirmed;
        float finalSoundAngle = sound_direction_confirmed ? last_sound_angle : 90.0f;

        xSemaphoreTake(sMutex, portMAX_DELAY);
        gSoundAngle = finalSoundAngle;
        gSoundActive = finalSoundActive;
        xSemaphoreGive(sMutex);
    }
}

// ── TFT eyes ──────────────────────────────────────────────────────────────────
void drawEyes(int off) {
    if (off == prevEyeOff) return;
    prevEyeOff = off;
    tft.fillRect(L_EYE_X - 20, EYE_Y, 130, 60, BG_COLOR);
    tft.fillRoundRect(L_EYE_X + off, EYE_Y, 26, 50, 12, EYE_COLOR);
    tft.fillRoundRect(R_EYE_X + off, EYE_Y, 26, 50, 12, EYE_COLOR);
}

// ── Servo drive ───────────────────────────────────────────────────────────────
void driveServo() {
    float next = EMA_ALPHA * wantAngle + (1.0f - EMA_ALPHA) * liveAngle;
    float d = constrain(next - liveAngle, -RATE_DEG, RATE_DEG);
    if (fabsf(d) < DEADBAND) return;
    liveAngle += d;
    panServo.write((int)liveAngle);
}

// ── Serial parser ─────────────────────────────────────────────────────────────
void parseSerial() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n') {
            rxBuf.trim();
            if (rxBuf.length() > 0 && rxBuf[0] == 'F') {
                faceAngle = constrain(rxBuf.substring(1).toFloat(), 0.0f, 180.0f);
                lastFaceMs = millis();
            }
            rxBuf = "";
        } else if (rxBuf.length() < 32) {
            rxBuf += c;
        }
    }
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    tft.init();
    tft.setRotation(1);
    tft.fillScreen(BG_COLOR);
    drawEyes(0);

    ESP32PWM::allocateTimer(0);
    panServo.setPeriodHertz(50);
    panServo.attach(SERVO_PIN, 500, 2500);
    panServo.write(90);

    i2s_config_t i2sCfg = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .dma_buf_count = 4,
        .dma_buf_len = NUM_SAMPLES,
        .use_apll = false
    };
    i2s_pin_config_t i2sPins = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = -1,
        .data_in_num = I2S_SD
    };

    i2s_driver_install(I2S_NUM_0, &i2sCfg, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &i2sPins);

    sMutex = xSemaphoreCreateMutex();
    xTaskCreatePinnedToCore(taskI2S, "i2s", 4096, NULL, 1, NULL, 0);

    Serial.println("READY");
}

// ── Loop — Core 1 ─────────────────────────────────────────────────────────────
void loop() {
    parseSerial();

    bool faceActive = (millis() - lastFaceMs < FACE_TIMEOUT_MS);
    float snapAngle;
    bool snapActive;

    xSemaphoreTake(sMutex, portMAX_DELAY);
    snapAngle = gSoundAngle;
    snapActive = gSoundActive;
    xSemaphoreGive(sMutex);

    // Sound override: face is active but confirmed sound is coming from a clearly
    // different direction — someone else started talking, hand off to them.
    bool soundOverride = faceActive && snapActive &&
                         fabsf(snapAngle - faceAngle) > SOUND_OVERRIDE_DEG;

    static bool wasOverride = false;
    if (wasOverride && !soundOverride) {
        // Override just ended. The Pi's faceAngle is stale (it was tracking
        // Person A while we were pointing at Person B). Flush it so the system
        // waits for fresh commands based on wherever the camera is now.
        // Pi's sync_pan_to_esp32() will re-acquire from the current position.
        lastFaceMs = 0;
        faceAngle  = liveAngle;
    }
    wasOverride = soundOverride;

    if (soundOverride) {
        mode = SOUND;
        wantAngle = snapAngle;
    } else if (faceActive) {
        mode = FACE;
        wantAngle = faceAngle;
    } else if (snapActive) {
        mode = SOUND;
        wantAngle = snapAngle;
        faceAngle = liveAngle;
    } else {
        mode = IDLE;
        wantAngle = liveAngle;
        faceAngle = liveAngle;
    }

    driveServo();

    // Report live angle to Pi every 200 ms
    static uint32_t lastRptMs = 0;
    if (millis() - lastRptMs >= 200) {
        lastRptMs = millis();
        Serial.print('A');
        Serial.println(liveAngle, 1);
    }

    float diff = wantAngle - 90.0f;
    int off = (diff < -15.0f) ? -16 : (diff > 15.0f) ? 16 : 0;
    drawEyes(off);

    delay(20);
}