#include <Arduino.h>
#include "driver/i2s.h"
#include <ESP32Servo.h>
#include <TFT_eSPI.h>
#include <math.h>

// ── Pins ──────────────────────────────────────────────────────────────────────
#define I2S_SD    32
#define I2S_WS    15
#define I2S_SCK   14
#define SERVO_PIN 13

// ── Audio ─────────────────────────────────────────────────────────────────────
#define SAMPLE_RATE      16000
#define NUM_SAMPLES      512
#define MIC_SPACING_M    0.08f
#define SOUND_ON_THRESH  500000.0f
#define SOUND_OFF_THRESH 250000.0f

static int32_t dmaBuf[NUM_SAMPLES * 2];
static float   lCh[NUM_SAMPLES];
static float   rCh[NUM_SAMPLES];

// ── TFT ───────────────────────────────────────────────────────────────────────
TFT_eSPI tft = TFT_eSPI();
#define EYE_COLOR  TFT_CYAN
#define BG_COLOR   TFT_BLACK
static const int EYE_Y   = 30;
static const int L_EYE_X = 38;
static const int R_EYE_X = 103;
static int prevEyeOff    = 999;

// ── Servo ─────────────────────────────────────────────────────────────────────
Servo panServo;
static float liveAngle = 90.0f;
static float wantAngle = 90.0f;

static const float EMA_ALPHA = 0.12f;
static const float RATE_DEG  = 2.5f;
static const float DEADBAND  = 1.2f;

// ── Mode ──────────────────────────────────────────────────────────────────────
enum Mode { FACE, SOUND, IDLE } mode = IDLE;

static volatile float  gSoundAngle  = 90.0f;
static volatile bool   gSoundActive = false;
static SemaphoreHandle_t sMutex;

// ── Serial / face ─────────────────────────────────────────────────────────────
static String   rxBuf;
static float    faceAngle  = 90.0f;
static uint32_t lastFaceMs = 0;
#define FACE_TIMEOUT_MS 2000u

// ── I2S task — Core 0 ─────────────────────────────────────────────────────────
void taskI2S(void*) {
    bool soundOn = false;

    for (;;) {
        size_t bytes;
        i2s_read(I2S_NUM_0, dmaBuf, sizeof(dmaBuf), &bytes, portMAX_DELAY);

        int n = (int)(bytes / (2 * sizeof(int32_t)));
        if (n < 1) continue;

        for (int i = 0; i < n; i++) {
            lCh[i] = (float)(dmaBuf[i * 2]     >> 14);
            rCh[i] = (float)(dmaBuf[i * 2 + 1] >> 14);
        }

        float maxCorr = -1e10f;
        int   best    = 0;
        const int MS  = 12;
        for (int s = -MS; s <= MS; s++) {
            float c = 0;
            for (int i = MS; i < n - MS; i++) c += lCh[i] * rCh[i + s];
            if (c > maxCorr) { maxCorr = c; best = s; }
        }

        if (!soundOn && maxCorr > SOUND_ON_THRESH)  soundOn = true;
        if ( soundOn && maxCorr < SOUND_OFF_THRESH) soundOn = false;

        float angle = 90.0f;
        if (soundOn) {
            float td = (float)best / SAMPLE_RATE;
            float sv = constrain(td * 343.0f / MIC_SPACING_M, -1.0f, 1.0f);
            angle = 90.0f - asinf(sv) * (180.0f / (float)M_PI);
        }

        xSemaphoreTake(sMutex, portMAX_DELAY);
        gSoundAngle  = angle;
        gSoundActive = soundOn;
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
    float d    = constrain(next - liveAngle, -RATE_DEG, RATE_DEG);
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
                faceAngle  = constrain(rxBuf.substring(1).toFloat(), 0.0f, 180.0f);
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
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format       = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .dma_buf_count        = 4,
        .dma_buf_len          = NUM_SAMPLES,
        .use_apll             = false
    };
    i2s_pin_config_t i2sPins = {
        .bck_io_num   = I2S_SCK,
        .ws_io_num    = I2S_WS,
        .data_out_num = -1,
        .data_in_num  = I2S_SD
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

    bool  faceActive = (millis() - lastFaceMs < FACE_TIMEOUT_MS);

    float snapAngle;
    bool  snapActive;
    xSemaphoreTake(sMutex, portMAX_DELAY);
    snapAngle  = gSoundAngle;
    snapActive = gSoundActive;
    xSemaphoreGive(sMutex);

    if (faceActive) {
        mode      = FACE;
        wantAngle = faceAngle;
    } else if (snapActive) {
        mode      = SOUND;
        wantAngle = snapAngle;
        faceAngle = liveAngle;  // keep faceAngle current so no jump on re-entry
    } else {
        mode      = IDLE;
        wantAngle = liveAngle;  // anchor EMA so servo holds exactly here
        faceAngle = liveAngle;  // same — no jump when face reappears
    }

    driveServo();

    // Report live angle to Pi every 200 ms so Python can sync its pan state
    // after SOUND mode moves the servo autonomously.
    static uint32_t lastRptMs = 0;
    if (millis() - lastRptMs >= 200) {
        lastRptMs = millis();
        Serial.print('A');
        Serial.println(liveAngle, 1);
    }

    float diff = wantAngle - 90.0f;
    int   off  = (diff < -15.0f) ? -16 : (diff > 15.0f) ? 16 : 0;
    drawEyes(off);

    delay(20);
}
