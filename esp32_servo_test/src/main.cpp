/*
  esp32_servo_test — manual servo test firmware with IMU jitter detection
  ─────────────────────────────────────────────────────────────────────────
  Wiring
    Pan  servo signal  →  GPIO 13
    Tilt servo signal  →  GPIO 12
    Servo power        →  separate 5 V rail (NOT the ESP32 3.3 V pin)

    MPU6050 (mount on servo horn/arm — measures real mechanical movement)
    SDA  →  GPIO 21
    SCL  →  GPIO 22
    VCC  →  3.3 V
    GND  →  GND
    AD0  →  GND  (I2C address 0x68)

  If no MPU6050 is connected the firmware runs normally — GYRO lines simply
  won't appear in the serial output.

  Serial protocol (115200 baud, newline-terminated)
  ─────────────────────────────────────────────────
  RX (Pi → ESP32):
    "P<float> T<float>\n"   move servos
    "C\n"                   centre both servos

  TX (ESP32 → Pi):
    "READY"                             on boot
    "LIMITS P30-150 T40-140"            mechanical limits reminder
    "IMU ok"  or  "IMU not found"       sensor probe result
    "MOVED  P<pan> T<tilt>"             after every move
    "CLAMPED P<req>→<act> T<req>→<act>" angle was out of range
    "STATUS P<pan> T<tilt> uptime=<s>"  heartbeat every 2 s
    "GYRO gx=<> gy=<> gz=<> rms=<>"    raw gyro (rad/s) + jitter RMS, 20 Hz
    "ERR bad command"                   unrecognised input
*/

#include <Arduino.h>
#include <ESP32Servo.h>
#include <Wire.h>

// ── Servo pins ────────────────────────────────────────────────────────────────
static const int PAN_PIN  = 13;
static const int TILT_PIN = 12;
static const int PW_MIN   = 500;
static const int PW_MAX   = 2500;

static const float PAN_MIN  = 30.0f,  PAN_MAX  = 150.0f;
static const float TILT_MIN = 40.0f,  TILT_MAX = 140.0f;
static const float PAN_CTR  = 90.0f,  TILT_CTR = 90.0f;

Servo pan_servo;
Servo tilt_servo;

float cur_pan  = PAN_CTR;
float cur_tilt = TILT_CTR;

// ── MPU6050 bare-metal I2C ────────────────────────────────────────────────────
// Using Wire directly avoids adding a library dependency.
static const uint8_t MPU_ADDR      = 0x68;
static const uint8_t REG_PWR_MGMT  = 0x6B;
static const uint8_t REG_GYRO_CFG  = 0x1B;  // ±250 °/s  → 131 LSB/(°/s)
static const uint8_t REG_GYRO_OUT  = 0x43;  // 6 bytes: XH XL YH YL ZH ZL
static const float   GYRO_SCALE    = 1.0f / 131.0f * (3.14159f / 180.0f); // to rad/s

static bool imu_ok = false;

// Rolling RMS window for jitter score (last N gyro magnitude samples)
static const int RMS_WIN = 20;
static float     rms_buf[RMS_WIN];
static int       rms_idx = 0;
static bool      rms_full = false;

static bool mpu_write(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(reg);
    Wire.write(val);
    return Wire.endTransmission() == 0;
}

static bool mpu_read_gyro(float &gx, float &gy, float &gz) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(REG_GYRO_OUT);
    if (Wire.endTransmission(false) != 0) return false;
    Wire.requestFrom(MPU_ADDR, (uint8_t)6);
    if (Wire.available() < 6) return false;
    int16_t rx = (Wire.read() << 8) | Wire.read();
    int16_t ry = (Wire.read() << 8) | Wire.read();
    int16_t rz = (Wire.read() << 8) | Wire.read();
    gx = rx * GYRO_SCALE;
    gy = ry * GYRO_SCALE;
    gz = rz * GYRO_SCALE;
    return true;
}

static float rms_jitter() {
    int n = rms_full ? RMS_WIN : rms_idx;
    if (n == 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += rms_buf[i] * rms_buf[i];
    return sqrt(sum / n);
}

// ── helpers ───────────────────────────────────────────────────────────────────
static int angle_to_us(float deg) {
    deg = constrain(deg, 0.0f, 180.0f);
    return (int)(PW_MIN + (deg / 180.0f) * (PW_MAX - PW_MIN));
}

static void apply(float pan, float tilt) {
    pan_servo.writeMicroseconds(angle_to_us(pan));
    tilt_servo.writeMicroseconds(angle_to_us(tilt));
    cur_pan  = pan;
    cur_tilt = tilt;
}

// ── setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);

    // servos
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    pan_servo.setPeriodHertz(50);
    tilt_servo.setPeriodHertz(50);
    pan_servo.attach(PAN_PIN,  PW_MIN, PW_MAX);
    tilt_servo.attach(TILT_PIN, PW_MIN, PW_MAX);
    apply(PAN_CTR, TILT_CTR);

    // IMU
    Wire.begin();
    Wire.beginTransmission(MPU_ADDR);
    imu_ok = (Wire.endTransmission() == 0);
    if (imu_ok) {
        mpu_write(REG_PWR_MGMT, 0x00);   // wake up
        mpu_write(REG_GYRO_CFG, 0x00);   // ±250 °/s full scale
        delay(100);
        Serial.println("IMU ok");
    } else {
        Serial.println("IMU not found — mount MPU6050 on servo arm for jitter data");
    }

    Serial.println("READY");
    Serial.printf("LIMITS P%.0f-%.0f T%.0f-%.0f\n",
                  PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX);
}

// ── loop ──────────────────────────────────────────────────────────────────────
static String line_buf;
static unsigned long last_status_ms = 0;
static unsigned long last_gyro_ms   = 0;
static const unsigned long STATUS_MS = 2000;
static const unsigned long GYRO_MS   = 50;   // 20 Hz IMU reports

void loop() {
    // ── serial command parser ─────────────────────────────────────────────────
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n') {
            line_buf.trim();

            if (line_buf.length() == 0) { line_buf = ""; continue; }

            if (line_buf.equalsIgnoreCase("C")) {
                apply(PAN_CTR, TILT_CTR);
                Serial.printf("MOVED P%.1f T%.1f (centred)\n", cur_pan, cur_tilt);
                line_buf = ""; continue;
            }

            int p_pos = line_buf.indexOf('P');
            int t_pos = line_buf.indexOf('T');
            if (p_pos >= 0 && t_pos > p_pos) {
                float req_pan  = line_buf.substring(p_pos + 1, t_pos).toFloat();
                float req_tilt = line_buf.substring(t_pos + 1).toFloat();
                float act_pan  = constrain(req_pan,  PAN_MIN,  PAN_MAX);
                float act_tilt = constrain(req_tilt, TILT_MIN, TILT_MAX);
                bool  clamped  = (req_pan != act_pan) || (req_tilt != act_tilt);

                apply(act_pan, act_tilt);

                if (clamped)
                    Serial.printf("CLAMPED P%.1f->%.1f T%.1f->%.1f\n",
                                  req_pan, act_pan, req_tilt, act_tilt);
                else
                    Serial.printf("MOVED P%.1f T%.1f\n", act_pan, act_tilt);
            } else {
                Serial.printf("ERR bad command: %s\n", line_buf.c_str());
            }
            line_buf = "";
        } else {
            line_buf += c;
        }
    }

    unsigned long now = millis();

    // ── IMU gyro read + jitter score ─────────────────────────────────────────
    if (imu_ok && now - last_gyro_ms >= GYRO_MS) {
        last_gyro_ms = now;
        float gx, gy, gz;
        if (mpu_read_gyro(gx, gy, gz)) {
            float mag = sqrt(gx*gx + gy*gy + gz*gz);
            rms_buf[rms_idx] = mag;
            rms_idx = (rms_idx + 1) % RMS_WIN;
            if (rms_idx == 0) rms_full = true;

            Serial.printf("GYRO gx=%.4f gy=%.4f gz=%.4f rms=%.4f\n",
                          gx, gy, gz, rms_jitter());
        }
    }

    // ── status heartbeat ─────────────────────────────────────────────────────
    if (now - last_status_ms >= STATUS_MS) {
        last_status_ms = now;
        Serial.printf("STATUS P%.1f T%.1f uptime=%lus jitter=%.4f\n",
                      cur_pan, cur_tilt, now / 1000UL,
                      imu_ok ? rms_jitter() : -1.0f);
    }
}
