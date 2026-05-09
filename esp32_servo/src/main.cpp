#include <Arduino.h>
#include <ESP32Servo.h>

// Wire pan  servo signal → GPIO 13
// Wire tilt servo signal → GPIO 12
// Power servos from a separate 5 V rail (not the ESP32 3.3 V pin)

static const int PAN_PIN  = 13;
static const int TILT_PIN = 12;

// Pulse range matching the Python side: 500–2500 µs → 0°–180°
static const int PW_MIN = 500;
static const int PW_MAX = 2500;

Servo pan_servo;
Servo tilt_servo;

// Convert degrees → microseconds (clamp to safe range)
static int angle_to_us(float deg) {
    deg = constrain(deg, 0.0f, 180.0f);
    return (int)(PW_MIN + (deg / 180.0f) * (PW_MAX - PW_MIN));
}

void setup() {
    Serial.begin(115200);

    // Allocate LEDC timers before attaching
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);

    pan_servo.setPeriodHertz(50);
    tilt_servo.setPeriodHertz(50);

    pan_servo.attach(PAN_PIN,  PW_MIN, PW_MAX);
    tilt_servo.attach(TILT_PIN, PW_MIN, PW_MAX);

    // Start centred
    pan_servo.writeMicroseconds(angle_to_us(90));
    tilt_servo.writeMicroseconds(angle_to_us(90));

    Serial.println("READY");
}

// Parser state
static String line_buf;

void loop() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n') {
            line_buf.trim();

            // Protocol: "P<float> T<float>"
            // Example:  "P105.3 T72.0"
            float pan_deg = 90.0f, tilt_deg = 90.0f;
            int p_pos = line_buf.indexOf('P');
            int t_pos = line_buf.indexOf('T');

            if (p_pos >= 0 && t_pos > p_pos) {
                pan_deg  = line_buf.substring(p_pos + 1, t_pos).toFloat();
                tilt_deg = line_buf.substring(t_pos + 1).toFloat();

                pan_servo.writeMicroseconds(angle_to_us(pan_deg));
                tilt_servo.writeMicroseconds(angle_to_us(tilt_deg));
            }

            line_buf = "";
        } else {
            line_buf += c;
        }
    }
}
