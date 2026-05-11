#include <TFT_eSPI.h>
TFT_eSPI tft = TFT_eSPI();

int eye_w = 40, eye_h = 60;
int black_w = 32, black_h = 52;
int iris_r = 5;
int lookOffset = 0;
unsigned long lastActionTime = 0;

// --- Function Definitions ---

void drawEmoEye(int x, int y, uint16_t color, int h_mod = 0) {
  int final_h = eye_h - h_mod;
  int final_black_h = black_h - h_mod;
  if (final_h < 5) return; 

  tft.fillEllipse(x, y, eye_w / 2, final_h / 2, color);
  tft.fillEllipse(x, y, black_w / 2, final_black_h / 2, TFT_BLACK);
  
  if (final_h > 20) {
    // Iris moves with lookOffset
    tft.fillCircle(x + (lookOffset * 0.6), y - (final_black_h / 4), iris_r, color);
  }
}

void updateFace(uint16_t color = TFT_WHITE, int h_mod = 0) {
  tft.fillScreen(TFT_BLACK);
  drawEmoEye(45 + lookOffset, 64, color, h_mod);
  drawEmoEye(115 + lookOffset, 64, color, h_mod);
}

void blink() {
  for (int h = 0; h <= eye_h; h += 20) { updateFace(TFT_WHITE, h); }
  for (int h = eye_h; h >= 0; h -= 20) { updateFace(TFT_WHITE, h); }
}

// --- Main Logic ---

void setup() {
  Serial.begin(115200);   // USB debug
  Serial2.begin(115200);  // RX2=GPIO16 ← main ESP TX2 (GPIO17)
  tft.init();
  tft.setRotation(1);
  tft.fillScreen(TFT_BLACK);
  updateFace();
}

void loop() {
  if (Serial2.available() > 0) {
    char cmd = Serial2.read();
    
    // Tracking Data: Expected format "X[value]" e.g., "X20"
    if (cmd == 'X') {
      int targetX = Serial.parseInt();
      // Map the Pi error (-160 to 160) to a small eye movement (-15 to 15)
      lookOffset = map(targetX, -160, 160, -15, 15);
      updateFace();
      lastActionTime = millis();
    }
    // Direction from main ESP — sent when wantAngle deviates >15° from center
    else if (cmd == 'L') { lookOffset = -12; updateFace(); lastActionTime = millis(); }
    else if (cmd == 'R') { lookOffset =  12; updateFace(); lastActionTime = millis(); }
    else if (cmd == 'C') { lookOffset =   0; updateFace(); lastActionTime = millis(); }
    // Static Animations
    else if (cmd == '1') { /* Rainbow code */ }
    else if (cmd == '2') { /* Crying code */ }
  }

  // Background Blink (only if not tracking actively)
  if (millis() - lastActionTime > 4000) {
    blink();
    lastActionTime = millis();
  }
}