# Face Login Elara - Tracking System Working Logic

## Overview
The system uses a **Raspberry Pi + ESP32** setup for smooth face tracking with intelligent hybrid sound localization. It supports seamless handoff between different people, including the ability to shift attention toward a new speaker **even while another person remains in the frame**.

## System Components & Their Roles

### 1. **tracker.py** (The Brain - Runs on Raspberry Pi)
- Captures video from the camera.
- Detects faces using fast Haar Cascade.
- Uses **PID controller code** (one for Pan, one for Tilt) to calculate required movement.
- Periodically runs face recognition to identify the person.
- Decides target angles and sends commands to `servo.py`.
- Manages `_no_face_count` to detect when the tracked person has left the frame.

### 2. **servo.py** (The Messenger - Runs on Raspberry Pi)
- Communicates with ESP32 over USB Serial.
- Sends commands like `F105.4` (Face mode + angle).
- Reads angle feedback from ESP32 (`Axxx.x`).
- Contains `sync_pan_to_esp32()` function to update Pi’s internal angle with the real servo position.
- Applies safety limits and slew rate limiting.

### 3. **ESP32** (`may_10th` project)
- Receives commands from Pi and controls the physical servo motors.
- Performs **sound localization** using two microphones with bandpass filtering and dynamic noise floor.
- Supports **Hybrid Face + Sound** behavior.
- Automatically switches between **FACE**, **SOUND**, and **IDLE** modes.
- Sends current angle back to Pi every 200ms.
- Shows animated eyes on the TFT display.

---

## Current Working Logic

### Step-by-Step Flow

1. **Person A Detected & Tracking Starts**
   - Camera sees Person A.
   - `tracker.py` detects face → runs PID → sends `Fxxx.x` commands continuously.
   - ESP32 switches to **FACE** mode and follows the commands smoothly.

2. **Person A Stops Talking but Remains in Frame + Person B Starts Talking**
   - ESP32 continues monitoring sound direction and energy **even in FACE mode**.
   - If Person B produces significantly louder sound (after filtering and confirmation), the system triggers a **soft attention shift**.
   - Servo gradually pans toward Person B’s direction while still respecting face tracking.

3. **Person A Leaves the Frame**
   - `_no_face_count` increases in `tracker.py`.
   - After **2 seconds**, ESP32 fully switches to **SOUND** mode if needed.

4. **Person B Appears in Front of Camera**
   - `tracker.py` detects face → calls `sync_pan_to_esp32()` → resumes normal face tracking smoothly on Person B.

---

## Key Features Implemented

- **Hybrid Attention Shift** (New): Camera can shift toward a new louder speaker (Person B) **even while Person A is still visible** in the frame.
- Improved Sound Processing: Bandpass filter + dynamic noise floor + confirmation logic to reduce false triggers.
- Smooth handoff without jumping back to previous positions.
- Face tracking remains the primary behavior, with sound acting as an intelligent attractor.

---

## Mode Switching Summary

| Situation                                      | Mode on ESP32     | Who Controls Movement                  | Sound Monitoring |
|-----------------------------------------------|-------------------|----------------------------------------|------------------|
| Face tracked (normal)                         | FACE              | Raspberry Pi + PID                     | Background       |
| Person A silent + Person B speaking loudly    | FACE (Hybrid)     | Pi + Sound Suggestion (soft shift)     | Active           |
| No face for > 2 seconds                       | SOUND / IDLE      | ESP32                                  | Full             |
| Face reappears                                | FACE              | Back to Raspberry Pi + PID             | Background       |

---

## Important Notes
- The system now feels more natural and socially aware.
- Sound localization is reliable thanks to filtering.
- Tunable parameters:
  - `FACE_TIMEOUT_MS` in ESP32 code
  - `_no_face_count` threshold in `tracker.py`
  - Sound filtering: `NOISE_MULTIPLIER`, `SOUND_CONFIRM_MS`, `noise_floor`

---
