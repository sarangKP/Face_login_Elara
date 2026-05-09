#!/usr/bin/env python3
"""
pi_facetrack.py — face tracking bridge for the ESP32 combined firmware.

Detects the largest face in every camera frame, computes a pan angle, and
sends it to the ESP32 over Serial so the ESP32 can override its mic-based
sound tracking.

Protocol  (Pi → ESP32 only):
    "F<degrees>\\n"   face detected, pan servo to this angle (0–180)
    "N\\n"            no face; ESP32 will fall back to mic / idle after timeout

Calibration flags (top of file):
    FLIP_FRAME   True if the picamera image needs horizontal mirror
    FLIP_SERVO   True if the servo moves the wrong way (pan direction inverted)

Run:
    python3 pi_facetrack.py

Requires:
    picamera2   (pre-installed on Pi OS)
    opencv-python
    pyserial    (pip install pyserial)
"""

import sys
import time
import logging
import numpy as np
import cv2
import serial

# picamera2 lives in the system packages on Pi OS
sys.path.insert(0, "/usr/lib/python3/dist-packages")
from picamera2 import Picamera2

# ── User-tunable settings ─────────────────────────────────────────────────────

SERIAL_PORT   = "/dev/ttyUSB0"   # change to /dev/ttyACM0 if needed
SERIAL_BAUD   = 115200

FRAME_W       = 640
FRAME_H       = 480
DETECT_W      = 320              # Haar runs at this resolution (faster)
DETECT_H      = 240

# Mirror the camera image so left/right match physical reality.
# If the face moves right but the servo pans left, toggle FLIP_SERVO instead.
FLIP_FRAME    = True

# Invert the pan direction sent to the servo.
# Set True if the servo consistently moves the wrong way.
FLIP_SERVO    = False

# PAN_KP: degrees added per pixel of face-centre error.
# Higher → more aggressive tracking; lower → smoother but slower.
#   Typical range: 0.02 (very gentle) … 0.08 (snappy)
PAN_KP        = 0.04

PAN_CENTER    = 90.0             # servo rest position
PAN_MIN       = 5.0              # never command below this
PAN_MAX       = 175.0            # never command above this

# Face must be this many pixels off-centre before we move the servo.
# The Haar bounding box wobbles ~15–20 px on a still face — keep this above
# that wobble or the servo will never settle.
DEAD_ZONE_PX  = 25

# Target send rate to the ESP32 (Hz).
SEND_HZ       = 20
_SEND_DT      = 1.0 / SEND_HZ

# After losing a face keep sending "N" for this many seconds, then go quiet.
# The ESP32 has its own 600 ms face-timeout, so anything > 0.6 s is fine.
NO_FACE_SEND_S = 1.5

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Face detector
# ─────────────────────────────────────────────────────────────────────────────
class FaceDetector:
    def __init__(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cc = cv2.CascadeClassifier(path)
        if self._cc.empty():
            raise RuntimeError(f"Haar cascade not found: {path}")

    def detect(self, frame_rgb: np.ndarray):
        """
        Return (cx, cy, w, h) in full-resolution coordinates of the largest
        face, or None if no face is found.
        """
        small = cv2.resize(frame_rgb, (DETECT_W, DETECT_H),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        cv2.equalizeHist(gray, gray)

        faces = self._cc.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if not len(faces):
            return None

        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        sx = FRAME_W / DETECT_W
        sy = FRAME_H / DETECT_H
        return (int((x + w // 2) * sx),
                int((y + h // 2) * sy),
                int(w * sx),
                int(h * sy))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Serial ────────────────────────────────────────────────────────────────
    log.info("Opening %s @ %d baud", SERIAL_PORT, SERIAL_BAUD)
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    except serial.SerialException as e:
        log.error("Cannot open serial port: %s", e)
        log.error("Check connection.  List ports:  ls /dev/ttyUSB*  /dev/ttyACM*")
        return

    # ESP32 resets when the serial port opens (DTR pulse) — wait for it
    time.sleep(2.0)
    ready = ser.readline().decode(errors="ignore").strip()
    log.info("ESP32: %r", ready)

    # ── Camera ────────────────────────────────────────────────────────────────
    log.info("Opening picamera2  %d×%d", FRAME_W, FRAME_H)
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
    ))
    cam.start()
    time.sleep(0.5)   # let AGC/AWB settle

    det = FaceDetector()

    # ── State ─────────────────────────────────────────────────────────────────
    # pan_deg is the absolute angle we tell the ESP32 to target.
    # We update it incrementally each frame based on face error.
    pan_deg     = PAN_CENTER
    last_tx     = 0.0
    last_face_t = 0.0   # monotonic time of last face detection

    log.info("Running.  Press Ctrl+C to stop.")
    log.info("  FLIP_FRAME=%s  FLIP_SERVO=%s  PAN_KP=%.3f  DEAD_ZONE=%dpx",
             FLIP_FRAME, FLIP_SERVO, PAN_KP, DEAD_ZONE_PX)

    try:
        while True:
            frame = cam.capture_array()          # RGB888
            if FLIP_FRAME:
                frame = cv2.flip(frame, 1)       # horizontal mirror

            result = det.detect(frame)
            now    = time.monotonic()

            if result is not None:
                cx, cy, fw, fh = result
                last_face_t = now

                error_x = cx - FRAME_W // 2     # px; +ve = face to the right

                if abs(error_x) > DEAD_ZONE_PX:
                    delta = PAN_KP * error_x
                    if FLIP_SERVO:
                        delta = -delta
                    pan_deg = float(np.clip(pan_deg + delta, PAN_MIN, PAN_MAX))

                if now - last_tx >= _SEND_DT:
                    ser.write(f"F{pan_deg:.1f}\n".encode())
                    last_tx = now

            else:
                # Send "N" for NO_FACE_SEND_S seconds after losing the face,
                # then go quiet and let the ESP32 fall back on its own.
                since_face = now - last_face_t
                if since_face < NO_FACE_SEND_S:
                    if now - last_tx >= _SEND_DT:
                        ser.write(b"N\n")
                        last_tx = now

    except KeyboardInterrupt:
        log.info("Interrupted — shutting down")
    finally:
        try:
            ser.write(b"N\n")
            ser.close()
        except Exception:
            pass
        cam.stop()
        log.info("Done.")


if __name__ == "__main__":
    main()
