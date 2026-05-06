"""
Runtime mode + all tunable tracking parameters in one place.

── Mode selection ────────────────────────────────────────────────────────────
Set DEVICE_MODE=pi     on the Raspberry Pi 5  (pi camera + ESP32 servos)
Set DEVICE_MODE=laptop on a laptop            (browser webcam, simulated servos)

If DEVICE_MODE is unset, auto-detect: picamera2 importable → pi, else laptop.

── How to tune tracking speed / smoothness ──────────────────────────────────
All the numbers that matter live in the "TRACKING TUNING" section below.
Change them here — no need to touch tracker.py or servo.py.

Quick cheat-sheet:
  Tracking too slow   → increase PID_KP  (e.g. 0.012 → 0.020)
  Tracking too fast   → decrease PID_KP  (e.g. 0.012 → 0.008)
  Servo oscillates    → increase PID_KD  (e.g. 0.003 → 0.006)
  Jitter at centre    → increase DEAD_ZONE_PX (e.g. 30 → 40)
  Movement still jumpy→ decrease SLEW_MAX_DEG (e.g. 2.0 → 1.0)
  Movement too laggy  → increase SLEW_MAX_DEG (e.g. 2.0 → 3.5)
"""

import os


# ─────────────────────────────────────────────────────────────────────────────
# Mode selection (do not change this section)
# ─────────────────────────────────────────────────────────────────────────────

def _auto_detect_mode() -> str:
    try:
        import sys
        if "/usr/lib/python3/dist-packages" not in sys.path:
            sys.path.append("/usr/lib/python3/dist-packages")
        import picamera2  # noqa: F401
        return "pi"
    except Exception:
        return "laptop"


MODE: str = os.environ.get("DEVICE_MODE", "").lower() or _auto_detect_mode()
if MODE not in ("pi", "laptop"):
    raise ValueError(f"DEVICE_MODE must be 'pi' or 'laptop', got: {MODE!r}")

IS_PI:     bool = MODE == "pi"
IS_LAPTOP: bool = MODE == "laptop"

# Mirror picamera2 output so the servo follows the face correctly and the
# browser view matches the laptop browser-feed path (which mirrors in JS).
MIRROR_PI_FRAME: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# TRACKING TUNING — edit these freely
# ─────────────────────────────────────────────────────────────────────────────

# ── Dead zone ─────────────────────────────────────────────────────────────────
# Face must be this many pixels off-centre before the servo moves at all.
# Prevents micro-jitter when the face bbox wobbles around centre.
# Tracker always works on a 640×480 frame.
#   30px = ~4.7% of frame width — recommended default
#   40px = more settled, less twitchy on noisy detection
#   20px = tight, only if mount + lighting are rock-solid
DEAD_ZONE_PX: int = 30

# ── PID gains ─────────────────────────────────────────────────────────────────
# PID_KP  — proportional gain: how hard the servo reacts to offset
#            higher = faster tracking, lower = smoother but slower
# PID_KD  — derivative gain: dampens overshoot / oscillation
#            increase if the servo hunts back and forth around the face
# PID_KI  — integral gain: corrects persistent steady-state error
#            leave at 0.0 until KP/KD are stable on real hardware
PID_KP: float = 0.012
PID_KD: float = 0.003
PID_KI: float = 0.0

# ── Output limits (degrees per tick = slew rate at PID level) ────────────────
# Maximum angle delta the PID controller can request in a single tracker tick.
# At 20 fps:  1.5 °/tick × 20 = 30 °/sec.  Previous default was 5 °/tick = 100 °/sec.
# Increase if tracking feels laggy; decrease if it still feels jerky.
PID_OUTPUT_LIMIT_PAN:  float = 1.5   # degrees / tick, pan axis
PID_OUTPUT_LIMIT_TILT: float = 1.2   # degrees / tick, tilt axis (smaller range)

# ── Slew rate (hard cap in servo.py, independent of PID) ─────────────────────
# Belt-and-suspenders: even if PID output_limits are increased, the servo
# will never receive a command more than this many degrees from its current
# position per move() call.
# At 20 fps:  2.0 °/call × 20 = 40 °/sec max physical travel.
SLEW_MAX_DEG: float = 2.0

# ── Tracker loop rate ─────────────────────────────────────────────────────────
TARGET_FPS: int = 20
