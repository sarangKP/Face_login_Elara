"""
Runtime mode + all tunable parameters for Elara.

── Mode selection ────────────────────────────────────────────────────────────
Set DEVICE_MODE=pi     on the Raspberry Pi  (Pi camera + ESP32 servos)
Set DEVICE_MODE=laptop on a laptop          (browser webcam, simulated servos)

If DEVICE_MODE is unset, auto-detect: picamera2 importable → pi, else laptop.

── Jitter quick-fix cheat sheet ─────────────────────────────────────────────
Servo twitches at rest          → raise DEAD_ZONE_PX  (e.g. 50 → 70)
Servo hunts back and forth      → raise PID_KD        (e.g. 0.006 → 0.010)
Tracking too slow / face lost   → raise PID_KP        (e.g. 0.008 → 0.014)
Commands still too jumpy        → lower SLEW_MAX_DEG  (e.g. 1.2 → 0.8)
Movement too laggy              → raise SLEW_MAX_DEG  (e.g. 1.2 → 2.0)

── ESP32 constants (in may_9th_esp2pi/src/main.cpp) ─────────────────────────
These require a reflash. Documented here so everything is in one place.

  EMA_ALPHA          0.05–0.30   lower = smoother tracking, more lag
  RATE_DEG           1.0–4.0     max degrees/tick, lower = less jerky
  DEADBAND           0.5–2.0     higher = fewer servo commands, less buzz
  SOUND_ON_THRESH    300k–800k   higher = only loud clear speech triggers
  SOUND_OFF_THRESH   100k–500k   must be lower than ON (hysteresis gap)
  FACE_TIMEOUT_MS    300–1500    ms without a Pi command before "no face"
"""

import os


# ─────────────────────────────────────────────────────────────────────────────
# Mode selection (do not edit)
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

# Mirror picamera2 output so the servo follows the face correctly.
MIRROR_PI_FRAME: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# PI-SIDE TRACKING TUNING
# ─────────────────────────────────────────────────────────────────────────────

# ── Dead zone ─────────────────────────────────────────────────────────────────
# Face must be this many pixels off-centre before the servo moves at all.
# The Haar cascade bbox wobbles ±15–25 px even on a still face — set this
# above that wobble range or the servo will never settle.
# Frame is always 640×480.
#   Range : 20 (tight, rock-solid lighting) … 80 (very settled)
#   20 px = servo reacts to any small movement
#   50 px = recommended default — absorbs bbox wobble, tracks real movement
#   70 px = for very jittery detection or slow-moving subjects
DEAD_ZONE_PX: int = 30

# ── PID gains ─────────────────────────────────────────────────────────────────
# KP — proportional: how hard the servo reacts to offset
#       range 0.005 (very slow/smooth) … 0.025 (fast, may oscillate)
# KD — derivative: braking force — dampens overshoot and oscillation
#       range 0.001 (almost no braking) … 0.015 (heavy braking)
#       raise KD first when servo hunts; raise KP if tracking is too slow
# KI — integral: corrects steady-state error; leave 0.0 until KP/KD stable
PID_KP: float = 0.008   # was 0.012 — reduced for less jitter
PID_KD: float = 0.006   # was 0.003 — increased for more damping
PID_KI: float = 0.0

# ── Output limits ────────────────────────────────────────────────────────────
# Max angle change the PID may request per tracker tick (before slew cap).
# At 20 fps: 1.0 °/tick × 20 = 20 °/sec max PID-requested speed.
#   Range: 0.5 (very smooth, slow) … 3.0 (fast, may feel jerky)
PID_OUTPUT_LIMIT_PAN:  float = 1.0   # degrees / tick  (was 1.5)
PID_OUTPUT_LIMIT_TILT: float = 0.8   # degrees / tick  (was 1.2)

# ── Slew rate ────────────────────────────────────────────────────────────────
# Hard cap in servo.py: servo never jumps more than this per move() call,
# regardless of PID output. Belt-and-suspenders against sudden bbox jumps.
#   Range: 0.5 (very gentle) … 3.0 (no real limiting)
#   1.0 = smooth, good for elderly-care context
#   2.0 = original default, faster but jumpier
SLEW_MAX_DEG: float = 1.0   # was 2.0

# ── Tracker loop rate ─────────────────────────────────────────────────────────
# Higher FPS = smoother tracking but more CPU on the Pi.
#   Range: 10 (light CPU) … 30 (max useful for 640×480 Haar)
TARGET_FPS: int = 20


# ─────────────────────────────────────────────────────────────────────────────
# IDENTITY BRIDGE — face login ↔ tracker
# ─────────────────────────────────────────────────────────────────────────────

# How often the background worker re-runs face recognition.
# Don't go below ~2 s on Pi 5 — encoder runs overlap and cause lag.
#   Range: 2.0 (updates label faster) … 10.0 (barely checks)
IDENTITY_CHECK_INTERVAL_S: float = 3.0

# Face match threshold (lower = stricter, fewer false positives).
#   Range: 0.4 (strict) … 0.6 (lenient)
IDENTITY_TOLERANCE: float = 0.50

# Max faces passed to the dlib encoder per recognition pass.
IDENTITY_MAX_FACES: int = 5
