"""
Runtime mode for the project.

Set DEVICE_MODE=pi    on the Raspberry Pi 5 (production: pi camera + ESP32 servos)
Set DEVICE_MODE=laptop on a laptop          (development: browser webcam, simulated servos)

If DEVICE_MODE is unset, we auto-detect: picamera2 importable → pi, else laptop.

Other knobs (rarely touched):
    MIRROR_PI_FRAME   Horizontally flip pi camera frames so the user sees themselves
                      mirrored (matches a typical selfie-cam UX) AND the PID error
                      sign matches the browser-feed path (which mirrors in JS).
                      Without this, the servo follows the face the wrong way.
"""

import os


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

IS_PI: bool = MODE == "pi"
IS_LAPTOP: bool = MODE == "laptop"

# Mirror picamera2 output. Fixes inverted servo direction and matches laptop UX.
MIRROR_PI_FRAME: bool = True
