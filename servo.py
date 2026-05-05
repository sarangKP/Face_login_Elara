"""
Servo abstraction — identical public API for real hardware and simulation.

Hardware path: Python → USB serial → ESP32 → PWM → servos
  1. Flash esp32_servo/ with PlatformIO (open esp32_servo/ folder → Upload)
  2. Wire pan  servo signal → ESP32 GPIO 13
     Wire tilt servo signal → ESP32 GPIO 12
     Power servos from a separate 5 V supply, not the ESP32 3.3 V pin
  3. Install pyserial: uv add pyserial   (or: pip install pyserial)
  4. Set SIMULATE = False and set SERIAL_PORT to your device, then run

Finding the ESP32 port on Linux:
  ls /dev/ttyUSB*   # CP2102 / CH340 adapters
  ls /dev/ttyACM*   # native USB CDC (rare on ESP32 DevKit)
  dmesg | tail -20  # check kernel message after plugging in

Protocol sent to ESP32 over serial (115200 baud, newline-terminated):
  "P<pan_deg> T<tilt_deg>\n"   e.g. "P105.3 T72.0\n"
"""

import threading
import logging
import glob

log = logging.getLogger(__name__)

# ── Toggle this one flag to go live ──────────────────────────────────────────
SIMULATE = False          # False → real ESP32 serial control

# ── Serial port ──────────────────────────────────────────────────────────────
# Set explicitly or leave as None to auto-detect the first ttyUSB*/ttyACM*
SERIAL_PORT: str | None = None
BAUD_RATE = 115200

# ── Servo geometry ────────────────────────────────────────────────────────────
PAN_CENTER  = 90.0
TILT_CENTER = 90.0

PAN_LIMITS  = (30.0, 150.0)
TILT_LIMITS = (40.0, 140.0)


def _find_port() -> str:
    """Return the first available ESP32-like serial port on Linux."""
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    raise RuntimeError(
        "No ESP32 serial port found.\n"
        "Plug in the ESP32, then check: ls /dev/ttyUSB* /dev/ttyACM*\n"
        "If permission denied: sudo usermod -aG dialout $USER  (then re-login)"
    )


class ServoController:
    """
    Thread-safe servo controller.

    Real mode : sends angle commands over USB serial to the ESP32 firmware.
    Sim mode  : tracks angle state only — no serial port opened.
    """

    def __init__(self, simulate: bool = SIMULATE):
        self.simulate = simulate
        self._pan   = PAN_CENTER
        self._tilt  = TILT_CENTER
        self._lock  = threading.Lock()
        self._serial = None

        if not simulate:
            self._init_serial()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def pan(self) -> float:
        return self._pan

    @property
    def tilt(self) -> float:
        return self._tilt

    def move(self, pan: float, tilt: float) -> None:
        """Move both axes to the given angles, clamped to safe limits."""
        pan  = max(PAN_LIMITS[0],  min(PAN_LIMITS[1],  pan))
        tilt = max(TILT_LIMITS[0], min(TILT_LIMITS[1], tilt))

        with self._lock:
            self._pan  = pan
            self._tilt = tilt
            if not self.simulate and self._serial is not None:
                self._write(pan, tilt)

    def center(self) -> None:
        """Return both servos to the home position."""
        self.move(PAN_CENTER, TILT_CENTER)

    def shutdown(self) -> None:
        """Centre servos then close the serial port."""
        if self._serial is not None:
            self.center()
            import time; time.sleep(0.3)   # let the ESP32 finish the move
            self._serial.close()
            self._serial = None
            log.info("ServoController: serial port closed.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_serial(self) -> None:
        try:
            import serial
        except ImportError:
            raise ImportError(
                "pyserial not installed.\n"
                "Run: uv add pyserial   or   pip install pyserial"
            )

        port = SERIAL_PORT or _find_port()
        self._serial = serial.Serial(port, BAUD_RATE, timeout=2)

        import time; time.sleep(2)      # wait for ESP32 reset after DTR toggle

        # Drain the "READY" banner sent by the firmware
        self._serial.reset_input_buffer()

        self._write(PAN_CENTER, TILT_CENTER)
        log.info("ServoController: connected to ESP32 on %s — servos centred.", port)

    def _write(self, pan: float, tilt: float) -> None:
        """Send one command frame. Caller must hold self._lock."""
        cmd = f"P{pan:.1f} T{tilt:.1f}\n"
        try:
            self._serial.write(cmd.encode())
        except Exception as exc:
            log.warning("ServoController serial write error: %s", exc)
