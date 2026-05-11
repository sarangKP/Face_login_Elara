"""
Servo abstraction — identical public API for real hardware and simulation.

Hardware path: Python → USB serial → ESP32 → PWM → servos
  1. Flash esp32_servo/ with PlatformIO (open esp32_servo/ folder → Upload)
  2. Wire pan  servo signal → ESP32 GPIO 13
     Wire tilt servo signal → ESP32 GPIO 12
     Power servos from a separate 5 V supply, not the ESP32 3.3 V pin
  3. Install pyserial: uv add pyserial   (or: pip install pyserial)
  4. Run with DEVICE_MODE=pi (or on Pi hardware — auto-detected)

Finding the ESP32 port on Linux:
  ls /dev/ttyUSB*   # CP2102 / CH340 adapters
  ls /dev/ttyACM*   # native USB CDC (rare on ESP32 DevKit)
  dmesg | tail -20  # check kernel message after plugging in

Protocol sent to ESP32 over serial (115200 baud, newline-terminated):
  "F<pan_deg>\n"   e.g. "F105.3\n"

Tuning
──────
SLEW_MAX_DEG is read from config.py — change it there.
"""

import threading
import logging
import glob

import config

log = logging.getLogger(__name__)

# Always try to drive real servos. If no ESP32 is attached, _init_serial()
# raises and ServoController falls back to simulate mode automatically — so
# laptop-without-hardware still works, and laptop-with-hardware just works.
SIMULATE = False


# ── Serial port ──────────────────────────────────────────────────────────────
# Set explicitly or leave as None to auto-detect the first ttyUSB*/ttyACM*
SERIAL_PORT: str | None = "/dev/ttyUSB1"  # main ESP (eyes ESP is /dev/ttyUSB0)
BAUD_RATE = 115200

# ── Servo geometry ────────────────────────────────────────────────────────────
PAN_CENTER  = 90.0
TILT_CENTER = 90.0

PAN_LIMITS  = (30.0, 150.0)
TILT_LIMITS = (30.0, 150.0)

# Slew rate — read from config so it's tunable from one place
SLEW_MAX_DEG: float = config.SLEW_MAX_DEG


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

    In both modes move() applies a slew rate limit (SLEW_MAX_DEG per call)
    so the servo never jumps more than a small increment per tracker tick,
    regardless of how large the PID output happens to be.
    """

    def __init__(self, simulate: bool = SIMULATE):
        self.simulate = simulate
        self._pan        = PAN_CENTER
        self._tilt       = TILT_CENTER
        self._lock       = threading.Lock()
        self._serial     = None
        self._esp32_angle: float | None = None  # last angle reported by ESP32

        if not simulate:
            try:
                self._init_serial()
            except Exception as e:
                # No ESP32 attached → degrade to simulate so tracking can still
                # run (camera + face detection + virtual angle state). Servos
                # will simply not move; the rest of the system is unaffected.
                log.warning("ServoController: serial unavailable (%s) — "
                            "falling back to simulate mode.", e)
                self.simulate = True
                self._serial = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def pan(self) -> float:
        return self._pan

    @property
    def tilt(self) -> float:
        return self._tilt

    def move(self, pan: float, tilt: float) -> None:
        """
        Move both axes toward the given angles, slew-rate limited.

        Two-stage clamping:
          1. Slew limit  — requested angle must be within ±SLEW_MAX_DEG of the
                           current position. Prevents violent jumps even if the
                           PID produces a large delta in a single tick.
          2. Hard limits — clamp to the safe mechanical travel range.

        Works identically in simulate and real modes — in simulate mode the
        slew just constrains the virtual angle state shown in the HUD/compass.
        """
        with self._lock:
            # Stage 1 — slew rate: never move more than SLEW_MAX_DEG per call
            pan  = max(self._pan  - SLEW_MAX_DEG, min(self._pan  + SLEW_MAX_DEG, pan))
            tilt = max(self._tilt - SLEW_MAX_DEG, min(self._tilt + SLEW_MAX_DEG, tilt))

            # Stage 2 — mechanical limits
            pan  = max(PAN_LIMITS[0],  min(PAN_LIMITS[1],  pan))
            tilt = max(TILT_LIMITS[0], min(TILT_LIMITS[1], tilt))

            self._pan  = pan
            self._tilt = tilt
            if not self.simulate and self._serial is not None:
                self._write(pan, tilt)

    def center(self) -> None:
        """Return both servos to the home position (bypasses slew rate)."""
        # Bypass slew for centering — immediate hard return on stop/shutdown,
        # not a slow crawl back to centre after tracking ends.
        pan  = max(PAN_LIMITS[0],  min(PAN_LIMITS[1],  PAN_CENTER))
        tilt = max(TILT_LIMITS[0], min(TILT_LIMITS[1], TILT_CENTER))
        with self._lock:
            self._pan  = pan
            self._tilt = tilt
            if not self.simulate and self._serial is not None:
                self._write(pan, tilt)

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
        threading.Thread(target=self._serial_reader, daemon=True, name="ServoReader").start()

    def _serial_reader(self) -> None:
        """Read angle reports (A{angle}\n) sent by the ESP32 every 200ms."""
        buf = ""
        while self._serial is not None:
            try:
                n = self._serial.in_waiting
                if n:
                    buf += self._serial.read(n).decode(errors="ignore")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if line.startswith("A"):
                            try:
                                with self._lock:
                                    self._esp32_angle = float(line[1:])
                            except ValueError:
                                pass
            except Exception:
                pass
            import time; time.sleep(0.05)

    def sync_pan_to_esp32(self) -> None:
        """Sync Python's pan state to the angle last reported by the ESP32.
        Call this when re-acquiring a face after SOUND mode so the next
        absolute move() command is anchored to the real servo position."""
        with self._lock:
            if self._esp32_angle is not None:
                self._pan = self._esp32_angle

    def send_face_x(self, face_cx: int, frame_w: int) -> None:
        """Send face x position so ESP can zone-map directly from pixel position.
        Protocol: F<0-100>\n  (face centre as % of frame width)
        """
        if not self.simulate and self._serial is not None:
            pct = int(face_cx * 100 / frame_w)
            try:
                self._serial.write(f"F{pct}\n".encode())
            except Exception as exc:
                log.warning("ServoController serial write error: %s", exc)

    def _write(self, pan: float, tilt: float) -> None:
        """Send one command frame. Caller must hold self._lock."""
        cmd = f"P{pan:.1f}T{tilt:.1f}\n"
        try:
            self._serial.write(cmd.encode())
        except Exception as exc:
            log.warning("ServoController serial write error: %s", exc)
