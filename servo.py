"""
Servo abstraction — identical public API for real hardware and simulation.

To go live:
  1. Set SIMULATE = False
  2. Wire pan servo  signal wire → BCM GPIO 17
     Wire tilt servo signal wire → BCM GPIO 18
     (Power servos from a separate 5 V supply, not the Pi 3.3 V pin)
  3. Install lgpio: pip install lgpio
  4. Restart the service — no daemon needed

Why lgpio, not pigpio or RPi.GPIO?
───────────────────────────────────
Pi 5 uses the RP1 southbridge chip for all GPIO. Neither pigpio nor RPi.GPIO
has a driver for RP1 — both fail silently or crash on Pi 5.

lgpio is the Pi Foundation's official replacement:
  • Works on Pi 1 through Pi 5 (RP1 aware)
  • No background daemon required (unlike pigpio which needs `sudo pigpiod`)
  • Provides tx_pwm() — software-timed PWM accurate enough for 50 Hz servos
  • Install: pip install lgpio   (or: sudo apt install python3-lgpio)

For absolute precision (hardware PWM):
  If you need rock-solid pulse widths (e.g. industrial servos), use GPIO 12
  or GPIO 13 which have dedicated hardware PWM on all Pi models. Access via
  /sys/class/pwm (sysfs). lgpio software PWM is fine for SG90/MG996R servos.
"""
import threading
import logging

log = logging.getLogger(__name__)

# ── Toggle this one flag to go live ──────────────────────────────────────────
SIMULATE = True          # False → real lgpio servo control

# ── GPIO (BCM numbering) ─────────────────────────────────────────────────────
PAN_PIN  = 17            # horizontal servo  (BCM 17 = physical pin 11)
TILT_PIN = 18            # vertical servo    (BCM 18 = physical pin 12)

# ── Servo geometry ────────────────────────────────────────────────────────────
PAN_CENTER  = 90.0       # degrees — home / resting position
TILT_CENTER = 90.0

PAN_LIMITS  = (30.0, 150.0)   # physical safe range — adjust for your bracket
TILT_LIMITS = (40.0, 140.0)   # narrower tilt to protect the camera ribbon cable

# ── PWM parameters ────────────────────────────────────────────────────────────
_SERVO_HZ  = 50          # standard servo PWM frequency (50 Hz = 20 ms period)
_PERIOD_US = 1_000_000 / _SERVO_HZ   # 20 000 µs

# Standard servo pulse range — measure your servo if behaviour is off
_PW_MIN_US = 500.0       # µs → 0°
_PW_MAX_US = 2500.0      # µs → 180°


def _angle_to_duty(angle: float) -> float:
    """
    Convert servo angle (degrees) → lgpio PWM duty cycle (0.0–100.0 %).

    lgpio tx_pwm() takes a percentage, not a raw pulse width.
    duty = pulse_width_µs / period_µs × 100
    """
    pw_us = _PW_MIN_US + (angle / 180.0) * (_PW_MAX_US - _PW_MIN_US)
    return (pw_us / _PERIOD_US) * 100.0   # e.g. 1500 µs → 7.5 %


class ServoController:
    """
    Thread-safe servo controller.

    Real mode : sends 50 Hz PWM via lgpio (Pi 5 compatible, no daemon needed)
    Sim mode  : tracks angle state only — zero GPIO calls
    """

    def __init__(self, simulate: bool = SIMULATE):
        self.simulate = simulate
        self._pan   = PAN_CENTER
        self._tilt  = TILT_CENTER
        self._lock  = threading.Lock()
        self._handle = None          # lgpio chip handle

        if not simulate:
            self._init_lgpio()

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
            if not self.simulate and self._handle is not None:
                self._write(PAN_PIN,  pan)
                self._write(TILT_PIN, tilt)

    def center(self) -> None:
        """Return both servos to the home position."""
        self.move(PAN_CENTER, TILT_CENTER)

    def shutdown(self) -> None:
        """
        Centre servos then stop PWM so they de-energise at rest.
        Leaving PWM active keeps the motor holding torque — wastes power,
        generates heat, and wears out plastic gears over time.
        """
        if self._handle is not None:
            import lgpio
            self.center()
            # duty=0 → no pulses → servo de-energises
            lgpio.tx_pwm(self._handle, PAN_PIN,  _SERVO_HZ, 0)
            lgpio.tx_pwm(self._handle, TILT_PIN, _SERVO_HZ, 0)
            lgpio.gpiochip_close(self._handle)
            self._handle = None
            log.info("ServoController: lgpio closed, PWM stopped.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_lgpio(self) -> None:
        try:
            import lgpio
        except ImportError:
            raise ImportError(
                "lgpio not installed.\n"
                "Run: pip install lgpio\n"
                "Or:  sudo apt install python3-lgpio"
            )

        self._handle = lgpio.gpiochip_open(0)   # /dev/gpiochip0

        # Claim both pins as outputs before starting PWM
        lgpio.gpio_claim_output(self._handle, PAN_PIN)
        lgpio.gpio_claim_output(self._handle, TILT_PIN)

        # Centre on startup — prevents the mount snapping to a random angle
        self._write(PAN_PIN,  PAN_CENTER)
        self._write(TILT_PIN, TILT_CENTER)
        log.info("ServoController: lgpio opened — servos centred (Pi 5 compatible).")

    def _write(self, pin: int, angle: float) -> None:
        import lgpio
        duty = _angle_to_duty(angle)
        lgpio.tx_pwm(self._handle, pin, _SERVO_HZ, duty)
