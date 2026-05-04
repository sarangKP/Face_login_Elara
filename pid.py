"""
Discrete PID controller with anti-windup and output clamping.

Designed for pan-tilt servo tracking where:
  - error  = pixel offset of face from frame centre
  - output = angle delta to apply to the servo this tick

Start with Ki=0 (PD only). Integral windup on a physical mount
causes hunting — only add Ki once Kp/Kd are tuned on real hardware.
"""
import time


class PID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limits: tuple = (-8.0, 8.0),
        integral_limits: tuple = (-15.0, 15.0),
        deadband: float = 0.0,
    ):
        """
        Args:
            kp, ki, kd       : gains
            output_limits    : clamp the final output (degrees/tick)
            integral_limits  : anti-windup clamp on the accumulated integral
            deadband         : errors smaller than this are treated as zero
                               (prevents micro-jitter when face is near centre)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits   = output_limits
        self.integral_limits = integral_limits
        self.deadband        = deadband

        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time: float | None = None

    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Call when tracking is lost and resumes — prevents integral kick."""
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = None

    def compute(self, error: float) -> float:
        """
        Feed the current error, get back an angle delta.

        Args:
            error: signed pixel offset (+ = face right of / below centre)

        Returns:
            Angle delta in degrees to add to the current servo position.
        """
        # Deadband — ignore tiny errors so the mount doesn't chase noise
        if abs(error) <= self.deadband:
            error = 0.0

        now = time.monotonic()
        dt  = (now - self._prev_time) if self._prev_time is not None else 0.033
        dt  = max(dt, 1e-4)          # guard division by zero on first tick
        self._prev_time = now

        # Proportional
        p = self.kp * error

        # Integral with anti-windup clamp
        self._integral += error * dt
        self._integral  = max(self.integral_limits[0],
                              min(self.integral_limits[1], self._integral))
        i = self.ki * self._integral

        # Derivative (rate of error change)
        d = self.kd * (error - self._prev_error) / dt
        self._prev_error = error

        output = p + i + d
        return max(self.output_limits[0], min(self.output_limits[1], output))
