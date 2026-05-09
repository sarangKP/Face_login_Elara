"""
Servo motor test tool — manual arrow-key control + live serial monitor
                        + IMU jitter meter (requires MPU6050 on servo arm).

  LEFT / RIGHT  →  pan servo  (−5° / +5°)
  UP   / DOWN   →  tilt servo (−5° / +5°)
  SHIFT + arrow →  fine step  (−1° / +1°)
  c             →  centre both servos
  q / ESC       →  quit

Run: python servo_test.py
     python servo_test.py --port /dev/ttyUSB0   (override auto-detect)
     python servo_test.py --step 2               (default step is 5°)
"""

import argparse
import curses
import glob
import math
import re
import sys
import threading
import time
from collections import deque

try:
    import serial
except ImportError:
    sys.exit("pyserial not installed — run: pip install pyserial")

# ── servo geometry (must match ESP32 firmware) ────────────────────────────────
PAN_CENTER  = 90.0
TILT_CENTER = 90.0
PAN_LIMITS  = (30.0, 150.0)
TILT_LIMITS = (40.0, 140.0)
BAUD        = 115200
STEP_COARSE = 5.0
STEP_FINE   = 1.0
SERIAL_LOG_LINES = 20

# Jitter thresholds (rad/s RMS) — tune to your servo after baseline testing
JITTER_OK   = 0.05   # below this → smooth (green)
JITTER_WARN = 0.20   # below this → some jitter (yellow)
                     # above      → replace servo (red)

_GYRO_RE = re.compile(
    r"GYRO gx=([0-9e.+-]+) gy=([0-9e.+-]+) gz=([0-9e.+-]+) rms=([0-9e.+-]+)"
)


def find_port() -> str:
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    raise RuntimeError(
        "No ESP32 port found.\n"
        "Plug in the ESP32 then try: ls /dev/ttyUSB* /dev/ttyACM*\n"
        "Or pass --port /dev/ttyUSBx explicitly."
    )


# ── serial reader thread ──────────────────────────────────────────────────────
class SerialMonitor:
    def __init__(self, port: str, baud: int, maxlines: int = SERIAL_LOG_LINES):
        self.ser       = serial.Serial(port, baud, timeout=0.1)
        self._buf      = deque(maxlen=maxlines)
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        self.imu_ok    = False          # set True when "IMU ok" received
        self.last_rms  = None           # latest jitter RMS from GYRO lines
        self.last_gyro = (0.0, 0.0, 0.0)
        self._thread   = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._stop.is_set():
            try:
                line = self.ser.readline()
                if not line:
                    continue
                decoded = line.decode("utf-8", errors="replace").rstrip()
                ts = time.strftime("%H:%M:%S")

                # track IMU presence
                if "IMU ok" in decoded:
                    self.imu_ok = True
                elif "IMU not found" in decoded:
                    self.imu_ok = False

                # parse GYRO lines — store but don't flood the log
                m = _GYRO_RE.search(decoded)
                if m:
                    gx, gy, gz, rms = (float(v) for v in m.groups())
                    self.last_gyro = (gx, gy, gz)
                    self.last_rms  = rms
                    # only add to log occasionally so it doesn't scroll too fast
                    # (GYRO arrives at 20 Hz; log every 10th = 2 Hz)
                    with self._lock:
                        # use a counter stored in deque length parity as a cheap tick
                        if len(self._buf) % 10 == 0:
                            self._buf.append(
                                f"[{ts}] GYRO  gx={gx:+.3f}  gy={gy:+.3f}  "
                                f"gz={gz:+.3f}  RMS={rms:.4f} rad/s"
                            )
                else:
                    with self._lock:
                        self._buf.append(f"[{ts}] {decoded}")

            except Exception as exc:
                with self._lock:
                    self._buf.append(f"[ERR] {exc}")

    def send(self, pan: float, tilt: float):
        cmd = f"P{pan:.1f} T{tilt:.1f}\n"
        self.ser.write(cmd.encode())
        ts = time.strftime("%H:%M:%S")
        with self._lock:
            self._buf.append(f"[{ts}] TX: {cmd.rstrip()}")

    def lines(self) -> list[str]:
        with self._lock:
            return list(self._buf)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        try:
            self.ser.close()
        except Exception:
            pass


# ── helpers ───────────────────────────────────────────────────────────────────
def _jitter_bar(rms: float, width: int) -> tuple[str, int]:
    """Return (bar_string, color_pair) for the given RMS jitter value."""
    if rms < JITTER_OK:
        label = "SMOOTH"
        pair  = 3   # green
    elif rms < JITTER_WARN:
        label = "SOME JITTER"
        pair  = 4   # yellow
    else:
        label = "REPLACE SERVO"
        pair  = 5   # red

    # bar scaled so JITTER_WARN fills ~75% of width
    filled = min(width, int(rms / JITTER_WARN * width * 0.75))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {rms:.4f} rad/s  {label}", pair


def _angle_bar(label: str, val: float, lo: float, hi: float, bar_w: int) -> str:
    span   = hi - lo
    filled = int((val - lo) / span * bar_w)
    bar    = "█" * filled + "░" * (bar_w - filled)
    return f"{label}: {val:6.1f}°  [{bar}]  {lo:.0f}–{hi:.0f}°"


# ── curses draw ───────────────────────────────────────────────────────────────
def draw(stdscr, pan: float, tilt: float, port: str, monitor: SerialMonitor,
         step: float, status: str):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    bar_w = max(10, w - 32)

    # header
    title = " SERVO TEST + JITTER METER "
    stdscr.addstr(0, max(0, (w - len(title)) // 2), title,
                  curses.A_BOLD | curses.A_REVERSE)

    stdscr.addstr(1, 2, f"Port: {port}   Baud: {BAUD}   Step: {step:.0f}° (Shift=1°)")

    # angle bars
    stdscr.addstr(3, 2, _angle_bar("PAN ", pan,  *PAN_LIMITS,  bar_w))
    stdscr.addstr(4, 2, _angle_bar("TILT", tilt, *TILT_LIMITS, bar_w))

    # jitter meter
    row = 5
    if monitor.imu_ok and monitor.last_rms is not None:
        rms = monitor.last_rms
        gx, gy, gz = monitor.last_gyro
        bar_str, pair = _jitter_bar(rms, bar_w)
        stdscr.addstr(row,     2, "JITTER: ", curses.A_BOLD)
        try:
            stdscr.addstr(row, 10, bar_str, curses.color_pair(pair) | curses.A_BOLD)
        except curses.error:
            pass
        stdscr.addstr(row + 1, 2,
                      f"  gyro  gx={gx:+.4f}  gy={gy:+.4f}  gz={gz:+.4f}  rad/s")
        row += 2
    elif not monitor.imu_ok:
        stdscr.addstr(row, 2,
                      "JITTER: no IMU — mount MPU6050 on servo arm (SDA→21, SCL→22)",
                      curses.color_pair(4))
        row += 1

    # status + controls
    if status:
        stdscr.addstr(row, 2, f"Status: {status}", curses.A_BOLD)
    row += 1
    stdscr.addstr(row, 2,
                  "←→: pan   ↑↓: tilt   Shift+arrow: fine   c: centre   q/ESC: quit")
    row += 1

    # serial monitor
    stdscr.addstr(row, 0, "─" * w)
    row += 1
    stdscr.addstr(row, 2, "SERIAL MONITOR", curses.A_BOLD)
    row += 1

    available = h - row - 1
    log = monitor.lines()
    shown = log[-available:]
    for i, line in enumerate(shown):
        y = row + i
        if y >= h - 1:
            break
        try:
            if "TX:" in line:
                attr = curses.color_pair(2)   # yellow
            elif "GYRO" in line:
                attr = curses.color_pair(6)   # magenta
            elif "CLAMPED" in line or "ERR" in line:
                attr = curses.color_pair(5)   # red
            else:
                attr = curses.color_pair(1)   # cyan
            stdscr.addstr(y, 2, line[:w - 3], attr)
        except curses.error:
            pass

    stdscr.refresh()


# ── main loop ─────────────────────────────────────────────────────────────────
def run(stdscr, args):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN,    -1)   # RX lines
    curses.init_pair(2, curses.COLOR_YELLOW,  -1)   # TX lines
    curses.init_pair(3, curses.COLOR_GREEN,   -1)   # smooth
    curses.init_pair(4, curses.COLOR_YELLOW,  -1)   # some jitter
    curses.init_pair(5, curses.COLOR_RED,     -1)   # bad jitter / errors
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)   # GYRO log lines

    port = args.port or find_port()
    step = float(args.step)

    monitor = SerialMonitor(port, BAUD)
    time.sleep(2)
    monitor.ser.reset_input_buffer()

    pan, tilt = PAN_CENTER, TILT_CENTER
    monitor.send(pan, tilt)
    status = "centred"

    try:
        while True:
            key = stdscr.getch()
            moved = False

            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('c'), ord('C')):
                pan, tilt = PAN_CENTER, TILT_CENTER
                status = "centred"
                moved = True
            elif key == curses.KEY_LEFT:
                pan = max(PAN_LIMITS[0], pan - step)
                status = f"pan ← {pan:.1f}°"
                moved = True
            elif key == curses.KEY_RIGHT:
                pan = min(PAN_LIMITS[1], pan + step)
                status = f"pan → {pan:.1f}°"
                moved = True
            elif key == curses.KEY_UP:
                tilt = max(TILT_LIMITS[0], tilt - step)
                status = f"tilt ↑ {tilt:.1f}°"
                moved = True
            elif key == curses.KEY_DOWN:
                tilt = min(TILT_LIMITS[1], tilt + step)
                status = f"tilt ↓ {tilt:.1f}°"
                moved = True
            elif key == curses.KEY_SLEFT:
                pan = max(PAN_LIMITS[0], pan - STEP_FINE)
                status = f"pan ← (fine) {pan:.1f}°"
                moved = True
            elif key == curses.KEY_SRIGHT:
                pan = min(PAN_LIMITS[1], pan + STEP_FINE)
                status = f"pan → (fine) {pan:.1f}°"
                moved = True
            elif key == 337:   # Shift+Up
                tilt = max(TILT_LIMITS[0], tilt - STEP_FINE)
                status = f"tilt ↑ (fine) {tilt:.1f}°"
                moved = True
            elif key == 336:   # Shift+Down
                tilt = min(TILT_LIMITS[1], tilt + STEP_FINE)
                status = f"tilt ↓ (fine) {tilt:.1f}°"
                moved = True

            if moved:
                monitor.send(pan, tilt)

            draw(stdscr, pan, tilt, port, monitor, step, status)
            time.sleep(0.02)   # 50 Hz UI refresh

    finally:
        monitor.send(PAN_CENTER, TILT_CENTER)
        time.sleep(0.3)
        monitor.close()


def main():
    parser = argparse.ArgumentParser(description="Servo motor test tool")
    parser.add_argument("--port", default=None,
                        help="Serial port (default: auto-detect)")
    parser.add_argument("--step", default=5, type=float,
                        help="Coarse step size in degrees (default: 5)")
    args = parser.parse_args()

    try:
        curses.wrapper(run, args)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
