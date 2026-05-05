"""
Face Tracking Engine — pan-tilt servo follows the detected face in real time.

Architecture
────────────
CameraManager  singleton — opens the Pi camera once, shared with face login.
               A background thread continuously captures frames into a buffer
               so callers always get the latest frame without blocking.

FaceTracker    daemon thread — reads buffer → detects face → PID → servo move.
               Exposes .state (TrackerState) for the FastAPI status endpoint.

Why Haar Cascade for tracking (not face_recognition)?
  face_recognition uses HOG + dlib: ~150-300 ms/frame — too slow for tracking.
  Haar Cascade at 320×240: ~10-30 ms/frame on Pi 5 — smooth 20+ fps tracking.
  We only need the bounding box here, not identity.

Usage (from FastAPI lifespan)
─────────────────────────────
    tracker = FaceTracker()
    tracker.start()
    ...
    state = tracker.state        # thread-safe snapshot
    frame_jpeg = state.annotated_jpeg
    tracker.stop()
"""

import cv2
import threading
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from pid   import PID
from servo import ServoController, SIMULATE, PAN_CENTER, TILT_CENTER

log = logging.getLogger(__name__)

# ── Detection parameters ──────────────────────────────────────────────────────
DETECT_W = 320            # downscale to this width before running Haar
DETECT_H = 240            # keeps detection fast regardless of camera resolution

# Haar tuning — trade-off between speed and false positives
SCALE_FACTOR    = 1.15    # pyramid scale per step; smaller = slower but finds more
MIN_NEIGHBOURS  = 4       # higher = fewer false positives, may miss quick moves
MIN_FACE_PX     = 40      # ignore detections smaller than this (noise, background)

DEADBAND_PX = 15          # pixel radius around centre — no servo move inside this zone

# ── PID gains ─────────────────────────────────────────────────────────────────
# Tuned conservatively for a standard SG90/MG996R on a Pi 5.
# If tracking is sluggish → increase kp slightly (e.g. 0.07 → 0.10).
# If mount oscillates  → increase kd (e.g. 0.003 → 0.006).
# Ki = 0 intentionally — add only after physical testing (integral windup hunts).
PAN_PID_GAINS  = dict(kp=0.03, ki=0.0, kd=0.002,
                      output_limits=(-5.0,  5.0),  deadband=DEADBAND_PX)
TILT_PID_GAINS = dict(kp=0.03, ki=0.0, kd=0.002,
                      output_limits=(-4.0,  4.0),  deadband=DEADBAND_PX)

TARGET_FPS = 20           # tracker loop rate (frames processed per second)


# ─────────────────────────────────────────────────────────────────────────────
# CameraManager — opens camera once, shared across the whole application
# ─────────────────────────────────────────────────────────────────────────────

class CameraManager:
    """
    Singleton camera source with two modes:

    Hardware mode (Pi / USB webcam not held by browser):
        picamera2 → libcamera stack (Pi Camera Module)
        OpenCV    → V4L2 fallback

    Browser-feed mode (laptop dev / single camera shared with browser):
        Browser captures via getUserMedia and POSTs frames to /track/feed.
        inject_frame() is called by that endpoint — no camera opened here.
        Automatically activated when no hardware camera is available.
    """

    _instance: Optional["CameraManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._init_camera()
                cls._instance = obj
        return cls._instance

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_camera(self) -> None:
        self._frame_lock   = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._cap          = None
        self._picam        = None
        self._running      = True

        # 1. Try Pi camera module
        try:
            from picamera2 import Picamera2
            self._picam = Picamera2()
            config = self._picam.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self._picam.configure(config)
            self._picam.start()
            self.source = "picamera2"
            log.info("CameraManager: picamera2 opened (640×480 RGB)")
            threading.Thread(target=self._capture_loop, daemon=True, name="CamCapture").start()
            return
        except Exception:
            pass

        # 2. Browser-feed mode — browser owns the webcam via getUserMedia.
        #    On a laptop there is only one camera. If the server opens it with
        #    OpenCV, the browser's getUserMedia gets locked out → black screen.
        #    So on non-Pi hardware we never touch the webcam from the server.
        #    The /track page captures frames via getUserMedia and POSTs them
        #    to /track/feed; the server annotates and streams them back.
        self.source = "browser"
        log.info("CameraManager: browser-feed mode — "
                 "/track page will send frames via POST /track/feed")

    def _capture_loop(self) -> None:
        """Hardware capture thread — only runs in picamera2 / opencv modes."""
        while self._running:
            frame = None
            try:
                if self._picam:
                    frame = self._picam.capture_array()
                elif self._cap:
                    ok, bgr = self._cap.read()
                    if ok:
                        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                log.warning("CameraManager capture error: %s", e)

            if frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
            time.sleep(0.008)

    # ── Public API ────────────────────────────────────────────────────────────

    def inject_frame(self, frame: np.ndarray) -> None:
        """
        Accept a frame from an external source (browser POST /track/feed).
        Works in any mode — also lets the browser override hardware in a pinch.
        """
        with self._frame_lock:
            self._latest_frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
        """Returns the latest RGB frame (copy). Thread-safe, non-blocking."""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    @property
    def is_browser_mode(self) -> bool:
        return self.source == "browser"

    def shutdown(self) -> None:
        self._running = False
        time.sleep(0.05)
        if self._picam:
            self._picam.stop()
        if self._cap:
            self._cap.release()
        CameraManager._instance = None
        log.info("CameraManager: shut down.")


# ─────────────────────────────────────────────────────────────────────────────
# TrackerState — thread-safe snapshot exposed to FastAPI endpoints
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackerState:
    pan:            float = PAN_CENTER
    tilt:           float = TILT_CENTER
    error_x:        float = 0.0   # + = face is right of centre
    error_y:        float = 0.0   # + = face is below centre
    face_found:     bool  = False
    fps:            float = 0.0
    annotated_jpeg: bytes = field(default_factory=bytes)


# ─────────────────────────────────────────────────────────────────────────────
# FaceTracker — the main tracking engine
# ─────────────────────────────────────────────────────────────────────────────

class FaceTracker:
    """
    Runs a daemon thread that:
      1. Reads frames from CameraManager
      2. Detects the largest face via Haar cascade (fast, CPU-only)
      3. Computes X/Y pixel error from frame centre
      4. Feeds error into PID controllers (one per axis)
      5. Applies resulting angle delta to ServoController
      6. Writes an annotated debug frame to TrackerState
    """

    def __init__(self, simulate: bool = SIMULATE):
        self.simulate    = simulate
        self._cam        = CameraManager()
        self._servo      = ServoController(simulate=simulate)
        self._pan_pid    = PID(**PAN_PID_GAINS)
        self._tilt_pid   = PID(**TILT_PID_GAINS)

        # Haar cascade — bundled with OpenCV, no download needed
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(cascade_path)
        if self._detector.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

        self._state      = TrackerState()
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running    = False

        # How many consecutive frames had no face — used to reset PIDs
        self._no_face_count = 0
        self._NO_FACE_RESET = 10   # reset PIDs after this many missed frames

        self._last_face_time = time.monotonic()
        self._NO_FACE_CENTER_S = 3.0   # seconds before returning to centre

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            log.warning("FaceTracker already running.")
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="FaceTracker"
        )
        self._thread.start()
        log.info("FaceTracker started (simulate=%s, target=%d fps)",
                 self.simulate, TARGET_FPS)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._servo.center()
        self._servo.shutdown()
        log.info("FaceTracker stopped.")

    # ── State access (thread-safe) ────────────────────────────────────────────

    @property
    def state(self) -> TrackerState:
        with self._state_lock:
            s = self._state
            return TrackerState(
                pan=s.pan, tilt=s.tilt,
                error_x=s.error_x, error_y=s.error_y,
                face_found=s.face_found, fps=s.fps,
                annotated_jpeg=s.annotated_jpeg,
            )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        interval = 1.0 / TARGET_FPS
        t_fps    = time.monotonic()

        while self._running:
            t0    = time.monotonic()
            frame = self._cam.get_frame()

            if frame is None:
                time.sleep(0.02)
                continue

            h, w     = frame.shape[:2]
            cx, cy   = w // 2, h // 2          # frame centre (target point)
            face_box = self._detect(frame)

            if face_box is not None:
                self._no_face_count = 0
                self._last_face_time = time.monotonic()
                fx, fy, fw, fh = face_box
                face_cx = fx + fw // 2
                face_cy = fy + fh // 2

                # Signed pixel error: positive → face is right of / below centre
                err_x =  float(face_cx - cx)
                err_y =  float(face_cy - cy)

                # PID → angle delta
                # Pan:  face right  (+err_x) → increase pan angle (turn right)
                # Tilt: face below  (+err_y) → decrease tilt angle (tilt down)
                d_pan  =  self._pan_pid.compute(err_x)
                d_tilt = -self._tilt_pid.compute(err_y)

                self._servo.move(
                    self._servo.pan  + d_pan,
                    self._servo.tilt + d_tilt,
                )

                if self.simulate:
                    log.debug(
                        "[SIM] pan=%.1f° tilt=%.1f° | err=(%+.0f, %+.0f)px",
                        self._servo.pan, self._servo.tilt, err_x, err_y,
                    )
            else:
                err_x = err_y = 0.0
                self._no_face_count += 1
                if self._no_face_count >= self._NO_FACE_RESET:
                    self._pan_pid.reset()
                    self._tilt_pid.reset()
                if time.monotonic() - self._last_face_time >= self._NO_FACE_CENTER_S:
                    self._servo.center()

            # Measure loop FPS
            now  = time.monotonic()
            fps  = 1.0 / max(now - t_fps, 1e-6)
            t_fps = now

            # Build annotated debug frame and encode as JPEG
            annotated = self._annotate(
                frame.copy(), face_box, cx, cy,
                self._servo.pan, self._servo.tilt, fps,
            )
            _, buf = cv2.imencode(
                ".jpg",
                cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )

            with self._state_lock:
                self._state.pan            = self._servo.pan
                self._state.tilt           = self._servo.tilt
                self._state.error_x        = round(err_x, 1)
                self._state.error_y        = round(err_y, 1)
                self._state.face_found     = face_box is not None
                self._state.fps            = round(fps, 1)
                self._state.annotated_jpeg = buf.tobytes()

            # Sleep just enough to hit TARGET_FPS
            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, interval - elapsed)
            if sleep_t:
                time.sleep(sleep_t)

    # ── Face detection ────────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> Optional[tuple]:
        """
        Detect the largest face in the frame.

        Downscales to 320×240 for speed, then maps the bounding box
        coordinates back to the original resolution.

        Returns (x, y, w, h) in original pixels, or None.
        """
        orig_h, orig_w = frame.shape[:2]

        # Downscale — Haar runs much faster at low resolution
        small = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_AREA)
        gray  = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

        # Histogram equalisation — improves detection under uneven lighting
        cv2.equalizeHist(gray, gray)

        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBOURS,
            minSize=(MIN_FACE_PX, MIN_FACE_PX),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if not len(faces):
            return None

        # Use the largest detected face (most likely the subject, not background)
        x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])

        # Scale bounding box back to original frame resolution
        scale_x = orig_w / DETECT_W
        scale_y = orig_h / DETECT_H
        return (
            int(x  * scale_x),
            int(y  * scale_y),
            int(fw * scale_x),
            int(fh * scale_y),
        )

    # ── Debug overlay ─────────────────────────────────────────────────────────

    def _annotate(
        self,
        frame:    np.ndarray,
        face_box: Optional[tuple],
        cx: int, cy: int,
        pan: float, tilt: float,
        fps: float,
    ) -> np.ndarray:
        """
        Draw tracking information on the frame:
          • Crosshair at frame centre (target)
          • Face bounding box + centre dot
          • Error vector (line from frame centre to face centre)
          • Pixel error labels
          • HUD with pan/tilt angles and FPS
          • Mini compass (bottom-right) showing servo position
        """
        h, w = frame.shape[:2]

        GREEN  = (0,   230, 100)
        YELLOW = (255, 210,   0)
        RED    = (255,  70,  70)
        WHITE  = (255, 255, 255)
        GRAY   = (110, 110, 110)

        # ── Centre crosshair ──
        cv2.line(frame, (cx - 22, cy), (cx + 22, cy), GRAY, 1)
        cv2.line(frame, (cx, cy - 22), (cx, cy + 22), GRAY, 1)
        cv2.circle(frame, (cx, cy), 3, GRAY, -1)

        if face_box is not None:
            fx, fy, fw, fh = face_box
            face_cx = fx + fw // 2
            face_cy = fy + fh // 2

            # Face bounding box
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), GREEN, 2)
            cv2.circle(frame,    (face_cx, face_cy), 4,        GREEN, -1)

            # Error vector — shows where the mount needs to move
            cv2.line(frame, (cx, cy), (face_cx, face_cy), YELLOW, 1)
            cv2.arrowedLine(frame, (cx, cy), (face_cx, face_cy),
                            YELLOW, 1, tipLength=0.15)

            # Error labels above the bounding box
            err_x = face_cx - cx
            err_y = face_cy - cy
            cv2.putText(frame, f"dx={err_x:+d}px",
                        (fx, max(fy - 22, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)
            cv2.putText(frame, f"dy={err_y:+d}px",
                        (fx, max(fy - 8, 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No face detected",
                        (cx - 60, cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)

        # ── HUD bar (bottom-left) ──
        mode   = "[SIM]" if self.simulate else "[LIVE]"
        hud    = f"{mode}  Pan:{pan:.1f}\u00b0  Tilt:{tilt:.1f}\u00b0  {fps:.0f} fps"
        cv2.putText(frame, hud, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, WHITE, 1, cv2.LINE_AA)

        # ── Mini compass (bottom-right) ──
        self._draw_compass(frame, pan, tilt, w - 52, h - 52, 38)

        return frame

    def _draw_compass(
        self, frame: np.ndarray,
        pan: float, tilt: float,
        cx: int, cy: int, r: int,
    ) -> None:
        """
        Two-needle compass in the corner showing current servo angles.
        Blue horizontal needle = pan offset from centre.
        Blue vertical needle   = tilt offset from centre.
        """
        BLUE  = (80,  150, 255)
        DGRAY = (70,   70,  70)

        cv2.circle(frame, (cx, cy), r,     DGRAY, 1)
        cv2.circle(frame, (cx, cy), r + 1, DGRAY, 1)   # double ring

        # Normalise: 90° centre → 0, limits → ±1
        pan_n  = (pan  - 90) / 60.0     # 30–150° range → -1..+1
        tilt_n = (tilt - 90) / 50.0     # 40–140° range → -1..+1

        pan_n  = max(-1.0, min(1.0, pan_n))
        tilt_n = max(-1.0, min(1.0, tilt_n))

        # Needle endpoints
        px = int(cx + pan_n  * r * 0.78)
        py = int(cy + tilt_n * r * 0.78)

        cv2.line(frame, (cx, cy), (px, cy), BLUE, 2)   # pan  (horizontal)
        cv2.line(frame, (cx, cy), (cx, py), BLUE, 2)   # tilt (vertical)
        cv2.circle(frame, (cx, cy), 3, BLUE, -1)
