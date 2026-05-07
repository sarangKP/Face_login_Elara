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

Identity bridge
───────────────
After /login succeeds, set_identified_person(name) is called. A background
recognition worker then runs face_recognition every IDENTITY_CHECK_INTERVAL_S
seconds on a snapshot of the latest frame. It publishes:
  tracked_label      — name | "Unknown" for the largest (tracked) face
  identified_present — True if the named person appears anywhere in frame

The main tracking loop is never blocked by recognition.

Tuning
──────
All tunable parameters live in config.py.
"""

import cv2
import json
import threading
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import face_recognition

import config
from pid   import PID
from servo import ServoController, SIMULATE, PAN_CENTER, TILT_CENTER

log = logging.getLogger(__name__)

# Tracker re-reads this when login hands it a name — keeps the contract a
# single string and avoids state drift if the DB is updated between calls.
_FACES_DB = Path("faces/db.json")

# ── Detection parameters ──────────────────────────────────────────────────────
DETECT_W = 320
DETECT_H = 240

SCALE_FACTOR   = 1.15
MIN_NEIGHBOURS = 4
MIN_FACE_PX    = 40

# ── Pull all tuning knobs from config.py ──────────────────────────────────────
DEAD_ZONE_PX = config.DEAD_ZONE_PX

PAN_PID_GAINS  = dict(
    kp=config.PID_KP,
    ki=config.PID_KI,
    kd=config.PID_KD,
    output_limits=(-config.PID_OUTPUT_LIMIT_PAN,  config.PID_OUTPUT_LIMIT_PAN),
    deadband=DEAD_ZONE_PX,
)
TILT_PID_GAINS = dict(
    kp=config.PID_KP,
    ki=config.PID_KI,
    kd=config.PID_KD,
    output_limits=(-config.PID_OUTPUT_LIMIT_TILT, config.PID_OUTPUT_LIMIT_TILT),
    deadband=DEAD_ZONE_PX,
)

TARGET_FPS                = config.TARGET_FPS
IDENTITY_CHECK_INTERVAL_S = config.IDENTITY_CHECK_INTERVAL_S
IDENTITY_TOLERANCE        = config.IDENTITY_TOLERANCE
IDENTITY_MAX_FACES        = config.IDENTITY_MAX_FACES


# ─────────────────────────────────────────────────────────────────────────────
# CameraManager — opens camera once, shared across the whole application
# ─────────────────────────────────────────────────────────────────────────────

class CameraManager:
    """
    Singleton camera source with two modes:

    Hardware mode (Pi):
        picamera2 → libcamera stack (Pi Camera Module)

    Browser-feed mode (laptop dev):
        Browser captures via getUserMedia and POSTs frames to /track/feed.
        inject_frame() is called by that endpoint — no camera opened here.
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

    def _init_camera(self) -> None:
        self._frame_lock   = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._cap          = None
        self._picam        = None
        self._running      = True

        import sys
        if "/usr/lib/python3/dist-packages" not in sys.path:
            sys.path.append("/usr/lib/python3/dist-packages")

        if config.IS_PI:
            try:
                from picamera2 import Picamera2
                self._picam = Picamera2()
                cam_cfg = self._picam.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                self._picam.configure(cam_cfg)
                self._picam.start()
                self.source = "picamera2"
                log.info("CameraManager: picamera2 opened (640×480)")
                threading.Thread(target=self._capture_loop, daemon=True, name="CamCapture").start()
                return
            except Exception as e:
                log.warning("picamera2 failed: %s", e)

        self.source = "browser"
        log.info("CameraManager: browser-feed mode — "
                 "/track page will send frames via POST /track/feed")

    def _capture_loop(self) -> None:
        while self._running:
            frame = None
            try:
                if self._picam:
                    # libcamera RGB888 is actually BGR byte order — convert.
                    raw = self._picam.capture_array()
                    frame = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    if config.MIRROR_PI_FRAME:
                        frame = cv2.flip(frame, 1)
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

    def inject_frame(self, frame: np.ndarray) -> None:
        with self._frame_lock:
            self._latest_frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
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
    error_x:        float = 0.0
    error_y:        float = 0.0
    face_found:     bool  = False
    fps:            float = 0.0
    annotated_jpeg: bytes = field(default_factory=bytes)

    # ── Identity bridge ──
    identified_name:     Optional[str]   = None   # who login told us to find
    tracked_label:       Optional[str]   = None   # name | "Unknown" | None
    identified_present:  bool            = False  # named person anywhere in frame
    tracked_position:    Optional[tuple] = None   # (cx, cy) of tracked bbox
    last_recognition_ts: float           = 0.0   # epoch seconds of last check


# ─────────────────────────────────────────────────────────────────────────────
# FaceTracker — the main tracking engine
# ─────────────────────────────────────────────────────────────────────────────

class FaceTracker:
    """
    Runs two daemon threads:
      _loop              — 20 fps Haar detection → PID → servo
      _recognition_loop  — every N seconds: dlib encode → identity check
    """

    def __init__(self, simulate: bool = SIMULATE):
        self.simulate    = simulate
        self._cam        = CameraManager()
        self._servo      = ServoController(simulate=simulate)
        self._pan_pid    = PID(**PAN_PID_GAINS)
        self._tilt_pid   = PID(**TILT_PID_GAINS)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(cascade_path)
        if self._detector.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

        self._state      = TrackerState()
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running    = False

        self._no_face_count    = 0
        self._NO_FACE_RESET    = 10
        self._last_face_time   = time.monotonic()
        self._NO_FACE_CENTER_S = 3.0

        # ── Identity bridge ──────────────────────────────────────────────────
        self._identity_lock = threading.Lock()
        self._identified_name: Optional[str] = None
        self._identified_encodings: list[np.ndarray] = []

        self._latest_box_lock = threading.Lock()
        self._latest_face_box: Optional[tuple] = None

        self._recog_thread: Optional[threading.Thread] = None

    # ── Identity API ─────────────────────────────────────────────────────────

    def set_identified_person(self, name: str) -> bool:
        """
        Tell the tracker who just logged in. Loads their encodings from
        faces/db.json. Returns False if the name isn't in the DB.
        Called by /login on success.
        """
        try:
            db = json.loads(_FACES_DB.read_text()) if _FACES_DB.exists() else {}
        except Exception as e:
            log.warning("Could not read faces DB: %s", e)
            return False

        if name not in db:
            return False

        encodings = [np.array(e, dtype=np.float64) for e in db[name]]
        with self._identity_lock:
            self._identified_name      = name
            self._identified_encodings = encodings
        with self._state_lock:
            self._state.identified_name    = name
            self._state.tracked_label      = None
            self._state.identified_present = False
        log.info("Identity set: '%s' (%d encodings)", name, len(encodings))
        return True

    def clear_identified_person(self) -> None:
        """Forget the logged-in person; recognition worker idles."""
        with self._identity_lock:
            self._identified_name      = None
            self._identified_encodings = []
        with self._state_lock:
            self._state.identified_name    = None
            self._state.tracked_label      = None
            self._state.identified_present = False
        log.info("Identity cleared.")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            log.warning("FaceTracker already running.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="FaceTracker")
        self._thread.start()
        self._recog_thread = threading.Thread(
            target=self._recognition_loop, daemon=True, name="FaceRecognition"
        )
        self._recog_thread.start()
        log.info("FaceTracker started (simulate=%s, %d fps, recognition every %.1fs)",
                 self.simulate, TARGET_FPS, IDENTITY_CHECK_INTERVAL_S)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._recog_thread:
            self._recog_thread.join(timeout=3.0)
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
                identified_name=s.identified_name,
                tracked_label=s.tracked_label,
                identified_present=s.identified_present,
                tracked_position=s.tracked_position,
                last_recognition_ts=s.last_recognition_ts,
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

            h, w   = frame.shape[:2]
            cx, cy = w // 2, h // 2
            face_box = self._detect(frame)

            if face_box is not None:
                self._no_face_count  = 0
                self._last_face_time = time.monotonic()
                fx, fy, fw, fh = face_box
                face_cx = fx + fw // 2
                face_cy = fy + fh // 2

                err_x = float(face_cx - cx)
                err_y = float(face_cy - cy)

                d_pan  =  self._pan_pid.compute(err_x)
                d_tilt = -self._tilt_pid.compute(err_y)
                self._servo.move(self._servo.pan + d_pan, self._servo.tilt + d_tilt)

                if self.simulate:
                    log.debug("[SIM] pan=%.1f° tilt=%.1f° | err=(%+.0f, %+.0f)px",
                              self._servo.pan, self._servo.tilt, err_x, err_y)
            else:
                err_x = err_y = 0.0
                self._no_face_count += 1
                if self._no_face_count >= self._NO_FACE_RESET:
                    self._pan_pid.reset()
                    self._tilt_pid.reset()
                if time.monotonic() - self._last_face_time >= self._NO_FACE_CENTER_S:
                    self._servo.center()

            now   = time.monotonic()
            fps   = 1.0 / max(now - t_fps, 1e-6)
            t_fps = now

            with self._state_lock:
                identified_name    = self._state.identified_name
                tracked_label      = self._state.tracked_label
                identified_present = self._state.identified_present

            annotated = self._annotate(
                frame.copy(), face_box, cx, cy,
                self._servo.pan, self._servo.tilt, fps,
                identified_name, tracked_label, identified_present,
            )
            _, buf = cv2.imencode(
                ".jpg",
                cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )

            with self._latest_box_lock:
                self._latest_face_box = face_box

            tracked_pos = None
            if face_box is not None:
                fx, fy, fw, fh = face_box
                tracked_pos = (fx + fw // 2, fy + fh // 2)

            with self._state_lock:
                self._state.pan              = self._servo.pan
                self._state.tilt             = self._servo.tilt
                self._state.error_x          = round(err_x, 1)
                self._state.error_y          = round(err_y, 1)
                self._state.face_found       = face_box is not None
                self._state.fps              = round(fps, 1)
                self._state.annotated_jpeg   = buf.tobytes()
                self._state.tracked_position = tracked_pos
                if face_box is None:
                    self._state.tracked_label = None

            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, interval - elapsed)
            if sleep_t:
                time.sleep(sleep_t)

    # ── Face detection ────────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> Optional[tuple]:
        orig_h, orig_w = frame.shape[:2]
        small = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_AREA)
        gray  = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
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

        x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        scale_x = orig_w / DETECT_W
        scale_y = orig_h / DETECT_H
        return (int(x * scale_x), int(y * scale_y), int(fw * scale_x), int(fh * scale_y))

    # ── Debug overlay ─────────────────────────────────────────────────────────

    def _annotate(
        self,
        frame: np.ndarray,
        face_box: Optional[tuple],
        cx: int, cy: int,
        pan: float, tilt: float,
        fps: float,
        identified_name:    Optional[str] = None,
        tracked_label:      Optional[str] = None,
        identified_present: bool = False,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        GREEN  = (0,   230, 100)
        YELLOW = (255, 210,   0)
        RED    = (255,  70,  70)
        WHITE  = (255, 255, 255)
        GRAY   = (110, 110, 110)
        BLUE   = (80,  150, 255)

        cv2.line(frame, (cx - 22, cy), (cx + 22, cy), GRAY, 1)
        cv2.line(frame, (cx, cy - 22), (cx, cy + 22), GRAY, 1)
        cv2.circle(frame, (cx, cy), 3, GRAY, -1)
        cv2.circle(frame, (cx, cy), DEAD_ZONE_PX, BLUE, 1)

        if face_box is not None:
            fx, fy, fw, fh = face_box
            face_cx = fx + fw // 2
            face_cy = fy + fh // 2

            if tracked_label and identified_name and tracked_label == identified_name:
                box_colour = (90, 230, 255)   # cyan — confirmed identity
            elif tracked_label == "Unknown":
                box_colour = RED
            else:
                box_colour = GREEN

            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_colour, 2)
            cv2.circle(frame, (face_cx, face_cy), 4, box_colour, -1)

            if tracked_label:
                (tw, th), _ = cv2.getTextSize(
                    tracked_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1,
                )
                ly = max(fy - 28, th + 4)
                cv2.rectangle(frame, (fx, ly - th - 4), (fx + tw + 6, ly + 4), box_colour, -1)
                cv2.putText(frame, tracked_label, (fx + 3, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.arrowedLine(frame, (cx, cy), (face_cx, face_cy), YELLOW, 1, tipLength=0.15)

            err_x = face_cx - cx
            err_y = face_cy - cy
            cv2.putText(frame, f"dx={err_x:+d}px", (fx, max(fy - 22, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)
            cv2.putText(frame, f"dy={err_y:+d}px", (fx, max(fy - 8, 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No face detected", (cx - 60, cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)

        mode = "[SIM]" if self.simulate else "[LIVE]"
        hud  = f"{mode}  Pan:{pan:.1f}\u00b0  Tilt:{tilt:.1f}\u00b0  {fps:.0f} fps"
        cv2.putText(frame, hud, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, WHITE, 1, cv2.LINE_AA)

        if identified_name:
            tracking_them = (tracked_label == identified_name)
            if identified_present and not tracking_them:
                banner, colour = f"{identified_name} present (not tracked)", (90, 230, 255)
            elif tracking_them:
                banner, colour = f"Tracking: {identified_name}", (90, 230, 255)
            else:
                banner, colour = f"Looking for: {identified_name}", GRAY
            cv2.putText(frame, banner, (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

        self._draw_compass(frame, pan, tilt, w - 52, h - 52, 38)
        return frame

    # ── Recognition worker ────────────────────────────────────────────────────

    def _recognition_loop(self) -> None:
        """
        Every IDENTITY_CHECK_INTERVAL_S seconds, run face_recognition on the
        latest frame and update tracked_label + identified_present in state.
        Idles (sleeps) when nobody is logged in, so CPU cost is zero at rest.
        """
        while self._running:
            time.sleep(IDENTITY_CHECK_INTERVAL_S)
            if not self._running:
                break

            with self._identity_lock:
                name      = self._identified_name
                known_enc = list(self._identified_encodings)

            if not name or not known_enc:
                continue

            frame = self._cam.get_frame()
            if frame is None:
                continue

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                cv2.equalizeHist(gray, gray)
                faces = self._detector.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=4,
                    minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE,
                )
            except Exception as e:
                log.debug("Recognition detect failed: %s", e)
                continue

            if not len(faces):
                with self._state_lock:
                    self._state.tracked_label       = None
                    self._state.identified_present  = False
                    self._state.last_recognition_ts = time.time()
                continue

            # Largest first — mirrors the tracker's "largest = tracked" rule.
            faces_sorted = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
            faces_sorted = faces_sorted[:IDENTITY_MAX_FACES]

            # Pass Haar locations to encoder to skip dlib's slow HOG step.
            locations = [(int(y), int(x + w), int(y + h), int(x))
                         for (x, y, w, h) in faces_sorted]

            try:
                encs = face_recognition.face_encodings(
                    frame, known_face_locations=locations, num_jitters=0,
                )
            except Exception as e:
                log.debug("Recognition encode failed: %s", e)
                continue

            tracked_label      = "Unknown"
            identified_present = False

            for idx, enc in enumerate(encs):
                dists = face_recognition.face_distance(known_enc, enc)
                if len(dists) and float(np.min(dists)) <= IDENTITY_TOLERANCE:
                    if idx == 0:
                        tracked_label = name
                    identified_present = True

            with self._state_lock:
                self._state.tracked_label       = tracked_label
                self._state.identified_present  = identified_present
                self._state.last_recognition_ts = time.time()

    # ── Mini compass ──────────────────────────────────────────────────────────

    def _draw_compass(self, frame, pan, tilt, cx, cy, r) -> None:
        BLUE  = (80,  150, 255)
        DGRAY = (70,   70,  70)
        cv2.circle(frame, (cx, cy), r,     DGRAY, 1)
        cv2.circle(frame, (cx, cy), r + 1, DGRAY, 1)

        pan_n  = max(-1.0, min(1.0, (pan  - 90) / 60.0))
        tilt_n = max(-1.0, min(1.0, (tilt - 90) / 50.0))

        px = int(cx + pan_n  * r * 0.78)
        py = int(cy + tilt_n * r * 0.78)

        cv2.line(frame, (cx, cy), (px, cy), BLUE, 2)
        cv2.line(frame, (cx, cy), (cx, py), BLUE, 2)
        cv2.circle(frame, (cx, cy), 3, BLUE, -1)
