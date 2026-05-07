import sys
sys.path.append("/usr/lib/python3/dist-packages")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import face_recognition
import numpy as np
import base64
import json
import time
import uuid
import cv2
from pathlib import Path
from typing import List

import config
from tracker import FaceTracker, CameraManager

# ── Tracker (started/stopped with the server) ─────────────────────────────────
_tracker: FaceTracker | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tracker
    try:
        _tracker = FaceTracker()
        _tracker.start()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "FaceTracker could not start (%s). "
            "Tracking endpoints will return 503. "
            "Connect a camera and restart to enable tracking.", e
        )
        _tracker = None
    yield
    if _tracker:
        _tracker.stop()

app = FastAPI(title="Face Login Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FACES_DB = Path("faces/db.json")
FACES_DB.parent.mkdir(exist_ok=True)

_last_login_frame: bytes = b""

TOLERANCE = config.IDENTITY_TOLERANCE
MIN_FRAMES = 3

_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces_haar(rgb_img: np.ndarray) -> list:
    """
    Detect face locations using Haar cascade (fast, works where dlib HOG fails).
    Returns locations in face_recognition format: [(top, right, bottom, left), ...]
    """
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(gray, gray)
    faces = _haar.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40),
    )
    if not len(faces):
        return []
    return [(y, x + w, y + h, x) for (x, y, w, h) in faces]


def load_db() -> dict:
    if FACES_DB.exists():
        return json.loads(FACES_DB.read_text())
    return {}


def save_db(db: dict):
    FACES_DB.write_text(json.dumps(db))


def decode_image(b64_image: str) -> np.ndarray:
    if "," in b64_image:
        b64_image = b64_image.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_image)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    h, w = img.shape[:2]
    if w > 640:
        scale = 640 / w
        img = cv2.resize(img, (640, int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class RegisterRequest(BaseModel):
    name: str
    images: List[str]


class LoginRequest(BaseModel):
    image: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("templates/index.html").read_text()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/faces")
async def list_faces():
    db = load_db()
    return {"faces": list(db.keys())}


@app.post("/register")
async def register(req: RegisterRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    if not req.images:
        raise HTTPException(status_code=400, detail="No images provided")

    encodings = []
    for img_b64 in req.images:
        try:
            img  = decode_image(img_b64)
            locs = detect_faces_haar(img)
            found = face_recognition.face_encodings(img, known_face_locations=locs,
                                                    num_jitters=1)
            if found:
                encodings.append(found[0].tolist())
        except Exception:
            pass

    if len(encodings) < MIN_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Only {len(encodings)} usable frame(s) — need at least {MIN_FRAMES}. "
                   "Ensure good lighting and keep your face visible.",
        )

    db = load_db()
    db[name] = encodings
    save_db(db)
    return {
        "success": True,
        "message": f"Registered '{name}' with {len(encodings)} face angles",
        "frames_used": len(encodings),
    }


@app.post("/login")
async def login(req: LoginRequest):
    global _last_login_frame
    img = decode_image(req.image)

    _, _buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 90])
    _last_login_frame = _buf.tobytes()

    locations = detect_faces_haar(img)
    encodings = face_recognition.face_encodings(img, known_face_locations=locations,
                                                num_jitters=1)
    if not encodings:
        raise HTTPException(
            status_code=400,
            detail=f"No face detected — brightness={round(float(np.mean(img)),1)}/255. "
                   f"Open /debug/last-frame to see what the server received."
        )

    db = load_db()
    if not db:
        raise HTTPException(status_code=404, detail="No registered faces in database")

    probe = encodings[0]
    best_name = None
    best_dist = 1.0

    for name, stored in db.items():
        known = [np.array(e) for e in stored]
        distances = face_recognition.face_distance(known, probe)
        min_dist = float(np.min(distances))
        if min_dist < best_dist:
            best_dist = min_dist
            best_name = name

    if best_dist <= TOLERANCE:
        confidence = round((1 - best_dist) * 100, 1)

        # Bridge login → tracking: tell the tracker who just logged in so it
        # can run periodic identity checks against that person's encodings.
        if _tracker:
            try:
                _tracker.set_identified_person(best_name)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Failed to set identified person on tracker: %s", e
                )

        return {
            "success": True,
            "name": best_name,
            "token": str(uuid.uuid4()),
            "confidence": confidence,
        }

    return {"success": False, "name": None, "token": None, "message": "Face not recognized"}


@app.delete("/faces/{name}")
async def delete_face(name: str):
    db = load_db()
    if name not in db:
        raise HTTPException(status_code=404, detail=f"Face '{name}' not found")
    del db[name]
    save_db(db)
    return {"success": True, "message": f"Deleted '{name}'"}


# ── Tracking endpoints ────────────────────────────────────────────────────────

@app.get("/track/status")
async def track_status():
    """Current pan/tilt angles, face position, and loop FPS."""
    if not _tracker:
        raise HTTPException(status_code=503, detail="Tracker not running")
    s = _tracker.state
    return {
        "pan":        round(s.pan,  1),
        "tilt":       round(s.tilt, 1),
        "error_x":    s.error_x,
        "error_y":    s.error_y,
        "face_found": s.face_found,
        "fps":        s.fps,
        "simulate":   _tracker.simulate,
    }


@app.get("/track/snapshot")
async def track_snapshot():
    """Latest annotated camera frame as a JPEG image."""
    if not _tracker:
        raise HTTPException(status_code=503, detail="Tracker not running")
    jpeg = _tracker.state.annotated_jpeg
    if not jpeg:
        raise HTTPException(status_code=503, detail="No frame available yet")
    return Response(content=jpeg, media_type="image/jpeg")


@app.post("/track/feed")
async def track_feed(req: LoginRequest):
    """
    Receive a camera frame from the browser (base64 JPEG).
    Used in browser-feed mode when the hardware camera is held by the browser.
    """
    cam = CameraManager()
    img = decode_image(req.image)
    cam.inject_frame(img)
    return {"ok": True, "source": cam.source}


@app.get("/track/source")
async def track_source():
    return {"source": CameraManager().source}


@app.get("/config")
async def get_config():
    return {
        "mode":   config.MODE,
        "is_pi":  config.IS_PI,
        "source": CameraManager().source,
    }


@app.get("/camera/stream")
async def camera_stream():
    """Raw MJPEG stream from the camera, no overlay. Used by the login page preview."""
    cam = CameraManager()

    async def generate():
        while True:
            frame = cam.get_frame()
            if frame is not None:
                _, buf = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
            await asyncio.sleep(0.05)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/camera/frame")
async def camera_frame():
    """Latest raw (un-annotated) frame as a JPEG. Used by login capture on Pi."""
    cam = CameraManager()
    frame = cam.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available yet")
    _, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/debug/last-frame")
async def debug_last_frame():
    """Returns the last image received by POST /login. Remove before production."""
    if not _last_login_frame:
        raise HTTPException(status_code=404, detail="No login attempt yet")
    return Response(content=_last_login_frame, media_type="image/jpeg")


@app.get("/track/stream")
async def track_stream():
    """MJPEG stream of the annotated camera feed at ~20 fps."""
    if not _tracker:
        raise HTTPException(status_code=503, detail="Tracker not running")

    async def generate():
        while True:
            jpeg = _tracker.state.annotated_jpeg
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.05)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/track/person")
async def track_person():
    """
    Channel 1 for the cloud multi-agent system — poll every N seconds.

    person                   — name | "Unknown" | null (null = no face / not yet checked)
    position                 — bbox centre in 640×480 pixels
    identified_person_present— true if the logged-in person appears anywhere in frame,
                               even when they are not the largest tracked subject
    """
    if not _tracker:
        raise HTTPException(status_code=503, detail="Tracker not running")
    s = _tracker.state

    position = {"x": None, "y": None}
    if s.tracked_position is not None:
        position = {"x": int(s.tracked_position[0]), "y": int(s.tracked_position[1])}

    return {
        "person":                    s.tracked_label,
        "position":                  position,
        "timestamp":                 int(time.time()),
        "frame_size":                {"width": 640, "height": 480},
        "identified_person_present": s.identified_present,
        "identified_name":           s.identified_name,
        "last_recognition_ts":       s.last_recognition_ts,
    }


@app.get("/track/identity")
async def get_identity():
    """Returns who the tracker is currently looking for, if anyone."""
    if not _tracker:
        raise HTTPException(status_code=503, detail="Tracker not running")
    s = _tracker.state
    return {
        "identified_name":           s.identified_name,
        "identified_person_present": s.identified_present,
        "tracked_label":             s.tracked_label,
    }


@app.delete("/track/identity")
async def clear_identity():
    """Forget the logged-in person — recognition worker idles afterwards."""
    if not _tracker:
        raise HTTPException(status_code=503, detail="Tracker not running")
    _tracker.clear_identified_person()
    return {"success": True}


@app.get("/track", response_class=HTMLResponse)
async def track_monitor():
    """Live tracking monitor page."""
    return Path("templates/track.html").read_text()


# ── Frame endpoints for cloud / multi-agent consumers ─────────────────────────
#
# Two-channel contract for the off-Pi multi-agent system:
#
#   Channel 1 — events (poll every n sec):  GET /track/person
#   Channel 2 — frames (poll every t sec):  GET /frames/current   raw JPEG + metadata headers
#                                           GET /frames/event     atomic JSON with base64 JPEG
#                                           GET /frames/stream    continuous MJPEG
#
# /frames/* is a separate namespace from /camera/* — unambiguously for downstream consumers.

@app.get("/frames/current")
async def frames_current():
    """
    Latest raw camera frame as JPEG. Detection metadata is embedded in response
    headers so agents get frame + identity in one round-trip without parsing JSON.

    Response headers:
      X-Timestamp                  — unix epoch seconds
      X-Tracked-Person             — name | "Unknown" | "" (empty = no face tracked)
      X-Identified-Name            — who login set (empty if nobody logged in)
      X-Identified-Person-Present  — "true" / "false"
      X-Tracked-Position           — "x,y" in 640×480 pixels, or empty
      X-Frame-Width / X-Frame-Height
    """
    cam = CameraManager()
    frame = cam.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available yet")

    _, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 80],
    )
    h, w = frame.shape[:2]
    headers = {
        "X-Timestamp":    str(int(time.time())),
        "X-Frame-Width":  str(w),
        "X-Frame-Height": str(h),
        "Cache-Control":  "no-store",
    }
    if _tracker:
        s = _tracker.state
        headers["X-Tracked-Person"]            = s.tracked_label or ""
        headers["X-Identified-Name"]           = s.identified_name or ""
        headers["X-Identified-Person-Present"] = "true" if s.identified_present else "false"
        if s.tracked_position is not None:
            headers["X-Tracked-Position"] = f"{int(s.tracked_position[0])},{int(s.tracked_position[1])}"
        else:
            headers["X-Tracked-Position"] = ""

    return Response(content=buf.tobytes(), media_type="image/jpeg", headers=headers)


@app.get("/frames/event")
async def frames_event():
    """
    Atomic JSON snapshot: base64 JPEG + detection payload correlated by timestamp.
    Use when an LLM agent needs frame + identity in one parseable object.
    Use /frames/current instead when bandwidth or latency matters more.
    """
    cam = CameraManager()
    frame = cam.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available yet")

    _, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 80],
    )
    h, w = frame.shape[:2]

    payload = {
        "timestamp": int(time.time()),
        "frame": {
            "format": "jpeg",
            "width":  w,
            "height": h,
            "base64": base64.b64encode(buf.tobytes()).decode("ascii"),
        },
        "person":                    None,
        "position":                  {"x": None, "y": None},
        "box":                       None,
        "identified_name":           None,
        "identified_person_present": False,
        "last_recognition_ts":       0.0,
    }
    if _tracker:
        s = _tracker.state
        payload["person"]                    = s.tracked_label
        payload["identified_name"]           = s.identified_name
        payload["identified_person_present"] = s.identified_present
        payload["last_recognition_ts"]       = s.last_recognition_ts
        if s.tracked_position is not None:
            payload["position"] = {
                "x": int(s.tracked_position[0]),
                "y": int(s.tracked_position[1]),
            }
        if s.tracked_box is not None:
            fx, fy, fw, fh = s.tracked_box
            payload["box"] = {
                "top":    fy,
                "right":  fx + fw,
                "bottom": fy + fh,
                "left":   fx,
            }
    return payload


@app.get("/frames/stream")
async def frames_stream():
    """
    Continuous raw MJPEG stream for agents that prefer video over polling.
    No tracker overlay — use /track/stream for the annotated version.
    """
    cam = CameraManager()

    async def generate():
        while True:
            frame = cam.get_frame()
            if frame is not None:
                _, buf = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
            await asyncio.sleep(0.05)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
