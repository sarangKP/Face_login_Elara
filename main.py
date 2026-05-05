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
import uuid
import cv2
from pathlib import Path
from typing import List

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
        # Non-fatal — face login still works without a local camera
        import logging
        logging.getLogger(__name__).warning(
            "FaceTracker could not start (%s). "
            "Tracking endpoints will return 503. "
            "Connect a camera and restart to enable tracking.", e
        )
        _tracker = None
    yield                           # server runs here
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

# Stores the last image received by /login for debugging
_last_login_frame: bytes = b""

TOLERANCE = 0.50          # stricter with multi-angle encodings
MIN_FRAMES = 3            # minimum good frames required for registration

# Haar cascade — shared across requests, loaded once at startup
_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces_haar(rgb_img: np.ndarray) -> list:
    """
    Detect face locations using Haar cascade (fast, works where dlib HOG fails).
    Returns locations in face_recognition format: [(top, right, bottom, left), ...]

    Why: dlib's HOG detector (used by face_recognition internally) misses faces
    on many laptop webcams. Haar is more tolerant of lighting and face size.
    We detect with Haar, then pass those boxes straight to face_encodings() so
    dlib only computes the embedding — it skips its broken detection step.
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

    # Convert (x, y, w, h) → (top, right, bottom, left) for face_recognition
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
    # Downscale for speed on Pi — face_recognition works fine at 480p
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
    images: List[str]  # multiple base64 frames (one per pose)


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
    for i, img_b64 in enumerate(req.images):
        try:
            img  = decode_image(img_b64)
            locs = detect_faces_haar(img)      # Haar detects, dlib encodes
            found = face_recognition.face_encodings(img, known_face_locations=locs,
                                                    num_jitters=1)
            if found:
                encodings.append(found[0].tolist())
        except Exception:
            pass  # skip bad frames

    if len(encodings) < MIN_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Only {len(encodings)} usable frame(s) — need at least {MIN_FRAMES}. "
                   "Ensure good lighting and keep your face visible.",
        )

    db = load_db()
    db[name] = encodings  # store all pose encodings
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

    # Save for /debug/last-frame
    _, _buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 90])
    _last_login_frame = _buf.tobytes()

    locations = detect_faces_haar(img)         # Haar detects, dlib encodes
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
        # stored is a list of encodings (list of lists)
        known = [np.array(e) for e in stored]
        distances = face_recognition.face_distance(known, probe)
        min_dist = float(np.min(distances))
        if min_dist < best_dist:
            best_dist = min_dist
            best_name = name

    if best_dist <= TOLERANCE:
        confidence = round((1 - best_dist) * 100, 1)
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
    The tracker processes this frame exactly like a hardware frame.
    """
    cam = CameraManager()
    img = decode_image(req.image)   # → RGB numpy array, already resized to ≤640px
    cam.inject_frame(img)
    return {"ok": True, "source": cam.source}


@app.get("/track/source")
async def track_source():
    """Returns which camera source the tracker is using."""
    return {"source": CameraManager().source}


@app.get("/debug/last-frame")
async def debug_last_frame():
    """
    Returns the last image received by POST /login as a JPEG.
    Open in browser to verify the server actually sees your face.
    Remove this endpoint before production deployment.
    """
    if not _last_login_frame:
        raise HTTPException(status_code=404, detail="No login attempt yet")
    return Response(content=_last_login_frame, media_type="image/jpeg")


@app.get("/track/stream")
async def track_stream():
    """
    MJPEG stream of the annotated camera feed.
    Point an <img> tag at this URL — browser displays it as live video.
    Runs at the same rate as the tracker loop (~20 fps).
    """
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
            await asyncio.sleep(0.05)   # 20 fps ceiling

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/track", response_class=HTMLResponse)
async def track_monitor():
    """Live tracking monitor page."""
    return Path("templates/track.html").read_text()
