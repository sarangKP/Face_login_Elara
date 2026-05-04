from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
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

from tracker import FaceTracker

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

TOLERANCE = 0.50          # stricter with multi-angle encodings
MIN_FRAMES = 3            # minimum good frames required for registration


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
            img = decode_image(img_b64)
            found = face_recognition.face_encodings(img, num_jitters=1)
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
    img = decode_image(req.image)
    # num_jitters=1 keeps it fast; good enough with multi-angle DB
    encodings = face_recognition.face_encodings(img, num_jitters=1)

    if not encodings:
        raise HTTPException(status_code=400, detail="No face detected")

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
    """Latest annotated camera frame as a JPEG image (for browser debug view)."""
    if not _tracker:
        raise HTTPException(status_code=503, detail="Tracker not running")
    jpeg = _tracker.state.annotated_jpeg
    if not jpeg:
        raise HTTPException(status_code=503, detail="No frame available yet")
    return Response(content=jpeg, media_type="image/jpeg")
