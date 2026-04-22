# Face Login — Elara Service

Stateless face recognition microservice for the [Elara Service](https://github.com/sarangKP/Elara_service). Built with FastAPI + `face_recognition` (dlib/HOG).

---

## How it works

1. **Register** — guided 5-step flow captures your face at different angles (center → left → right → up → tilt). Multiple encodings are stored per user, giving robust matching.
2. **Login** — a single frame is compared against all stored encodings. The best-distance match across all angles is returned with a confidence score and a session token.

### Under the hood

No model is trained on your machine. The pipeline is:

1. **Detection** — a HOG (Histogram of Oriented Gradients) + SVM detector locates the face in the frame. Classical computer vision, no neural network, fast on Pi.
2. **Landmark alignment** — 68 points (eyes, nose, jaw, mouth) are predicted on the detected face. The eye centres and nose tip are used to compute an affine transform (rotation + scale + translation) that warps the crop into a fixed 150×150 template. This ensures the face is always upright and centred before the next step.
3. **Embedding** — the aligned crop is passed through a pre-trained **ResNet-34** (ships inside `face_recognition_models`). It outputs a **128-number vector** that encodes the geometry of the face. Similar faces produce similar vectors. You never train this network — the weights are fixed.
4. **Matching** — at login, the Euclidean distance between the probe vector and every stored vector is computed. Distance < 0.5 → same person. No neural network — just maths.

```
Register: frame → detect → align → ResNet-34 → 128 floats  →  saved to db.json
Login:    frame → detect → align → ResNet-34 → 128 floats  →  distance compare → match
```

`db.json` is not a model — it is a lookup table of 128-number vectors, one per registered face angle.

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager
- **Python 3.12** (see note below on 3.13+)
- `cmake` for building dlib:
  ```bash
  sudo apt install cmake build-essential
  ```

### Python version note

`face_recognition` depends on `face_recognition_models`, which uses `pkg_resources` from `setuptools`. In **Python 3.13+** `pkg_resources` is no longer bundled, causing an import error at runtime.

**Recommended: use Python 3.12** — already available on this machine and the default on Raspberry Pi OS Bookworm.

To switch to 3.12:

```bash
# 1. Pin the version
echo "3.12" > .python-version

# 2. In pyproject.toml, change:
#    requires-python = ">=3.13"  →  requires-python = ">=3.12"

# 3. Re-create the venv
uv sync
```

No patch needed on 3.12 — `pkg_resources` ships with setuptools there.

**If you must stay on Python 3.13**, pin an older setuptools instead:

```bash
uv add "setuptools<71"
```

setuptools < 71 still bundles `pkg_resources`. This is the minimal-friction workaround without changing Python.

---

## Installation

```bash
git clone https://github.com/sarangKP/Face_Login.git
cd Face_Login
uv sync
```

---

## Running

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8765
```

Open **http://localhost:8765** in your browser.

Auto-reload for development:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8765 --reload
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Login UI (HTML) |
| `GET` | `/health` | `{"status": "ok"}` |
| `GET` | `/faces` | List registered names |
| `POST` | `/register` | Register face (multi-angle) |
| `POST` | `/login` | Authenticate with face |
| `DELETE` | `/faces/{name}` | Remove a registered user |

### POST `/register`

```json
{
  "name": "Alice",
  "images": ["data:image/jpeg;base64,...", "..."]
}
```

`images` is a list of base64-encoded JPEG frames (one per pose). At least 3 usable face detections are required.

```json
{ "success": true, "message": "Registered 'Alice' with 5 face angles", "frames_used": 5 }
```

### POST `/login`

```json
{ "image": "data:image/jpeg;base64,..." }
```

```json
{ "success": true, "name": "Alice", "token": "uuid4", "confidence": 87.3 }
```

`token` is a random UUID issued per successful login. Pass it downstream to identify the session.

---

## Connecting to Elara Service

After a successful face login, use the returned `name` and `token` to start an Elara chat session.

```python
import httpx

# 1. Face login
login = httpx.post("http://localhost:8765/login", json={"image": frame_b64}).json()
assert login["success"], "Face not recognized"

# 2. Start Elara session
elara_state = {}
reply = httpx.post("http://elara-host:8000/chat", json={
    "message": "Hello",
    "state":   elara_state,
    "backend": "ollama",
    "model":   "llama3",
}).json()

elara_state = reply["state"]   # must be echoed back on every subsequent turn
print(reply["reply"])
```

> **Elara is fully stateless** — `state` must be echoed back on every request.
> The face login `name` can be used as the user identifier across sessions,
> and `token` can be checked client-side to gate access before making any Elara call.

For streaming responses use `/chat/stream` (SSE):

```python
with httpx.stream("POST", "http://elara-host:8000/chat/stream", json={...}) as r:
    for line in r.iter_lines():
        if line.startswith("data:"):
            print(line[5:], end="", flush=True)
```

---

## Raspberry Pi + Pi Camera

### Camera module (picamera2)

The browser UI uses `getUserMedia` — this works in Chromium on Pi. The OS exposes the Pi camera as a V4L2 device, so no code changes are needed for browser-based use.

For a **headless kiosk** (no browser, server captures frames directly):

```bash
sudo apt install python3-picamera2
uv add picamera2 --no-build-isolation
```

Add a `/camera/frame` endpoint to `main.py`:

```python
from picamera2 import Picamera2
import io, base64
from PIL import Image

_cam = None

def get_camera():
    global _cam
    if _cam is None:
        _cam = Picamera2()
        _cam.configure(_cam.create_preview_configuration(main={"size": (640, 480)}))
        _cam.start()
    return _cam

@app.get("/camera/frame")
async def camera_frame():
    cam = get_camera()
    arr = cam.capture_array()
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"image": f"data:image/jpeg;base64,{b64}"}
```

The front-end can then poll `/camera/frame` instead of `getUserMedia`.

### Performance tips for Pi

| Setting | Value | Reason |
|---------|-------|--------|
| Input resolution | 640 × 480 | Enforced in `decode_image()` |
| `num_jitters` on login | 1 | Fastest; multi-angle DB compensates |
| Face detector | HOG (default) | CNN is more accurate but too slow on Pi CPU |
| JPEG quality | 0.82 | Reduce to 0.7 if Pi-to-server bandwidth is tight |

---

## Project structure

```
Face_Login/
├── main.py              # FastAPI app — all endpoints
├── templates/
│   └── index.html       # Guided registration + login UI
├── faces/
│   └── db.json          # Face encoding database (auto-created)
├── pyproject.toml
└── .python-version      # Pin to 3.12
```
