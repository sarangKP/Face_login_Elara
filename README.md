# Face Login — Elara Service

Stateless face recognition + pan-tilt face tracking microservice for the [Elara Service](https://github.com/sarangKP/Elara_service). Built with FastAPI, `face_recognition` (dlib ResNet-34), and OpenCV Haar cascade.

Runs in two modes from the same codebase:

- **Pi mode** — Raspberry Pi 5 with the Pi Camera Module and an ESP32-driven pan/tilt mount. Production target.
- **Laptop mode** — your dev machine. Browser webcam for face login, simulated servos for tracking. No hardware needed.

The mode is picked automatically (presence of `picamera2`) or forced via the `DEVICE_MODE` environment variable.

---

## User flow

```
http://localhost:8765/          ← Register or login
        ↓  face recognised
http://localhost:8765/track     ← Live face tracking + pan-tilt mount
        ↓  logout button
http://localhost:8765/          ← Back to login
```

---

## How it works

### Mode selection

`config.py` exposes a single `MODE` constant resolved on import:

| Source | Result |
|--------|--------|
| `DEVICE_MODE=pi` env var | force Pi mode |
| `DEVICE_MODE=laptop` env var | force laptop mode |
| unset, `picamera2` importable | auto → `pi` |
| unset, `picamera2` not importable | auto → `laptop` |

Everything downstream — which camera path opens, whether the servo controller talks to the ESP32 or simulates, which preview the login page shows — keys off this single flag.

### Face login

1. **Register** — guided 5-step flow captures the user's face at different angles (center → left → right → up → tilt). Multiple encodings are stored per user for robust matching.
2. **Login** — a single frame is compared against all stored encodings. The closest match wins, returned with a confidence score and a session token. The browser then redirects to `/track`.

The capture source is mode-aware:

- **Laptop** → browser opens its webcam via `getUserMedia` and POSTs JPEGs to `/login`.
- **Pi** → page shows `/camera/stream` (MJPEG straight from the Pi Camera) for preview, and pulls a single still from `/camera/frame` to submit to `/login`.

### Face detection — Haar + dlib hybrid

dlib's HOG detector (used internally by `face_recognition`) is strict and misses faces on many laptop webcams. The fix: use **Haar cascade** (fast, tolerant) to locate the face, then pass those coordinates directly to `face_recognition.face_encodings()` so dlib only computes the embedding — it never runs its own detection.

```
Haar cascade  →  face location (top, right, bottom, left)
                          ↓
face_recognition.face_encodings(img, known_face_locations=locs)
                          ↓
                  128-float embedding  →  stored / compared
```

### Under the hood — no training

No model is trained on your machine. The full pipeline per frame:

| Step | Algorithm | Trained by you? |
|------|-----------|-----------------|
| Face detection | Haar cascade (OpenCV) | No |
| Face location → crop | Affine transform on 68 landmarks | No |
| Embedding | Pre-trained ResNet-34 (ships in `face_recognition_models`) | No |
| Matching | Euclidean distance < 0.5 threshold | No |

`db.json` is not a model — it is a lookup table of 128-float vectors, one per registered face angle.

### Face tracking

After login, the tracking page runs a **20 fps PID loop**:

```
Camera frame → Haar detect face → compute X/Y pixel error from centre
                                           ↓
                                    PID controller
                                           ↓
                               pan angle += Δpan
                               tilt angle += Δtilt
                                           ↓
                       USB serial → ESP32 → PWM → servos   (Pi)
                       virtual angle state only            (laptop / no ESP32)
```

If the ESP32 is unreachable on Pi mode (`/dev/ttyUSB*`/`/dev/ttyACM*` not found), `ServoController` logs a warning and falls back to simulate mode automatically. The camera and PID loop keep working — only the physical move is suppressed.

### Camera architecture — one camera, no conflicts

A single `CameraManager` singleton owns the hardware camera for the lifetime of the server process. Two HTTP streams expose its frames:

| Endpoint | Source | Used by |
|----------|--------|---------|
| `/camera/stream` | raw MJPEG, no overlay | Login page preview (Pi mode) |
| `/track/stream` | annotated by `FaceTracker` | Tracking page |
| `/camera/frame` | single raw JPEG | Login capture (Pi mode) |

Login and tracking are otherwise decoupled — the login flow does not depend on the tracker thread being alive.

```
Pi mode:
    picamera2 ──▶ CameraManager ──┬──▶ /camera/stream  (login preview)
                                  ├──▶ /camera/frame   (login capture)
                                  └──▶ FaceTracker ──▶ /track/stream

Laptop mode:
    Browser getUserMedia ──▶ /login           (face login, in-browser capture)
                         ──▶ POST /track/feed (browser-feed for tracker)
                                  └──▶ FaceTracker ──▶ /track/stream
```

### Pi camera quirks (handled)

Two issues bite anyone using `picamera2` for the first time, both fixed inside `CameraManager._capture_loop`:

1. **`RGB888` is actually BGR.** libcamera names pixel formats by *byte order*; OpenCV names them by *channel order*. The bytes returned for `format="RGB888"` are B, G, R — so without a `cv2.cvtColor(BGR2RGB)` reds and blues are swapped throughout the pipeline.
2. **Pi camera isn't mirrored, browser webcams typically are.** The PID error sign was tuned against the laptop browser-feed path, which mirrors frames in JS before sending. To keep the same logic working on Pi, picamera2 frames are flipped horizontally (`config.MIRROR_PI_FRAME`). Without this, the mount chases the face the wrong way.

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager
- **Python 3.12** (see note below on 3.13+)
- `cmake` for building dlib:
  ```bash
  sudo apt install cmake build-essential
  ```

### Python version note

`face_recognition` depends on `face_recognition_models`, which uses `pkg_resources` from `setuptools`. In **Python 3.13+** `pkg_resources` is no longer bundled.

**Recommended: use Python 3.12** — already the default on Raspberry Pi OS Bookworm.

```bash
echo "3.12" > .python-version
# In pyproject.toml: requires-python = ">=3.12"
uv sync
```

**If you must stay on 3.13**, pin an older setuptools:

```bash
uv add "setuptools<71"
```

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
# auto-detects mode (pi if picamera2 is installed, otherwise laptop)
uv run uvicorn main:app --host 0.0.0.0 --port 8765
```

Force a specific mode:

```bash
DEVICE_MODE=laptop uv run uvicorn main:app --host 0.0.0.0 --port 8765
DEVICE_MODE=pi     uv run uvicorn main:app --host 0.0.0.0 --port 8765
```

Open **http://localhost:8765** in your browser.

> **Tip:** if a USB webcam appears black, another process is holding `/dev/video0`.
> Run `fuser -k /dev/video0` to release it, then restart the server.

---

## API

### Face login

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Login + register UI |
| `GET` | `/health` | `{"status": "ok"}` |
| `GET` | `/config` | `{ mode, is_pi, source }` — frontend uses this to pick its capture path |
| `GET` | `/faces` | List registered names |
| `POST` | `/register` | Register face (multi-angle batch) |
| `POST` | `/login` | Authenticate — returns token, redirects to `/track` |
| `DELETE` | `/faces/{name}` | Remove a registered user |
| `GET` | `/debug/last-frame` | JPEG of what the last `/login` call received |

### Camera (raw, tracker-independent)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/camera/stream` | MJPEG stream straight from the camera, no overlay |
| `GET` | `/camera/frame`  | Single raw JPEG frame |

### Face tracking

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/track` | Live tracking monitor UI |
| `GET` | `/track/status` | Pan/tilt angles, error offsets, FPS, simulate flag |
| `GET` | `/track/stream` | MJPEG stream of annotated camera feed |
| `GET` | `/track/snapshot` | Single annotated JPEG frame |
| `POST` | `/track/feed` | Receive frame from browser (laptop browser-feed mode) |
| `GET` | `/track/source` | Which camera source is active (`picamera2` / `browser`) |

### POST `/register`

```json
{ "name": "Alice", "images": ["data:image/jpeg;base64,...", "..."] }
```

At least 3 usable face detections required across the 5 captured poses.

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

Browser automatically redirects to `/track?name=Alice&token=<uuid>` on success.

---

## Connecting to Elara Service

After a successful face login, use the returned `name` and `token` to start an Elara chat session.

```python
import httpx

# 1. Face login
login = httpx.post("http://localhost:8765/login", json={"image": frame_b64}).json()
assert login["success"], "Face not recognized"

# 2. Start Elara session — state starts empty, must be echoed back every turn
state = {}
reply = httpx.post("http://elara-host:8000/chat", json={
    "message": "Hello",
    "state":   state,
    "backend": "ollama",
    "model":   "llama3",
}).json()
state = reply["state"]
print(reply["reply"])
```

For streaming (SSE):

```python
with httpx.stream("POST", "http://elara-host:8000/chat/stream", json={...}) as r:
    for line in r.iter_lines():
        if line.startswith("data:"):
            print(line[5:], end="", flush=True)
```

---

## Raspberry Pi 5

### Servo control — ESP32 over USB serial

Pi 5 uses the **RP1 southbridge** for GPIO. Rather than fight RP1's PWM quirks (and the brown-out risk of driving servos from Pi rails), this project offloads servo control to a small ESP32 firmware.

```
Python (Pi)  ──USB serial──▶  ESP32  ──PWM──▶  pan + tilt servos
```

Wire-up:

```
ESP32 GPIO 13  →  pan  servo signal
ESP32 GPIO 12  →  tilt servo signal
Servo V+       →  separate 5 V supply  (do NOT use Pi 5 V or ESP32 3.3 V)
Servo GND      →  common GND with the ESP32
```

Protocol on `/dev/ttyUSB*` at 115200 baud, newline-terminated:

```
P<pan_deg> T<tilt_deg>\n        e.g.   P105.3 T72.0
```

Steps:

1. Flash the firmware in `esp32_servo/` (PlatformIO: open the folder → Upload).
2. Wire as above. Use a dedicated 5 V PSU for the servos.
3. `uv add pyserial` (already in `pyproject.toml`).
4. Plug in the ESP32 — `servo.py` auto-detects the first `ttyUSB*`/`ttyACM*` port.

If the ESP32 is missing or unreachable, the server starts anyway and prints:

```
ServoController: serial unavailable (...) — falling back to simulate mode.
```

You can then test the camera + PID + UI without any motors connected.

> Two SG90s at stall current can brown-out a Pi 5 and corrupt the SD card.
> Always power servos from a dedicated 5 V supply.

### PID tuning

Gains live at the top of `tracker.py`:

```python
PAN_PID_GAINS  = dict(kp=0.03, ki=0.0, kd=0.002, ...)
TILT_PID_GAINS = dict(kp=0.03, ki=0.0, kd=0.002, ...)
```

- Tracking **sluggish** → increase `kp` (e.g. 0.03 → 0.07)
- Mount **oscillates** → increase `kd` (e.g. 0.002 → 0.006)
- Keep `ki=0` until Kp/Kd are stable — integral windup causes hunting

### Performance

| Setting | Value | Reason |
|---------|-------|--------|
| Camera resolution | 640 × 480 | Enforced by `picamera2` config / `decode_image()` |
| Tracker detect resolution | 320 × 240 | Haar runs ~15–30 ms/frame |
| Face detector (tracking) | Haar cascade | 10–30 ms vs 150–300 ms for HOG/dlib |
| Face detector (login) | Haar + dlib encode | Better detection, same embedding quality |
| `num_jitters` on login | 1 | Fastest; multi-angle DB compensates |

---

## Diagnostics

If the camera or face detection isn't working:

```bash
pkill -f uvicorn          # stop server so it releases the camera
uv run python diagnose.py # saves test frames to debug_frames/
```

The script tests every video device × backend × codec combination, runs both Haar and `face_recognition` detection, and saves annotated PNGs so you can see exactly what the camera captures.

Quick sanity check that `picamera2` sees the Pi Camera:

```bash
uv run python -c "
import sys; sys.path.insert(0, '/usr/lib/python3/dist-packages')
from picamera2 import Picamera2
print(Picamera2.global_camera_info())
"
```

---

## Project structure

```
Face_Login/
├── config.py            # Mode selection (pi vs laptop) + frame-mirror flag
├── main.py              # FastAPI app — all endpoints + tracker lifespan
├── tracker.py           # CameraManager + FaceTracker (Haar detection + PID loop)
├── pid.py               # Discrete PID controller with anti-windup
├── servo.py             # Servo abstraction — ESP32 serial or simulate mode
├── diagnose.py          # Camera + face detection diagnostic script
├── esp32_servo/         # ESP32 firmware (PlatformIO project) for pan/tilt PWM
├── templates/
│   ├── index.html       # Guided registration + login UI → redirects to /track
│   └── track.html       # Live tracking monitor (MJPEG feed + servo compass)
├── faces/
│   └── db.json          # Face encoding database (auto-created)
├── debug_frames/        # Frames saved by diagnose.py (auto-created)
├── pyproject.toml
└── .python-version      # Pin to 3.12
```
