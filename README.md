# Face Login — Elara Service

Stateless face recognition + pan-tilt face tracking microservice for the [Elara Service](https://github.com/sarangKP/Elara_service). Built with FastAPI, `face_recognition` (dlib ResNet-34), and OpenCV Haar cascade. Runs on a laptop for development and on Raspberry Pi 5 with Pi Camera + servo mount for deployment.

---

## User flow

```
http://localhost:8765/          ← Register or login
        ↓  face recognised
http://localhost:8765/track     ← Live face tracking + pan-tilt simulation
        ↓  logout button
http://localhost:8765/          ← Back to login
```

---

## How it works

### Face login

1. **Register** — guided 5-step flow captures your face at different angles (center → left → right → up → tilt). Multiple encodings are stored per user for robust matching.
2. **Login** — a single frame is compared against all stored encodings. The closest match across all angles is returned with a confidence score and a session token. On success the browser automatically redirects to the tracking page.

### Face detection — Haar + dlib hybrid

dlib's HOG detector (used internally by `face_recognition`) is strict and misses faces on many laptop webcams. The solution: use **Haar cascade** (fast, tolerant) to locate the face, then pass those coordinates directly to `face_recognition.face_encodings()` so dlib only computes the embedding — it never runs its own detection.

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
                              lgpio PWM → servo move  (Pi)
                              simulated display        (laptop)
```

### Camera architecture — one camera, no conflicts

On a laptop there is only one camera. If the server opened it with OpenCV, the browser's `getUserMedia` would be locked out. The solution:

- **Laptop** — server never touches the webcam. The `/track` page captures via `getUserMedia` and POSTs frames to `POST /track/feed`. Server annotates them and streams back via `/track/stream`.
- **Pi** — `picamera2` opens the Pi Camera module on startup (separate hardware from the browser), so both run simultaneously.

```
Laptop:   Browser → getUserMedia → POST /track/feed → annotate → /track/stream
Pi:       picamera2 (server) → annotate → /track/stream
                               Browser → getUserMedia → face login
```

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
uv run uvicorn main:app --host 0.0.0.0 --port 8765
```

Open **http://localhost:8765** in your browser.

> **Tip:** if the camera appears black, another process is holding `/dev/video0`.
> Run `fuser -k /dev/video0` to release it, then restart the server.

---

## API

### Face login

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Login + register UI |
| `GET` | `/health` | `{"status": "ok"}` |
| `GET` | `/faces` | List registered names |
| `POST` | `/register` | Register face (multi-angle batch) |
| `POST` | `/login` | Authenticate — returns token, redirects to `/track` |
| `DELETE` | `/faces/{name}` | Remove a registered user |
| `GET` | `/debug/last-frame` | JPEG of what the last `/login` call received |

### Face tracking

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/track` | Live tracking monitor UI |
| `GET` | `/track/status` | Pan/tilt angles, error offsets, FPS, simulate flag |
| `GET` | `/track/stream` | MJPEG stream of annotated camera feed |
| `GET` | `/track/snapshot` | Single annotated JPEG frame |
| `POST` | `/track/feed` | Receive frame from browser (browser-feed mode) |
| `GET` | `/track/source` | Which camera mode is active (`picamera2`/`opencv`/`browser`) |

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

### Servo control — lgpio (not pigpio)

Pi 5 uses the **RP1 southbridge chip** for GPIO. `pigpio` and `RPi.GPIO` have no driver for RP1.

| Library | Pi 1–4 | Pi 5 | Notes |
|---------|--------|------|-------|
| `pigpio` | ✅ | ❌ | DMA registers absent on RP1 |
| `RPi.GPIO` | ⚠️ jittery | ❌ | Software PWM, RP1 unsupported |
| `lgpio` | ✅ | ✅ | Pi Foundation official, no daemon needed |

```bash
sudo apt install python3-lgpio
```

Set `SIMULATE = False` in `servo.py` to enable real servo output.

```
Pan servo  signal → BCM GPIO 17  (physical pin 11)
Tilt servo signal → BCM GPIO 18  (physical pin 12)
Servo power (+)   → separate 5 V supply  ← do NOT use Pi's 5 V pin
Servo GND         → common GND with Pi
```

> Power servos from a dedicated 5 V supply. Two SG90s at stall current can brown-out the Pi and corrupt the SD card.

### PID tuning

Gains live at the top of `tracker.py`:

```python
PAN_PID_GAINS  = dict(kp=0.07, ki=0.0, kd=0.003, ...)
TILT_PID_GAINS = dict(kp=0.07, ki=0.0, kd=0.003, ...)
```

- Tracking **sluggish** → increase `kp` (e.g. 0.07 → 0.10)
- Mount **oscillates** → increase `kd` (e.g. 0.003 → 0.006)
- Keep `ki=0` until Kp/Kd are stable — integral windup causes hunting

### Performance

| Setting | Value | Reason |
|---------|-------|--------|
| Camera resolution | 640 × 480 | Enforced in `decode_image()` |
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

The script tests every video device × backend × codec combination, runs both Haar and face_recognition detection, and saves annotated PNG files so you can see exactly what the camera captures.

---

## Project structure

```
Face_Login/
├── main.py              # FastAPI app — all endpoints + tracker lifespan
├── tracker.py           # CameraManager + FaceTracker (Haar detection + PID loop)
├── pid.py               # Discrete PID controller with anti-windup
├── servo.py             # Servo abstraction — lgpio (Pi 5) or simulate mode
├── diagnose.py          # Camera + face detection diagnostic script
├── templates/
│   ├── index.html       # Guided registration + login UI → redirects to /track
│   └── track.html       # Live tracking monitor (MJPEG feed + servo compass)
├── faces/
│   └── db.json          # Face encoding database (auto-created)
├── debug_frames/        # Frames saved by diagnose.py (auto-created)
├── pyproject.toml
└── .python-version      # Pin to 3.12
```
