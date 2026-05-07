# Face Login — Elara Service

A self-contained FastAPI microservice that runs on a Raspberry Pi 5 (or any laptop for dev). It handles face registration, face login, real-time pan-tilt servo tracking, and — after login — continuous identity verification so cloud agents always know who is in front of the camera.

---

## What it does

```
Browser / cloud agent
        │
        │  POST /login  ──► face match → name + token
        │                          │
        │                          ▼
        │               FaceTracker.set_identified_person("Alice")
        │                          │
        │  GET /track/person ◄──── │  (poll every N seconds — Channel 1)
        │  GET /frames/current ◄── │  (raw JPEG + headers      — Channel 2)
        │  GET /frames/event ◄──── │  (JSON + base64 JPEG      — Channel 2)
        │  GET /frames/stream ◄─── │  (MJPEG stream            — Channel 2)
```

---

## Architecture

### Threads

| Thread | Rate | Job |
|---|---|---|
| `CamCapture` | ~120 fps | Reads picamera2 (Pi) or accepts browser POSTs; keeps `_latest_frame` fresh |
| `FaceTracker` | 20 fps | Haar cascade → largest face → PID → servo move; writes `TrackerState` |
| `FaceRecognition` | every 3 s | dlib encoder → compares against login encodings; labels tracked face; never blocks the 20 fps loop |

### Why two detectors?

- **Haar Cascade** (tracking loop, 20 fps): ~10–30 ms/frame — fast enough for real-time servo control.
- **dlib encoder via face_recognition** (recognition worker, every 3 s): ~150–300 ms/frame on Pi 5 — accurate enough for identity, too slow for tracking.

### Camera modes

| Mode | How it works |
|---|---|
| `picamera2` (Pi) | libcamera stack; frame captured in background thread |
| `browser` (laptop dev) | Browser posts frames via `POST /track/feed`; no hardware camera needed |

`CameraManager` is a singleton — opened once, shared by the tracker and all frame endpoints.

---

## Setup

**Requirements:** Python ≥ 3.12, `uv`

```bash
git clone <repo>
cd Face_Login
uv sync
```

`face-recognition-models` is installed from git source (see `pyproject.toml`). No extra steps needed.

**Run:**

```bash
# Auto-detect mode (Pi camera → pi, else → laptop)
uv run uvicorn main:app --host 0.0.0.0 --port 8765

# Force a mode
DEVICE_MODE=laptop uv run uvicorn main:app --host 0.0.0.0 --port 8765
DEVICE_MODE=pi     uv run uvicorn main:app --host 0.0.0.0 --port 8765
```

Open `http://<pi-ip>:8765` to register and log in.

---

## Configuration

All tunable parameters live in [`config.py`](config.py). No need to touch `tracker.py` or `servo.py`.

### Tracking tuning

| Setting | Default | Effect |
|---|---|---|
| `DEAD_ZONE_PX` | `30` | Face must move this many pixels off-centre before servo reacts |
| `PID_KP` | `0.012` | Proportional gain — higher = faster tracking |
| `PID_KD` | `0.003` | Derivative gain — increase if servo hunts/oscillates |
| `PID_KI` | `0.0` | Integral gain — leave at 0 until KP/KD are stable |
| `PID_OUTPUT_LIMIT_PAN` | `1.5` | Max degrees/tick the PID can request (pan) |
| `PID_OUTPUT_LIMIT_TILT` | `1.2` | Max degrees/tick the PID can request (tilt) |
| `SLEW_MAX_DEG` | `2.0` | Hard cap in `servo.py` — servo never jumps more than this per call |
| `TARGET_FPS` | `20` | Tracker loop rate |

Quick fixes:

```
Tracking too slow    → increase PID_KP  (0.012 → 0.020)
Tracking too fast    → decrease PID_KP  (0.012 → 0.008)
Servo oscillates     → increase PID_KD  (0.003 → 0.006)
Jitter at centre     → increase DEAD_ZONE_PX (30 → 40)
Movement too jumpy   → decrease SLEW_MAX_DEG (2.0 → 1.0)
Movement too laggy   → increase SLEW_MAX_DEG (2.0 → 3.5)
```

### Identity bridge tuning

| Setting | Default | Effect |
|---|---|---|
| `IDENTITY_CHECK_INTERVAL_S` | `3.0` | How often the recognition worker re-checks the frame. Don't go below ~2 s on Pi 5 |
| `IDENTITY_TOLERANCE` | `0.50` | Face match strictness — lower = stricter. Shared by both `/login` and the recognition worker |
| `IDENTITY_MAX_FACES` | `5` | Max faces encoded per recognition pass — protects against crowded frames |

---

## API Reference

### Face DB

| Method | Path | Description |
|---|---|---|
| `GET` | `/faces` | List all registered names |
| `POST` | `/register` | Register a new face (JSON: `{name, images: [base64, ...]}`) |
| `DELETE` | `/faces/{name}` | Delete a registered face |

### Authentication

| Method | Path | Description |
|---|---|---|
| `POST` | `/login` | Match a face image; returns `{success, name, token, confidence}`. On success, notifies the tracker. |

### Tracking — servo state

| Method | Path | Description |
|---|---|---|
| `GET` | `/track/status` | Current pan/tilt angles, PID errors, FPS |
| `GET` | `/track/snapshot` | Latest annotated frame as JPEG |
| `GET` | `/track/stream` | MJPEG stream with tracker overlay (~20 fps) |
| `POST` | `/track/feed` | Push a browser frame into the camera buffer (browser-feed mode) |
| `GET` | `/track/source` | Whether camera source is `picamera2` or `browser` |

### Tracking — identity (Channel 1)

Poll these from cloud agents on a slow cadence (every few seconds).

| Method | Path | Description |
|---|---|---|
| `GET` | `/track/person` | Who is being tracked, their position, whether the logged-in person is present anywhere in frame |
| `GET` | `/track/identity` | Who the tracker is currently watching for |
| `DELETE` | `/track/identity` | Clear the logged-in identity; recognition worker idles |

`GET /track/person` response:

```json
{
  "person": "Alice",
  "position": {"x": 312, "y": 240},
  "timestamp": 1715000000,
  "frame_size": {"width": 640, "height": 480},
  "identified_person_present": true,
  "identified_name": "Alice",
  "last_recognition_ts": 1715000000.123
}
```

`person` values: `"Alice"` (confirmed), `"Unknown"` (unrecognised face), `null` (no face / recognition not yet run).

### Frames (Channel 2)

Raw camera feed without tracking overlay — for cloud / LLM agents that need to see the image.

| Method | Path | Description |
|---|---|---|
| `GET` | `/frames/current` | Raw JPEG + identity metadata in response headers |
| `GET` | `/frames/event` | Atomic JSON: `{timestamp, frame: {base64, ...}, person, position, ...}` |
| `GET` | `/frames/stream` | Continuous raw MJPEG (no overlay) |

`/frames/current` response headers:

```
X-Timestamp                 — unix epoch seconds
X-Tracked-Person            — name | "Unknown" | "" (empty = no face)
X-Identified-Name           — who login set (empty if nobody is logged in)
X-Identified-Person-Present — "true" / "false"
X-Tracked-Position          — "x,y" in 640×480 pixels, or empty
X-Frame-Width / X-Frame-Height
```

Use `/frames/current` when bandwidth matters (just read the headers). Use `/frames/event` when an LLM agent needs frame + identity in one parseable object.

### Camera (raw feed for login page)

| Method | Path | Description |
|---|---|---|
| `GET` | `/camera/stream` | Raw MJPEG, no overlay — used by the login page preview |
| `GET` | `/camera/frame` | Single raw JPEG — used by Pi login capture |

### Misc

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Login / registration web UI |
| `GET` | `/track` | Live tracking monitor page |
| `GET` | `/config` | Current mode, is_pi flag, camera source |
| `GET` | `/health` | `{"status": "ok"}` |
| `GET` | `/debug/last-frame` | Last image received by `/login` — remove before production |

---

## Visual overlay guide

The annotated stream (`/track/stream`) shows:

| Box colour | Meaning |
|---|---|
| **Cyan** | Tracked face confirmed as the logged-in person ✓ |
| **Red** | Tracked face is someone else ("Unknown") |
| **Green** | Face detected; recognition hasn't run yet |

Top-left banner states:
- `Tracking: Alice` — Alice is the largest face being actively tracked
- `Alice present (not tracked)` — Alice is in the background; a different person is being tracked
- `Looking for: Alice` — Nobody recognised yet this cycle

Bottom-left HUD: `[SIM]` or `[LIVE]`, current pan/tilt angles, loop FPS.

---

## Graceful degradation

| Missing hardware | Behaviour |
|---|---|
| No ESP32 | Servo commands are simulated (`[SIM]` in overlay); everything else works |
| No Pi camera | Switches to browser-feed mode automatically; login and tracking still work via the web UI |
| Tracker fails to start | All `/track/*` and `/frames/*` endpoints return `503`; `/login` and `/register` still work |

---

## Project structure

```
Face_Login/
├── main.py          # FastAPI app — all HTTP endpoints
├── tracker.py       # CameraManager singleton + FaceTracker daemon threads
├── config.py        # All tunable parameters — edit here, not in tracker.py
├── pid.py           # Discrete PID controller with anti-windup
├── servo.py         # ServoController — ESP32 serial abstraction
├── diagnose.py      # Standalone diagnostic tool
├── faces/
│   └── db.json      # Face encoding store — name → list of 128-float vectors
├── templates/
│   ├── index.html   # Login / registration page
│   └── track.html   # Live tracking monitor
└── esp32_servo/     # ESP32 firmware (PlatformIO)
    └── src/main.cpp
```
