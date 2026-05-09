# Elara — The Eye

> **Role in the multi-agent system:** This service is the perception layer — the eye. It runs locally on a Raspberry Pi 5, watches the physical world in front of it, identifies who is present, tracks where they are, reads their emotional state, and understands the scene around them. Every meaningful observation is logged as a snapshot and must be synced to the central multi-agent database running on RunPod so all cloud agents share the same ground truth about the user.

---

## What this is

Elara is a FastAPI microservice that does four things continuously:

1. **Presence detection** — knows when a registered user is in front of the camera
2. **Face tracking** — follows the user with a pan-tilt servo camera (via ESP32)
3. **Identity verification** — confirms *who* is present, not just *a* face
4. **Perception analysis** — reads emotion from the face, describes the scene, and reasons about causal links between environment and emotional state

The output of all four — who is present, where they are, what they feel, why — is written to a local SQLite timeline and must be **mirrored into the multi-agent system's database** on RunPod so every agent in the system can make decisions informed by real-world user state.

---

## Where this fits in the multi-agent system

```
┌─────────────────────────────────────────────────────────────┐
│                    RunPod (Cloud)                           │
│                                                             │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐              │
│   │  Agent A │   │  Agent B │   │  Agent C │  ...         │
│   └────┬─────┘   └────┬─────┘   └────┬─────┘              │
│        │              │              │                      │
│        └──────────────┴──────────────┘                     │
│                       │                                     │
│            ┌──────────▼──────────┐                         │
│            │  Central Agent DB   │  ← perception section   │
│            │  (perception table) │    written here         │
│            └──────────▲──────────┘                         │
│                       │  sync / push                        │
└───────────────────────┼─────────────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            │   Raspberry Pi 5      │
            │   (this service)      │
            │                       │
            │  ┌─────────────────┐  │
            │  │   Camera + Servos│  │
            │  └────────┬────────┘  │
            │           │           │
            │  ┌────────▼────────┐  │
            │  │  Face Tracker   │  │  ← 20 fps, Haar + PID
            │  └────────┬────────┘  │
            │           │           │
            │  ┌────────▼────────┐  │
            │  │  Identity Check │  │  ← dlib, every 3s
            │  └────────┬────────┘  │
            │           │           │
            │  ┌────────▼────────┐  │
            │  │  Analyzer       │  │  ← HSEmotion + Moondream + Mistral
            │  └────────┬────────┘  │
            │           │           │
            │  ┌────────▼────────┐  │
            │  │  timeline.db    │  │  ← local SQLite (source of truth)
            │  └─────────────────┘  │
            └───────────────────────┘
```

Cloud agents query the **perception section** of the central DB to know the current state of the user before deciding what to do. Elara is the only writer to this section.

---

## Perception data produced

Every time a meaningful change is detected (emotion shifts, scene changes, new subjects appear), a snapshot is recorded. Each snapshot contains:

| Field | Type | Description |
|---|---|---|
| `ts` | ISO-8601 UTC | When the snapshot was taken |
| `user` | string | Identified user name |
| `emotion` | string | Dominant emotion: `happiness`, `sadness`, `anger`, `fear`, `disgust`, `contempt`, `neutral`, `surprise` |
| `confidence` | float 0–1 | Emotion model confidence |
| `emotion_scores` | JSON object | Full probability distribution across all 8 emotions |
| `scene_description` | string | Free-form description of surroundings (objects, animals, environment) |
| `subjects` | JSON array | Up to 5 key entities in the scene |
| `affected` | bool | Whether the scene plausibly caused the emotion |
| `reason` | string | Causal explanation if `affected` is true |
| `summary` | string | One-sentence synthesis — e.g. `"user happy because dog is playing"` |
| `thumbnail` | BLOB | 320px-wide JPEG of the frame at snapshot time |

### Deduplication strategy

Elara does **not** write a row every N seconds blindly. A new row is written only when:
- Emotion label changes
- `affected` flag flips
- More than 50% of scene subjects change (Jaccard distance < 0.5)

This keeps the timeline lean — typically a few dozen rows per session, not thousands.

---

## Integration with the multi-agent DB on RunPod

### What needs to happen

The `faces/timeline.db` on the Pi is the local source of truth. The central DB on RunPod needs a **`perception_snapshots` table** (or equivalent) that mirrors this data. There are two integration patterns:

**Option A — Push (recommended):** After each `timeline.maybe_record()` call in `monitor.py`, push the new row to the RunPod DB via HTTP or direct DB connection. Low latency, no polling.

**Option B — Pull/Sync:** A RunPod agent periodically calls `GET /timeline` (to be added) and upserts rows by `(user, ts)` primary key. Simpler to implement, slight lag.

### Schema for the perception section of the central DB

```sql
CREATE TABLE perception_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,           -- ISO-8601 UTC from Pi
    user            TEXT    NOT NULL,           -- identified user name
    emotion         TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    emotion_scores  TEXT    NOT NULL,           -- JSON string
    scene           TEXT    NOT NULL,
    subjects        TEXT    NOT NULL,           -- JSON array string
    affected        INTEGER NOT NULL,           -- 0 or 1
    reason          TEXT    NOT NULL,
    summary         TEXT    NOT NULL,
    source_device   TEXT    DEFAULT 'pi',       -- for multi-device future
    synced_at       TEXT,                       -- when it arrived in central DB
    UNIQUE(user, ts)                            -- prevent duplicates on re-sync
);

CREATE INDEX idx_perception_user_ts ON perception_snapshots(user, ts);
CREATE INDEX idx_perception_emotion ON perception_snapshots(emotion);
```

### API endpoint for cloud agents to read current state

Add this to `main.py` (not yet implemented):

```
GET /timeline              → all users, newest first (JSON array)
GET /timeline/{user}       → one user, newest first
GET /timeline/{user}/latest → single most recent snapshot for quick polling
```

### What cloud agents should do with this data

Each agent should read the most recent perception snapshot for the active user before generating a response or taking an action. The `summary` field is the fastest signal; use `emotion_scores` for nuance.

Example agent prelude:
```
User: abhi
Current state: sadness (87% confidence)
Scene: sitting at desk, cat on floor nearby
Cause: user sad because cat is on the floor
→ Adjust tone: be gentle, avoid high-energy responses
```

---

## How the perception pipeline works

```
/frames/event  (GET, atomic JSON)
      │
      ▼
monitor.py  [polls every PERIOD_S seconds, skips if no identified user]
      │
      ├─── Stage 1: HSEmotion (EfficientNet-B0, face crop)
      │           → emotion label + confidence + 8-class scores
      │
      ├─── Stage 2: Moondream VLM (full frame)
      │           → scene description in natural language
      │
      └─── Stage 3: Mistral LLM (emotion + scene combined)
                  → structured JSON: subjects, affected flag, reason, summary
                        │
                        ▼
                  timeline.py  [write only if meaningfully changed]
                        │
                        ▼
                  faces/timeline.db  (SQLite)
                        │
                        ▼
               ── sync ──► RunPod central DB
```

---

## Running the service

**Requirements:** Python ≥ 3.12, `uv`, Ollama running locally with `moondream` and `mistral:latest`

```bash
cd Face_login_Elara-master
uv sync

# Start the main server
uv run uvicorn main:app --host 0.0.0.0 --port 8765

# Start the perception monitor (separate process)
python monitor.py --period 60

# Point monitor at the Pi from another machine
PI_URL=http://<pi-ip>:8765 python monitor.py --period 30
```

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `PI_URL` | `http://localhost:8765` | Where monitor.py finds the camera service |
| `MONITOR_PERIOD_S` | `60` | Seconds between perception ticks |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama instance for VLM + LLM |
| `ANALYZER_VLM_MODEL` | `moondream` | Vision model for scene description |
| `ANALYZER_LLM_MODEL` | `mistral:latest` | Language model for structured reasoning |
| `DEVICE_MODE` | auto | `pi` or `laptop` — forces camera mode |

---

## API reference

### Face database

| Method | Path | Description |
|---|---|---|
| `GET` | `/faces` | List all registered names |
| `POST` | `/register` | Register a new face — `{name, images: [base64, ...]}` |
| `DELETE` | `/faces/{name}` | Delete a face |

### Authentication

| Method | Path | Description |
|---|---|---|
| `POST` | `/login` | Match face → `{success, name, token, confidence}`. On success, tells tracker who to watch for. |

### Tracking — servo state

| Method | Path | Description |
|---|---|---|
| `GET` | `/track/status` | Pan/tilt angles, PID errors, FPS |
| `GET` | `/track/snapshot` | Annotated JPEG with tracking overlay |
| `GET` | `/track/stream` | MJPEG stream at 20 fps with overlay |
| `POST` | `/track/feed` | Push a browser frame (browser-feed mode) |

### Identity (Channel 1 — low bandwidth)

Poll these every few seconds from cloud agents that only need identity, not the image.

| Method | Path | Response |
|---|---|---|
| `GET` | `/track/person` | `{person, position, identified_person_present, identified_name, last_recognition_ts}` |
| `GET` | `/track/identity` | Who the tracker is watching for |
| `DELETE` | `/track/identity` | Clear identity; recognition worker idles |

### Frames (Channel 2 — for agents that need the image)

| Method | Path | Description |
|---|---|---|
| `GET` | `/frames/event` | Atomic JSON: `{timestamp, frame:{base64,...}, person, position, box}` — use this for analyzer |
| `GET` | `/frames/current` | Raw JPEG + identity in response headers — bandwidth-efficient |
| `GET` | `/frames/stream` | Continuous MJPEG, no overlay |

`/frames/event` response shape:
```json
{
  "timestamp": 1715000000.123,
  "frame": {"base64": "...", "width": 640, "height": 480},
  "person": "abhi",
  "identified_name": "abhi",
  "identified_person_present": true,
  "position": {"x": 312, "y": 240},
  "box": {"top": 180, "right": 380, "bottom": 300, "left": 250}
}
```

### Timeline (to be added for cloud sync)

| Method | Path | Description |
|---|---|---|
| `GET` | `/timeline` | All snapshots, newest first |
| `GET` | `/timeline/{user}` | Snapshots for one user |
| `GET` | `/timeline/{user}/latest` | Single most recent snapshot |

### Misc

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Login / registration web UI |
| `GET` | `/track` | Live tracking monitor page |
| `GET` | `/health` | `{"status": "ok"}` |
| `GET` | `/config` | Mode, is_pi flag, camera source |

---

## Hardware

- **Raspberry Pi 5** — runs this service, camera, and Ollama
- **Pi Camera Module** — captured via libcamera / picamera2
- **Pan-tilt servo mount** — two hobby servos (pan + tilt)
- **ESP32** — receives servo commands over USB serial (115200 baud), drives PWM
- **Laptop mode** — works without Pi or servos; browser sends frames via POST

### Tracking tuning (config.py)

| Setting | Default | Effect |
|---|---|---|
| `DEAD_ZONE_PX` | `30` | Pixels off-centre before servo moves |
| `PID_KP` | `0.012` | Proportional gain — higher = faster |
| `PID_KD` | `0.003` | Derivative gain — higher = less oscillation |
| `SLEW_MAX_DEG` | `2.0` | Max servo jump per tick |
| `TARGET_FPS` | `20` | Tracker loop rate |
| `IDENTITY_CHECK_INTERVAL_S` | `3.0` | How often dlib re-confirms identity |
| `IDENTITY_TOLERANCE` | `0.50` | Face match strictness (lower = stricter) |

---

## Project structure

```
Face_login_Elara-master/
├── main.py              # FastAPI app — all HTTP endpoints
├── tracker.py           # CameraManager singleton + FaceTracker threads
├── config.py            # All tunable parameters
├── pid.py               # Discrete PID with anti-windup
├── servo.py             # ESP32 serial abstraction + simulation mode
├── analyzer.py          # 3-stage perception pipeline (HSEmotion + Moondream + Mistral)
├── monitor.py           # Polls /frames/event, runs analyzer, stores timeline
├── timeline.py          # SQLite persistence with smart deduplication
├── webcam_injector.py   # Feed laptop webcam frames to server
├── diagnose.py          # Camera and detection diagnostics
├── faces/
│   ├── db.json          # Face encoding store
│   └── timeline.db      # Perception snapshot history
├── templates/
│   ├── index.html       # Login / registration page
│   └── track.html       # Live tracking monitor
└── esp32_servo/
    └── src/main.cpp     # ESP32 firmware — serial parser + PWM servo
```

---

## Graceful degradation

| Missing component | Behaviour |
|---|---|
| No ESP32 | Servo commands simulated (`[SIM]` shown in stream overlay) |
| No Pi camera | Browser-feed mode — login and tracking still work via web UI |
| Ollama not running | Analyzer returns empty/neutral; timeline still records presence |
| Tracker fails to start | `/track/*` and `/frames/*` return 503; login/register still work |
