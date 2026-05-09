"""
monitor.py — Emotion + scene monitor.

Polls GET /frames/event every PERIOD_S seconds.
Extracts: frame, identified user, bounding box, position.
Runs: HSEmotion (face crop) → Moondream (scene) → Mistral (JSON).
Stores: timeline.db via timeline.py.

Usage:
    python monitor.py                   # continuous, default 60s
    python monitor.py --period 30       # every 30s
    python monitor.py --once            # one tick and exit
    python monitor.py --history         # print stored timeline
    python monitor.py --history --user abhi
"""
import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
from typing import Optional

import cv2
import httpx
import numpy as np

import analyzer
import timeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("monitor")

PI_URL   = os.environ.get("PI_URL", "http://localhost:8765")
PERIOD_S = int(os.environ.get("MONITOR_PERIOD_S", "60"))
THUMB_W  = 320


# ── Fetch ─────────────────────────────────────────────────────────────────────

def _fetch_event(pi_url: str) -> Optional[dict]:
    try:
        with httpx.Client(timeout=10) as client:
            r = client.get(f"{pi_url}/frames/event")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        log.warning("server error: %s", e.response.status_code)
        return None
    except Exception as e:
        log.error("cannot reach server: %s", e)
        return None


def _decode_frame(event: dict) -> Optional[np.ndarray]:
    try:
        b64       = event["frame"]["base64"]
        img_bytes = base64.b64decode(b64)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        bgr       = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
    except Exception as e:
        log.error("frame decode: %s", e)
        return None


def _make_thumbnail(frame_rgb: np.ndarray) -> Optional[bytes]:
    h, w = frame_rgb.shape[:2]
    if w > THUMB_W:
        frame_rgb = cv2.resize(frame_rgb, (THUMB_W, int(h * THUMB_W / w)))
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes() if ok else None


# ── One tick ──────────────────────────────────────────────────────────────────

def tick(pi_url: str) -> dict:
    t0 = time.monotonic()

    # 1. Fetch event from server
    event = _fetch_event(pi_url)
    if event is None:
        return {"ok": False, "reason": "could not fetch /frames/event"}

    # 2. Get identified user — use identified_name (set by login) as primary,
    #    fall back to tracked_label (set by recognition worker)
    user = event.get("identified_name") or event.get("person")
    if not user or user == "Unknown":
        return {"ok": False, "reason": f"no identified user (got {user!r})"}

    # 3. Decode frame
    frame = _decode_frame(event)
    if frame is None:
        return {"ok": False, "reason": "frame decode failed"}

    # 4. Extract bounding box and position
    box      = event.get("box")       # {"top","right","bottom","left"} or None
    pos      = event.get("position") or {}
    position = {
        "x": pos.get("x") or frame.shape[1] // 2,
        "y": pos.get("y") or frame.shape[0] // 2,
    }

    if box:
        log.info("tick → user=%-12s  box=%s", user, box)
    else:
        log.info("tick → user=%-12s  pos=%s  (no box yet)", user, position)

    # 5. Previous snapshot for temporal context
    prev = timeline.last_for(user)

    # 6. Analyze — passes box for precise crop
    result = analyzer.analyze(
        frame, user,
        box=box,
        position=position,
        prev_snapshot=prev,
    )

    # 7. Store if changed
    thumb  = _make_thumbnail(frame)
    rec_id = timeline.maybe_record(user, result, thumbnail=thumb)

    elapsed = time.monotonic() - t0
    return {
        "ok":        True,
        "user":      user,
        "emotion":   result["emotion"],
        "confidence": result["emotion_confidence"],
        "scores":    result["emotion_scores"],
        "scene":     result["scene_description"],
        "affected":  result["emotion_affected_by_scene"],
        "reason":    result["reason"],
        "summary":   result["summary"],
        "recorded":  rec_id,
        "had_box":   box is not None,
        "elapsed_s": round(elapsed, 1),
    }


# ── Pretty print ──────────────────────────────────────────────────────────────

def _print_result(r: dict) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    if not r["ok"]:
        print(f"  SKIP  {r['reason']}")
        print(sep)
        return

    rec     = f"→ snapshot #{r['recorded']}" if r["recorded"] else "→ no change"
    box_tag = "✓ box" if r.get("had_box") else "~ centre fallback"

    print(f"  USER     {r['user']}  [{box_tag}]")
    print(f"  EMOTION  {r['emotion'].upper()}  ({r['confidence']:.0%})")

    scores = r.get("scores", {})
    if scores:
        print("  SCORES")
        for label, score in sorted(scores.items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 20)
            print(f"    {label:10s} {bar:20s} {score:.3f}")

    print(f"  SCENE    {r['scene']}")
    if r["affected"]:
        print(f"  CAUSE    {r['reason']}")
    print(f"  SUMMARY  {r['summary']}")
    print(f"  {rec}  ({r['elapsed_s']}s)")
    print(sep)


# ── Timeline display ──────────────────────────────────────────────────────────

def print_timeline(user: Optional[str] = None, limit: int = 20) -> None:
    rows = timeline.recent(user=user, limit=limit)
    if not rows:
        print("No snapshots recorded yet.")
        return
    sep = "─" * 70
    print(f"\n{'TIMELINE':^70}\n{sep}")
    for r in reversed(rows):
        tag = "⚡" if r["emotion_affected"] else "  "
        print(
            f"  {r['ts']}  {r['user']:12s}  "
            f"{r['emotion']:10s}  {tag}  {r['summary']}"
        )
    print(f"{sep}\n  {len(rows)} snapshot(s)")


# ── Loop ──────────────────────────────────────────────────────────────────────

async def run_loop(pi_url: str, period_s: int) -> None:
    print(f"\n{'='*60}")
    print(f"  EMOTION MONITOR")
    print(f"  Server  : {pi_url}")
    print(f"  Period  : {period_s}s")
    print(f"  VLM     : {analyzer.VLM_MODEL}")
    print(f"  LLM     : {analyzer.LLM_MODEL}")
    print(f"{'='*60}\n")
    log.info("running — Ctrl+C to stop")

    n = 0
    while True:
        n += 1
        log.info("─── tick #%d ───", n)
        try:
            result = await asyncio.to_thread(tick, pi_url)
            _print_result(result)
        except Exception as e:
            log.exception("tick crashed: %s", e)
        log.info("next tick in %ds", period_s)
        await asyncio.sleep(period_s)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",     default=PI_URL)
    parser.add_argument("--period",  default=PERIOD_S, type=int)
    parser.add_argument("--once",    action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--user",    default=None)
    args = parser.parse_args()

    if args.history:
        print_timeline(user=args.user)
        sys.exit(0)

    if args.once:
        r = tick(args.url)
        _print_result(r)
        sys.exit(0 if r["ok"] else 1)

    asyncio.run(run_loop(args.url, args.period))