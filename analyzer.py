"""
analyzer.py — Three-stage emotion + scene analysis pipeline.

Stage 1 (HSEmotion): EfficientNet trained on AffectNet → precise emotion
                     from the identified user's face crop only.
Stage 2 (Moondream): VLM → free-form scene description of surroundings.
Stage 3 (Mistral):   LLM → combine emotion + scene → structured JSON.

Setup:
    pip install hsemotion-onnx
    uv add httpx

    export ANALYZER_VLM_MODEL=moondream
    export ANALYZER_LLM_MODEL=mistral:latest
    export OLLAMA_BASE_URL=http://localhost:11434
"""
import os
import re
import json
import base64
import logging
from typing import Optional, TypedDict

import cv2
import httpx
import numpy as np

log = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
VLM_MODEL       = os.environ.get("ANALYZER_VLM_MODEL", "moondream")
LLM_MODEL       = os.environ.get("ANALYZER_LLM_MODEL", "mistral:latest")
TIMEOUT_S       = int(os.environ.get("ANALYZER_TIMEOUT_S", "120"))

CROP_PADDING = 20    # pixels of padding around the bounding box
CROP_FALLBACK = 160  # size of fallback crop when no box available


class Analysis(TypedDict):
    emotion: str
    emotion_confidence: float
    emotion_scores: dict
    scene_description: str
    subjects: list
    emotion_affected_by_scene: bool
    reason: str
    summary: str


# ── HSEmotion ─────────────────────────────────────────────────────────────────

_emo_model = None

def _get_emo_model():
    global _emo_model
    if _emo_model is None:
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
            _emo_model = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
            log.info("HSEmotion model loaded.")
        except ImportError:
            raise ImportError("Run: pip install hsemotion-onnx")
    return _emo_model


def _crop_from_box(frame_rgb: np.ndarray, box: dict) -> np.ndarray:
    """
    Crop face region from bounding box with padding.
    box format: {"top": int, "right": int, "bottom": int, "left": int}
    Returns BGR crop for HSEmotion.
    """
    h, w = frame_rgb.shape[:2]
    top    = max(0, box["top"]    - CROP_PADDING)
    left   = max(0, box["left"]   - CROP_PADDING)
    bottom = min(h, box["bottom"] + CROP_PADDING)
    right  = min(w, box["right"]  + CROP_PADDING)

    crop = frame_rgb[top:bottom, left:right]
    if crop.size == 0:
        # Degenerate box — fall back to centre
        return _crop_from_centre(frame_rgb, {"x": w//2, "y": h//2})
    return cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)


def _crop_from_centre(frame_rgb: np.ndarray, position: dict) -> np.ndarray:
    """
    Fallback crop centred on position when no box is available.
    Returns BGR crop for HSEmotion.
    """
    h, w  = frame_rgb.shape[:2]
    cx    = position.get("x") or w // 2
    cy    = position.get("y") or h // 2
    half  = CROP_FALLBACK // 2
    x1    = max(0, cx - half)
    y1    = max(0, cy - half)
    x2    = min(w, cx + half)
    y2    = min(h, cy + half)
    crop  = frame_rgb[y1:y2, x1:x2]

    # Pad to square if edge was hit
    if crop.shape[0] < CROP_FALLBACK or crop.shape[1] < CROP_FALLBACK:
        padded = np.zeros((CROP_FALLBACK, CROP_FALLBACK, 3), dtype=np.uint8)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        crop = padded

    return cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)


def _run_hsemotion(
    frame_rgb: np.ndarray,
    box:       Optional[dict],
    position:  dict,
) -> tuple:
    """
    Run HSEmotion on the correct face crop.
    Uses bounding box if available, falls back to position centre.
    Returns (dominant_emotion, confidence, all_scores_dict).
    """
    model = _get_emo_model()

    if box is not None:
        log.info("using tracked bounding box for face crop")
        crop_bgr = _crop_from_box(frame_rgb, box)
    else:
        log.info("no bounding box — using position centre fallback")
        crop_bgr = _crop_from_centre(frame_rgb, position)

    emotion_label, scores = model.predict_emotions(crop_bgr, logits=False)

    labels = ["anger", "contempt", "disgust", "fear",
              "happiness", "neutral", "sadness", "surprise"]
    scores_dict = {l: round(float(s), 3) for l, s in zip(labels, scores)}
    confidence  = round(float(max(scores)), 3)

    return emotion_label.lower(), confidence, scores_dict


# ── Ollama helpers ────────────────────────────────────────────────────────────

def _chat(model: str, messages: list, json_mode: bool = False) -> str:
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": 0.2},
    }
    if json_mode:
        payload["format"] = "json"
    with httpx.Client(timeout=TIMEOUT_S) as client:
        r = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]


def _to_b64_jpeg(frame_rgb: np.ndarray, quality: int = 80) -> str:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


# ── Stage 2: Moondream → scene description ────────────────────────────────────

def _describe_scene(
    frame_rgb: np.ndarray,
    user_name: str,
    box:       Optional[dict],
    position:  dict,
) -> str:
    img_b64 = _to_b64_jpeg(frame_rgb)

    # Build position description for the prompt
    if box:
        cx = (box["left"] + box["right"]) // 2
        cy = (box["top"] + box["bottom"]) // 2
    else:
        cx = position.get("x")
        cy = position.get("y")

    pos_str = "in the frame"
    if cx is not None and cy is not None:
        h, w   = frame_rgb.shape[:2]
        h_part = "left" if cx < w/3 else "right" if cx > 2*w/3 else "centre"
        v_part = "upper" if cy < h/3 else "lower" if cy > 2*h/3 else "middle"
        pos_str = f"at the {v_part}-{h_part} of the frame"

    prompt = (
        f"The person named '{user_name}' is {pos_str}.\n\n"
        f"Describe only the surroundings and scene context — "
        f"objects, animals, other people, activities, environment. "
        f"Do NOT describe the person's emotion or facial expression. "
        f"Be specific and factual. Under 100 words."
    )
    return _chat(
        VLM_MODEL,
        [{"role": "user", "content": prompt, "images": [img_b64]}],
    ).strip()


# ── Stage 3: Mistral → structured JSON ───────────────────────────────────────

_SYSTEM = (
    "You combine an emotion reading and a scene description into a structured "
    "JSON object. Output ONLY valid JSON — no preamble, no markdown fences."
)

_SCHEMA = """{
  "emotion": "string — the dominant emotion",
  "emotion_confidence": 0.0..1.0,
  "emotion_scores": {"anger": 0.0, "contempt": 0.0, "disgust": 0.0, "fear": 0.0,
                     "happiness": 0.0, "neutral": 0.0, "sadness": 0.0, "surprise": 0.0},
  "scene_description": "1-2 sentences about the surroundings",
  "subjects": ["up to 5 short noun phrases for main entities"],
  "emotion_affected_by_scene": true|false,
  "reason": "brief causal sentence if affected, else empty string",
  "summary": "single short sentence e.g. user sad because cat on floor"
}"""


def _reason(
    emotion:    str,
    confidence: float,
    scores:     dict,
    scene:      str,
    user_name:  str,
    prev:       Optional[dict],
) -> "Analysis":
    prev_text = "(none — first observation)"
    if prev:
        prev_text = (
            f"emotion={prev.get('emotion','?')}, "
            f"subjects={prev.get('subjects',[])}, "
            f"summary={prev.get('summary','')!r}"
        )

    prompt = (
        f"User: {user_name}\n\n"
        f"Emotion from face recognition model:\n"
        f"  dominant = {emotion} (confidence={confidence:.0%})\n"
        f"  all scores = {json.dumps(scores)}\n\n"
        f"Scene description from vision model:\n---\n{scene}\n---\n\n"
        f"Previous snapshot: {prev_text}\n\n"
        f"Produce a JSON object with this schema:\n{_SCHEMA}\n\n"
        f"Rules:\n"
        f"- Use the emotion and scores exactly as provided.\n"
        f"- Set emotion_affected_by_scene=true ONLY when something in the scene "
        f"plausibly explains the emotion. A plain desk or wall does not count.\n"
        f"- Extract subjects from the scene description.\n"
        f"- Output JSON only."
    )
    raw = _chat(
        LLM_MODEL,
        [{"role": "system", "content": _SYSTEM},
         {"role": "user",   "content": prompt}],
        json_mode=True,
    )
    return _parse(raw, emotion, confidence, scores)


# ── Parse ─────────────────────────────────────────────────────────────────────

def _parse(raw: str, emotion: str, confidence: float, scores: dict) -> "Analysis":
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        log.warning("analyzer: JSON parse failed: %r", text[:200])
        return _empty(emotion=emotion, confidence=confidence, scores=scores,
                      reason=f"parse error: {text[:80]}")
    return Analysis(
        emotion                   = emotion,
        emotion_confidence        = confidence,
        emotion_scores            = scores,
        scene_description         = str(d.get("scene_description", "")),
        subjects                  = [str(s) for s in (d.get("subjects") or [])][:5],
        emotion_affected_by_scene = bool(d.get("emotion_affected_by_scene", False)),
        reason                    = str(d.get("reason", "")),
        summary                   = str(d.get("summary", "")),
    )


def _empty(emotion="neutral", confidence=0.0, scores=None, reason="") -> "Analysis":
    return Analysis(
        emotion="neutral", emotion_confidence=0.0,
        emotion_scores=scores or {},
        scene_description="", subjects=[],
        emotion_affected_by_scene=False,
        reason=reason, summary=""
    )


# ── Public entry point ────────────────────────────────────────────────────────

def analyze(
    frame_rgb:     np.ndarray,
    user_name:     str,
    box:           Optional[dict],   # {"top","right","bottom","left"} or None
    position:      dict,             # {"x","y"} centre fallback
    prev_snapshot: Optional[dict] = None,
) -> "Analysis":
    """
    Full three-stage analysis.

    Args:
        frame_rgb:     HxWx3 RGB numpy array from camera.
        user_name:     identified user name from /frames/event.
        box:           tracked face bounding box from /frames/event, or None.
        position:      face centre position as fallback when box is None.
        prev_snapshot: last timeline row for temporal context.
    """
    try:
        log.info("stage 1: HSEmotion...")
        emotion, confidence, scores = _run_hsemotion(frame_rgb, box, position)
        log.info("emotion: %s (%.0f%%)", emotion, confidence * 100)

        log.info("stage 2: moondream scene...")
        scene = _describe_scene(frame_rgb, user_name, box, position)
        log.debug("scene: %s", scene[:120])

        log.info("stage 3: mistral reasoning...")
        return _reason(emotion, confidence, scores, scene, user_name, prev_snapshot)

    except httpx.HTTPError as e:
        log.error("analyzer: HTTP error: %s", e)
        return _empty(reason=f"ollama error: {e}")
    except Exception as e:
        log.exception("analyzer: unexpected error")
        return _empty(reason=f"error: {e}")