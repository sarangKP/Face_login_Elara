"""
Camera + Face Detection Diagnostics
Run: uv run python diagnose.py
Saves captured frames as PNG files so you can actually see what the camera sees.
"""
import cv2
import numpy as np
from pathlib import Path

OUT = Path("debug_frames")
OUT.mkdir(exist_ok=True)

SEP = "─" * 60

def save(name, frame):
    p = OUT / name
    cv2.imwrite(str(p), frame)
    print(f"   → saved: {p}")

# ─────────────────────────────────────────────────────────────
print(SEP)
print("1. SCANNING VIDEO DEVICES")
print(SEP)
import glob
devs = sorted(glob.glob("/dev/video*"))
print(f"   Found: {devs}")

# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("2. TESTING EACH DEVICE × EACH BACKEND")
print(SEP)

backends = [
    ("V4L2",  cv2.CAP_V4L2),
    ("FFMPEG", cv2.CAP_FFMPEG),
    ("ANY",   cv2.CAP_ANY),
]
codecs = [
    ("MJPG", cv2.VideoWriter_fourcc(*'MJPG')),
    ("YUYV", cv2.VideoWriter_fourcc(*'YUYV')),
]

working = []   # (index, backend_name, codec_name, cap)

for dev in devs:
    idx = int(dev.replace("/dev/video", ""))
    for bname, backend in backends:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            print(f"   video{idx} + {bname}: FAILED to open")
            cap.release()
            continue

        for cname, fourcc in codecs:
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # warm up
            for _ in range(5):
                cap.read()

            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"   video{idx} + {bname} + {cname}: opened but read FAILED")
                continue

            mean = frame.mean()
            h, w = frame.shape[:2]
            print(f"   video{idx} + {bname} + {cname}: OK  {w}×{h}  brightness={mean:.1f}")
            fname = f"video{idx}_{bname}_{cname}.png"
            save(fname, frame)
            working.append((idx, bname, cname, mean))

        cap.release()
        break   # don't re-open same device with multiple backends

# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("3. FACE DETECTION on best frame")
print(SEP)

if not working:
    print("   ✗ No working camera found at all.")
    print("   Check: is another process using it?")
    import subprocess
    result = subprocess.run(["fuser", "/dev/video0", "/dev/video1"],
                            capture_output=True, text=True)
    print(f"   fuser output: '{result.stdout.strip()}' (PIDs holding camera)")
    exit(1)

# Pick brightest frame
best = max(working, key=lambda x: x[3])
idx, bname, cname, mean = best
print(f"   Best source: video{idx} + {bname} + {cname} (brightness={mean:.1f})")

# Re-open and do a proper warm-up (20 frames)
fourcc = cv2.VideoWriter_fourcc(*cname)
cap = cv2.VideoCapture(idx, cv2.CAP_V4L2 if bname == "V4L2" else cv2.CAP_ANY)
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("   Warming up (20 frames)…")
for _ in range(20):
    cap.read()
ok, frame = cap.read()
cap.release()

if not ok:
    print("   ✗ Warm-up read failed")
    exit(1)

mean_after = frame.mean()
print(f"   Brightness after warm-up: {mean_after:.1f}")
save("warmup_frame.png", frame)

# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("4. FACE_RECOGNITION detection test")
print(SEP)

import face_recognition
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

for upsample in [1, 2]:
    locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=upsample)
    print(f"   upsample={upsample}: {len(locs)} face(s) found  {locs}")
    if locs:
        for i, (top, right, bottom, left) in enumerate(locs):
            annotated = frame.copy()
            cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 0), 2)
            save(f"detected_upsample{upsample}.png", annotated)

# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("5. HAAR CASCADE detection test (tracker uses this)")
print(SEP)

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
small = cv2.resize(frame, (320, 240))
gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
cv2.equalizeHist(gray, gray)

for scale in [1.1, 1.15, 1.2]:
    for neighbours in [3, 4, 5]:
        faces = cascade.detectMultiScale(gray, scaleFactor=scale,
                                         minNeighbors=neighbours, minSize=(30, 30))
        if len(faces):
            print(f"   scale={scale} neighbours={neighbours}: {len(faces)} face(s) ✓")
            annotated = small.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            save(f"haar_s{scale}_n{neighbours}.png", annotated)
        else:
            print(f"   scale={scale} neighbours={neighbours}: 0 faces")

# ─────────────────────────────────────────────────────────────
print()
print(SEP)
print(f"DONE — check the '{OUT}/' folder for captured frames")
print(SEP)
