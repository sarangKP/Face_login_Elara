"""
Test the full analysis pipeline using laptop webcam directly.
No Pi server needed. Run: python test_webcam.py
"""
import cv2, json, analyzer, timeline

print("Opening webcam...")
cap = cv2.VideoCapture(0)
for _ in range(15):      # warm up
    cap.read()

ok, frame_bgr = cap.read()
cap.release()

if not ok:
    print("ERROR: could not read from webcam")
    exit(1)

frame_rgb = cv2.cvtColor(cv2.flip(frame_bgr, 1), cv2.COLOR_BGR2RGB)
print(f"Frame captured: {frame_rgb.shape}")

# Hardcode your name since we skip face recognition here
user     = "abhi"
position = {"x": 320, "y": 240}   # assume face is roughly centred

prev = timeline.last_for(user)
print(f"Previous snapshot: {prev['summary'] if prev else '(none)'}")
print("Analyzing... (first call: 60-90s, subsequent: ~15s)")

result = analyzer.analyze(frame_rgb, user, position, prev_snapshot=prev)
print(json.dumps(result, indent=2))

rec_id = timeline.maybe_record(user, result)
print(f"\n→ Recorded as #{rec_id}" if rec_id else "\n→ No change, not recorded")
