"""
webcam_injector.py — Feeds laptop webcam frames to the server.
With --autologin, automatically logs in using a captured face frame
so no browser is needed at all.

Usage:
    python webcam_injector.py                   # just feed frames
    python webcam_injector.py --autologin       # login + feed frames
    python webcam_injector.py --url http://localhost:8765 --autologin
"""
import argparse
import base64
import time
import sys

import cv2
import httpx

SERVER = "http://localhost:8765"
FPS    = 15


def capture_frame(cap: cv2.VideoCapture) -> bytes:
    _, buf = cv2.imencode(
        ".jpg",
        cv2.flip(cap.read()[1], 1),
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    return base64.b64encode(buf.tobytes()).decode()


def autologin(server_url: str, cap: cv2.VideoCapture) -> bool:
    print("Auto-login: capturing face...")

    # Warm up camera
    for _ in range(20):
        cap.read()

    b64 = capture_frame(cap)

    try:
        with httpx.Client(timeout=15) as client:
            r = client.post(
                f"{server_url}/login",
                json={"image": f"data:image/jpeg;base64,{b64}"},
            )
            data = r.json()
            if data.get("success"):
                print(f"  Logged in as: {data['name']} "
                      f"(confidence: {data['confidence']}%)")
                return True
            else:
                print(f"  Login failed: {data.get('message', 'face not recognized')}")
                print("  Make sure you are registered and facing the camera.")
                return False
    except Exception as e:
        print(f"  Login error: {e}")
        return False


def run(server_url: str, do_autologin: bool) -> None:
    print(f"Opening webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ok, test = cap.read()
    if not ok or test is None:
        print("ERROR: Could not open webcam")
        sys.exit(1)

    print(f"Webcam ready: {int(cap.get(3))}x{int(cap.get(4))}")

    # Auto-login if requested
    if do_autologin:
        success = autologin(server_url, cap)
        if not success:
            print("\nContinuing anyway — monitor will skip ticks until login succeeds.")
        print()

    feed_url  = f"{server_url}/track/feed"
    interval  = 1.0 / FPS
    frame_count = 0

    print(f"Feeding frames to {feed_url} at {FPS} fps")
    print("Press Ctrl+C to stop\n")

    with httpx.Client(timeout=5) as client:
        while True:
            t0 = time.monotonic()

            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            _, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            b64 = base64.b64encode(buf.tobytes()).decode()

            try:
                client.post(
                    feed_url,
                    json={"image": f"data:image/jpeg;base64,{b64}"},
                )
                frame_count += 1
                if frame_count % (FPS * 10) == 0:
                    print(f"  {frame_count} frames sent")
            except httpx.ConnectError:
                print("Waiting for server...")
                time.sleep(2)
            except Exception as e:
                print(f"WARNING: {e}")

            elapsed = time.monotonic() - t0
            time.sleep(max(0, interval - elapsed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",       default=SERVER)
    parser.add_argument("--autologin", action="store_true",
                        help="Login automatically using webcam before feeding frames")
    args = parser.parse_args()
    try:
        run(args.url, args.autologin)
    except KeyboardInterrupt:
        print("\nStopped.")