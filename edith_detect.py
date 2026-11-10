"""Open EDITH's camera and detect hands. Optionally label everyday objects too."""

import argparse
from pathlib import Path
from time import monotonic_ns

import cv2
import mediapipe as mp
import requests


WINDOW_NAME = "EDITH Detection - press Q to close"
MODEL_PATH = Path(__file__).resolve().parent / "yolo11n.pt"
HAND_MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
# Each pair joins two dots on the hand, forming the finger outlines.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def create_hand_detector():
    """Download the ready-trained hand model once, then load it."""
    if not HAND_MODEL_PATH.is_file():
        print("Downloading the hand detector for the first run...", flush=True)
        temporary_path = HAND_MODEL_PATH.with_suffix(".task.part")
        try:
            with requests.get(HAND_MODEL_URL, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                with temporary_path.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=65536):
                        output.write(chunk)
            temporary_path.replace(HAND_MODEL_PATH)
        finally:
            temporary_path.unlink(missing_ok=True)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def draw_hands(picture, result):
    """Draw a box, finger lines, and dots for each detected hand."""
    height, width = picture.shape[:2]
    for hand in result.hand_landmarks:
        points = [
            (max(0, min(width - 1, int(dot.x * width))),
             max(0, min(height - 1, int(dot.y * height))))
            for dot in hand
        ]
        x1 = max(0, min(x for x, y in points) - 15)
        y1 = max(0, min(y for x, y in points) - 15)
        x2 = min(width - 1, max(x for x, y in points) + 15)
        y2 = min(height - 1, max(y for x, y in points) + 15)
        cv2.rectangle(picture, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(picture, "Hand", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        for start, end in HAND_CONNECTIONS:
            cv2.line(picture, points[start], points[end], (0, 220, 255), 2)
        for point in points:
            cv2.circle(picture, point, 4, (255, 100, 0), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera number (default: 0)")
    parser.add_argument("--objects", action="store_true", help="Also label everyday objects")
    args = parser.parse_args()

    print("Loading EDITH's hand detector...", flush=True)
    object_model = None
    try:
        if args.objects:
            from ultralytics import YOLO
            object_model = YOLO(str(MODEL_PATH))
        hand_detector = create_hand_detector()
    except Exception as error:
        print(f"Could not load the detector: {error}", flush=True)
        print("Check your internet connection if this is the first run.", flush=True)
        return 1

    camera = None
    try:
        camera = cv2.VideoCapture(args.camera)
        if not camera.isOpened():
            print("Camera unavailable. Close other camera apps and check camera access in "
                  "System Settings > Privacy & Security > Camera.", flush=True)
            return 1

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("EDITH detection is online! Press Q or Escape in the camera window to close.",
              flush=True)
        print("Live video is processed on this computer; this script does not save it.",
              flush=True)
        first_frame = True
        last_timestamp = -1

        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read the camera. Check that it is connected.", flush=True)
                return 1

            # Mirror the view so moving your hand feels like looking in a mirror.
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp = max(last_timestamp + 1, monotonic_ns() // 1_000_000)
            last_timestamp = timestamp
            hand_result = hand_detector.detect_for_video(image, timestamp)
            picture = frame.copy()
            if object_model is not None:
                result = object_model.predict(frame, conf=0.45, imgsz=640,
                                              device="cpu", verbose=False)[0]
                picture = result.plot()
            draw_hands(picture, hand_result)
            cv2.putText(picture, "EDITH | Q / Esc to close", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(picture, f"Hands detected: {len(hand_result.hand_landmarks)}",
                        (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(picture, "Prototype: surgical tools not trained yet",
                        (12, picture.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
            cv2.imshow(WINDOW_NAME, picture)
            if first_frame:
                print("Live detection is running.", flush=True)
                first_frame = False

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"Detection stopped: {error}", flush=True)
        return 1
    finally:
        # Always free the camera, even if detection fails or the window is closed.
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        hand_detector.close()


if __name__ == "__main__":
    raise SystemExit(main())
