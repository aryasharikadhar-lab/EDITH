"""EDITH v2: hands and optional external body pose; not for clinical use.

Run: .venv/bin/python edith_detect_v2.py
Keys: P = body tracking, S = silhouette, C = coordinate details, Q/Esc = close.
Optional startup flags: --camera 1, --pose, --segmentation.
Models must already be local. This script never downloads models or saves video.
"""

import argparse
from pathlib import Path
from time import monotonic, monotonic_ns

import cv2
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parent
HAND_MODEL_PATH = ROOT / "hand_landmarker.task"
POSE_MODEL_PATH = ROOT / "pose_landmarker_lite.task"
WINDOW_NAME = "EDITH v2 | Q / Esc to close"
GREEN = (80, 240, 100)
CYAN = (255, 220, 80)
WHITE = (235, 235, 235)
YELLOW = (70, 220, 255)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
HAND_COORDINATES = (
    (0, "Wrist"), (4, "Thumb tip"), (8, "Index tip"),
    (12, "Middle tip"), (16, "Ring tip"), (20, "Little tip"),
)
# These are approximate external locations, not detected bones or anatomy.
BODY_JOINTS = (
    (11, "Left shoulder"), (12, "Right shoulder"),
    (13, "Left elbow"), (14, "Right elbow"),
    (15, "Left wrist"), (16, "Right wrist"),
    (23, "Left hip"), (24, "Right hip"),
    (25, "Left knee"), (26, "Right knee"),
    (27, "Left ankle"), (28, "Right ankle"),
)
BODY_CONNECTIONS = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32),
)


def create_hand_detector():
    """Reuse the existing hand model without importing the original script."""
    if not HAND_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Hand model missing: {HAND_MODEL_PATH}. No download attempted.")
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def create_pose_detector(segmentation=False):
    """The same local pose model supplies joints and optional silhouette masks."""
    if not POSE_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Pose model missing: {POSE_MODEL_PATH}. No download attempted.")
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=segmentation,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


class BodyTracker:
    """Keep optional body work off until requested; failures leave hands working."""

    def __init__(self):
        self.detector = None
        self.segmentation = False
        self.message = "Press P for body tracking."

    def configure(self, enabled, segmentation=False):
        self.close()
        if not enabled:
            self.message = "Body tracking off."
            return
        try:
            self.detector = create_pose_detector(segmentation)
            self.segmentation = segmentation
            self.message = "Body tracking ready."
        except Exception as error:
            self.message = "Body unavailable; hands still active."
            print(f"{self.message} {error}", flush=True)

    def detect(self, image, timestamp):
        if self.detector is None:
            return None
        try:
            return self.detector.detect_for_video(image, timestamp)
        except Exception as error:
            self.close()
            self.message = "Body stopped; hands still active."
            print(f"{self.message} {error}", flush=True)
            return None

    def close(self):
        detector, self.detector = self.detector, None
        self.segmentation = False
        if detector is not None:
            try:
                detector.close()
            except Exception as error:
                print(f"Could not fully close body detector: {error}", flush=True)


def medical_object_predictions(_frame):
    """Disabled integration point: never infer medical labels from everyday objects.

    Future custom-model connection goes HERE, after training and evaluation on
    labeled pictures of forceps, surgical scissors, clamps, replica scalpels,
    stethoscopes, and needle-free training syringes in the intended demo setup.
    Explicitly load the approved LOCAL model, validate its class names, and
    return only evaluated detections with boxes and confidence scores. Include
    pictures without props and confusing objects when evaluating false labels.
    Until then this returns no predictions. Neither YOLO nor any general
    everyday-object detector is loaded. Internal anatomy is outside this demo.
    """
    return ()


def pixel_point(landmark, width, height, pose=False):
    """Reject hidden/off-screen points rather than pinning them to frame edges."""
    if not np.isfinite(landmark.x) or not np.isfinite(landmark.y):
        return None
    if not (0 <= landmark.x <= 1 and 0 <= landmark.y <= 1):
        return None
    if pose:
        # These scores estimate whether a point is present and visible.
        for field in ("visibility", "presence"):
            score = getattr(landmark, field, None)
            if score is None or not np.isfinite(score) or score < 0.5:
                return None
    return (min(width - 1, int(landmark.x * width)),
            min(height - 1, int(landmark.y * height)))


def text(picture, message, position, color=WHITE, scale=0.45):
    cv2.putText(picture, message, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def draw_connections(picture, points, connections, color):
    for start, end in connections:
        if points[start] is not None and points[end] is not None:
            cv2.line(picture, points[start], points[end], color, 2, cv2.LINE_AA)


def tint_person(picture, result):
    """Mirror the raw-camera silhouette to match the displayed camera view."""
    if result is None or not result.segmentation_masks:
        return
    mask = result.segmentation_masks[0].numpy_view()
    mask = cv2.resize(mask, (picture.shape[1], picture.shape[0]))
    mask = cv2.flip(mask, 1)
    alpha = np.clip(mask, 0, 1)[..., None] * 0.28
    tint = np.array((160, 210, 60), dtype=np.float32)
    picture[:] = (picture * (1 - alpha) + tint * alpha).astype(np.uint8)


def draw_pose(picture, result):
    """Project raw-camera pose onto the mirror view, retaining person's left/right."""
    if result is None or not result.pose_landmarks:
        return {}
    height, width = picture.shape[:2]
    points = [pixel_point(dot, width, height, pose=True)
              for dot in result.pose_landmarks[0]]
    # Pose runs on the original image so left/right refer to the person.
    # Reflect pixel positions for display; keep landmark names unchanged.
    points = [None if point is None else (width - 1 - point[0], point[1])
              for point in points]
    draw_connections(picture, points, BODY_CONNECTIONS, CYAN)
    for point in points:
        if point is not None:
            cv2.circle(picture, point, 3, CYAN, -1, cv2.LINE_AA)
    # Labels identify only the external joints used in the coordinate panel.
    for index, name in BODY_JOINTS:
        point = points[index]
        if point is not None:
            short_name = name.replace("Left ", "L ").replace("Right ", "R ")
            label_width = cv2.getTextSize(short_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
            position = (max(0, min(width - label_width, point[0] + 5)),
                        max(14, point[1] - 5))
            text(picture, short_name, position, CYAN, 0.4)
    return {name: points[index] for index, name in BODY_JOINTS}


def draw_hands(picture, result):
    height, width = picture.shape[:2]
    summaries = []
    for number, hand in enumerate(result.hand_landmarks, 1):
        points = [pixel_point(dot, width, height) for dot in hand]
        visible = [point for point in points if point is not None]
        if not visible:
            continue
        x1 = max(0, min(x for x, _ in visible) - 15)
        y1 = max(0, min(y for _, y in visible) - 15)
        x2 = min(width - 1, max(x for x, _ in visible) + 15)
        y2 = min(height - 1, max(y for _, y in visible) + 15)
        cv2.rectangle(picture, (x1, y1), (x2, y2), GREEN, 2)
        # Numbers match the sidebar in this frame, not permanent person IDs.
        text(picture, f"Hand {number}", (x1, max(16, y1 - 6)), GREEN, 0.55)
        draw_connections(picture, points, HAND_CONNECTIONS, YELLOW)
        for point in visible:
            cv2.circle(picture, point, 4, (255, 100, 0), -1, cv2.LINE_AA)
        summaries.append((number, (x1, y1, x2, y2), points))
    return summaries


def coordinate_string(point):
    return str(point) if point is not None else "not visible"


def compose_display(picture, hands, joints, body, details, fps):
    """Keep coordinates beside the original-sized video so labels stay readable."""
    height, width = picture.shape[:2]
    display = np.full((max(height + 130, 640), width + 360, 3), 22, np.uint8)
    display[:height, :width] = picture
    x = width + 12
    text(display, "EDITH v2 | Live pixel coordinates", (x, 24), GREEN)
    text(display, f"FPS: {fps:.1f} | Hands: {len(hands)}", (x, 46))
    enabled = body.detector is not None
    text(display, f"Body: {'ON' if enabled else 'OFF'} | Silhouette: "
         f"{'ON' if body.segmentation else 'OFF'}", (x, 66))
    text(display, "Origin: video top-left; x right, y down", (x, 86))
    y = 110
    for number, box, points in hands:
        text(display, f"Hand {number} (number may change)", (x, y), GREEN)
        y += 17
        text(display, f"Box: {box[:2]} - {box[2:]}", (x, y))
        y += 17
        if details:
            for index, name in HAND_COORDINATES:
                text(display, f"{name}: {coordinate_string(points[index])}", (x, y))
                y += 17
        y += 6
    y = max(y + 12, 402)
    text(display, "Body: person's left/right (approx.)", (x, y), CYAN)
    y += 20
    if enabled and details:
        for _, name in BODY_JOINTS:
            text(display, f"{name}: {coordinate_string(joints.get(name))}", (x, y))
            y += 17
    elif not enabled:
        text(display, body.message, (x, y), YELLOW, 0.4)
    else:
        text(display, "Coordinate details hidden (press C).", (x, y))

    text(display, "Q / Esc: close | P: body | S: silhouette | C: details",
         (12, height + 24), GREEN)
    text(display, "Demonstration prototype - not for clinical use.",
         (12, height + 48), YELLOW)
    text(display, "Medical-object recognition: DISABLED (no trained model).",
         (12, height + 70), YELLOW)
    text(display, "Pose lines are estimates; no internal anatomy is detected.",
         (12, height + 92))
    text(display, "Silhouette needs body tracking. Move back for full-body view.",
         (12, height + 114), scale=0.42)
    return display


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera number (default: 0)")
    parser.add_argument("--pose", action="store_true", help="Start with body tracking on")
    parser.add_argument("--segmentation", action="store_true",
                        help="Start with body tracking and silhouette highlighting on")
    args = parser.parse_args()
    camera = None
    hand_detector = None
    body = BodyTracker()
    try:
        print("Loading EDITH v2 hands from the existing local model...", flush=True)
        hand_detector = create_hand_detector()
        if args.pose or args.segmentation:
            body.configure(True, args.segmentation)

        camera = cv2.VideoCapture(args.camera)
        if not camera.isOpened():
            print("Camera unavailable. Close other camera apps and check System Settings > "
                  "Privacy & Security > Camera.", flush=True)
            return 1
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("EDITH v2 online. Q/Esc: close; P: body; S: silhouette; C: coordinates.", flush=True)
        print("Video stays on this computer and is not saved. Medical recognition is disabled.",
              flush=True)
        details = True
        last_timestamp = -1
        previous_time = monotonic()
        fps = 0.0
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read the camera. Check its connection.", flush=True)
                return 1
            # Keep the original camera image for body pose. Feeding the mirrored
            # view to pose can reverse the person's left/right assignments.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.flip(rgb_frame, 1))
            timestamp = max(last_timestamp + 1, monotonic_ns() // 1_000_000)
            last_timestamp = timestamp
            hand_result = hand_detector.detect_for_video(image, timestamp)
            # Off means no pose inference or mask generation, not just hidden lines.
            pose_result = None
            if body.detector is not None:
                pose_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                pose_result = body.detect(pose_image, timestamp)
            picture = frame.copy()
            if body.segmentation:
                tint_person(picture, pose_result)
            joints = draw_pose(picture, pose_result)
            hands = draw_hands(picture, hand_result)
            # Future medical detections would be drawn here after integration.
            # Currently this deliberately returns nothing, with no model loading.
            medical_object_predictions(frame)

            now = monotonic()
            instantaneous_fps = 1.0 / max(now - previous_time, 1e-6)
            fps = instantaneous_fps if fps == 0 else 0.9 * fps + 0.1 * instantaneous_fps
            previous_time = now
            cv2.imshow(WINDOW_NAME,
                       compose_display(picture, hands, joints, body, details, fps))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("p"), ord("P")):
                body.configure(body.detector is None)
            elif key in (ord("s"), ord("S")):
                if body.detector is None:
                    body.message = "Press P before enabling silhouette."
                else:
                    # Rebuild only when toggled: OFF stops producing masks.
                    body.configure(True, not body.segmentation)
            elif key in (ord("c"), ord("C")):
                details = not details
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"EDITH v2 stopped: {error}", flush=True)
        return 1
    finally:
        if camera is not None:
            camera.release()
        body.close()
        try:
            if hand_detector is not None:
                hand_detector.close()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
