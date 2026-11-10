# EDITH

EDITH is a Conrad Challenge project. The final goal is to recognize hands and surgical tools through a camera.

The current prototype detects up to two hands. It draws a green box around each hand and dots and lines along the fingers. It has not been trained to recognize specific surgical tools yet.

## Start the detection camera

From this project's terminal, run:

```sh
.venv/bin/python edith_detect.py
```

The first run downloads a ready-trained hand model. Later runs use the saved model. The live camera pictures are processed on your computer and are not saved by the script.

Allow camera access if macOS asks. Hold your hand where the camera can see it. Click the camera window and press **Q** or **Escape** to close it.

If your camera is unavailable, close other apps using it and check **System Settings > Privacy & Security > Camera**. For another connected camera, try:

```sh
.venv/bin/python edith_detect.py --camera 1
```

To also show an everyday-object demo:

```sh
.venv/bin/python edith_detect.py --objects
```

An object name is the model's prediction and can be wrong. The everyday-object model is not a surgical-tool detector.

## How it works

The camera sends a series of pictures to a model—a program that has learned from example pictures. The hand model estimates points on the wrist and fingers. EDITH uses those points to draw the outlines you see. Testing with your intended setup, including gloves and tools covering parts of the hands, is still needed.

The code uses [Google's MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python) and an optional [Ultralytics YOLO11 object detector](https://docs.ultralytics.com/models/yolo11/).
