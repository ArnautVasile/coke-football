# Client Setup Guide

This guide is the client-facing setup and operation reference for the goal impact tracking system.

If you need the full technical documentation, model export notes, or training workflows, see [README.md](README.md).

## System Purpose

The system detects when a ball reaches the goal area and places a hit marker based on the configured scoring logic.

The recommended production workflow for this project is:

- printed goal-frame markers for goal pose
- ChArUco-based camera calibration
- YOLO detection
- `--impact-entry-point-mode deepest` so the marker is placed at the deepest point reached inside the goal

## Recommended Hardware

### Camera

- Recommended capture mode: `1920 x 1080` at `60 fps`
- Good low-light performance is strongly recommended
- A larger sensor is preferred because it reduces blur and improves detection reliability
- The camera should be mounted rigidly and should not move during normal operation

Reference camera used during testing:

- `Dell Pro Webcam WB5023`

### Computer

Reference computer used during testing:

- `MacBook Pro M4` with `24 GB RAM`

### Operating System

- The tested deployment path is macOS
- Windows is possible through the ONNX-based runtime path, but the runtime and acceleration stack must be adapted for the target machine

### Cabling

- The original webcam cable is relatively short
- If a longer cable run is required and the camera uses USB Type-A, use an active USB extension cable

## Printed Materials

### Goal Frame Markers

The black marker size is flexible, but in general bigger markers are more robust.

Recommended black marker sizes:

- `8 x 8 cm`: minimum recommended
- `10 x 10 cm`: recommended standard
- `12 x 12 cm`: preferred for longer distance or weaker light

Quiet zone:

- Add a white quiet zone of `1.5 cm` to `2.0 cm` on each side around the black square
- The white quiet zone is important and should not be removed

Important:

- If the printed black marker size changes, the software configuration must be updated to match the black square only
- The measurement that matters is the black square, not the white border

### ChArUco Board for Camera Calibration

A printed ChArUco board must be used to calibrate the camera before deployment.

Recommended print scale:

- board layout: `7 x 5` squares
- each square: `30 mm` (`3.0 cm`)
- each inner ArUco marker: `22 mm` (`2.2 cm`)
- active board area: `210 x 150 mm` (`21 x 15 cm`)

Calibration rule:

- During camera calibration, hide or remove the goal-frame markers from view
- Only the printed ChArUco board should be visible to the camera during camera calibration

## Files Used in This Project

Common files used in the tested setup:

- camera intrinsics: `data/calibration/camera_intrinsics_cam_front_1.json`
- goal marker layout: `data/calibration/goal_markers_layout_cola_zone_209x156.json`
- tested detector model: `runs/custom_ball/ball_yolo26s_identitymix_v2/weights/best.onnx`
- additional model to test: `runs/custom_ball/ball_yolo26s_identitymix_v2_fast/weights/best.onnx`

Printable assets:

- ChArUco board image: `data/calibration/charuco_board.png`
- ChArUco board spec: `data/calibration/charuco_board.json`
- goal marker sheet: `data/calibration/goal_markers_sheet.png`
- goal marker placement diagram: `data/calibration/goal_markers_layout.png`
- goal marker print reference pages:
  - `data/calibration/goal_markers_pages/goal_markers_page_1.png`
  - `data/calibration/goal_markers_pages/goal_markers_page_2.png`
  - `data/calibration/goal_markers_pages/goal_markers_page_3.png`

## Installation

Create the Python environment and install the requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-yolo.txt
```

Apple Silicon note:

- Use the official `onnxruntime` package
- Do not use `onnxruntime-silicon`

## Camera Calibration

Generate or use the provided ChArUco board:

```bash
python tools/generate_charuco_board.py \
  --output-image data/calibration/charuco_board.png \
  --output-spec data/calibration/charuco_board.json
```

Print the board at:

- `100%` scale
- no fit-to-page scaling

Run calibration:

```bash
python tools/calibrate_camera_charuco.py \
  --camera 0 \
  --backend avfoundation \
  --board-spec data/calibration/charuco_board.json \
  --output data/calibration/camera_intrinsics_cam_front_1.json
```

Calibration workflow:

1. Hide the goal markers from view.
2. Show only the printed ChArUco board to the camera.
3. Capture views from multiple positions, angles, and distances.
4. Press `Space` to accept good views.
5. Press `Enter` once enough diverse views have been collected.

## Goal Marker Installation

1. Print the goal markers at `100%` scale.
2. Mount them rigidly on the goal frame.
3. Make sure the correct marker layout JSON matches the real printed black-square size.
4. Make sure the markers are clearly visible to the camera during normal operation.

## Daily Operation

Recommended workflow:

1. Start the system.
2. Let the goal markers solve the goal pose.
3. Press `c` to freeze the current marker pose once it looks correct.
4. If the camera or goal moves, press `g` to refresh the pose once.
5. If you want continuous automatic pose updates again, press `a`.

Useful controls:

- `c`: freeze the current marker pose
- `g`: refresh marker pose once and keep it frozen
- `a`: resume automatic marker-pose updates
- `Shift+C`: manual 4-corner recalibration
- `q` or `Esc`: quit

## Tested Reference Command

This command worked well in testing and is the preferred reference mode for this project:

```bash
python run.py \
  --camera 0 \
  --backend avfoundation \
  --detector yolo \
  --yolo-model runs/custom_ball/ball_yolo26s_identitymix_v2/weights/best.onnx \
  --yolo-class-id 0 \
  --yolo-device mps \
  --yolo-imgsz 640 \
  --no-yolo-track \
  --detect-full-frame \
  --event-source candidate \
  --event-min-consecutive 1 \
  --track-min-consecutive 1 \
  --camera-calibration-file data/calibration/camera_intrinsics_cam_front_1.json \
  --goal-markers-layout data/calibration/goal_markers_layout_cola_zone_209x156.json \
  --goal-plane-depth-m 0.27 \
  --goal-pose-every 8 \
  --goal-pose-every-stable 24 \
  --goal-pose-stable-after 3 \
  --goal-pose-max-age 1.5 \
  --goal-presence-margin-px 140 \
  --ball-diameter-m 0.27 \
  --goal-plane-contact-tolerance-m 0.16 \
  --impact-entry-min-speed 25 \
  --impact-entry-confirm-frames 1 \
  --impact-entry-point-mode deepest \
  --impact-max-dt 0.22 \
  --impact-min-displacement 30 \
  --impact-entry-fallbacks \
  --impact-cooldown 2 \
  --impact-rearm-outside-ratio 1.0 \
  --impact-rearm-camera-margin-m 0.70 \
  --impact-rearm-miss-seconds 1.40 \
  --no-event-motion-after-reject \
  --process-every 1 \
  --async-capture \
  --ball-overlay-ttl 0 \
  --candidate-overlay-ttl 0 \
  --no-trajectory-bridge \
  --perf-breakdown \
  --stats-every 90
```

Additional model to test:

- `runs/custom_ball/ball_yolo26s_identitymix_v2_fast/weights/best.onnx`
- This is a recommended alternative model to test during deployment comparison
- If both models are available, compare them on the real installation and keep the one with the better balance of reliability and speed

## Most Important Parameters

- `--impact-entry-point-mode deepest`
  - preferred mode for this project
  - places the marker where the ball reached maximum depth inside the goal

- `--impact-cooldown 2`
  - blocks another hit for `2` seconds after a registered goal
  - increase this value if repeated or double-hit events appear after a real goal

- `--process-every 1`
  - processes every frame
  - this should stay as low as possible for the best tracking reliability
  - `1` is preferred when the hardware can sustain it

## Operational Notes

- If the camera or goal moves, refresh the marker pose with `g`
- If the marker overlay drifts or becomes unreliable, refresh or recalibrate before continuing
- Good lighting and low motion blur have a major effect on reliability
- The camera should stay fixed during operation

## If Retraining Is Needed

If the system shows too many false positives in the real installation, retraining may be required.

Recommended order:

1. First test the two available detector models:
   - `runs/custom_ball/ball_yolo26s_identitymix_v2/weights/best.onnx`
   - `runs/custom_ball/ball_yolo26s_identitymix_v2_fast/weights/best.onnx`
2. If false positives are still high, retrain the detector on footage from the real installation:
   - use the same camera
   - use the real ball
   - capture real lighting and background conditions
3. Evaluate the retrained detector on unseen validation footage from the same installation
4. If false positives are still a problem after detector retraining, train and enable the exact-ball verifier as a second-stage filter

In short:

- first try the current tested models
- then retrain the detector if needed
- then add the verifier only if false positives still remain

## Recommended Client Handover

Before the system is used on site, confirm:

- camera is mounted rigidly
- goal markers are printed at the correct size
- marker layout JSON matches the real black marker size
- camera was calibrated with the ChArUco board
- the tested reference command runs successfully
- marker pose is solved correctly before live use
