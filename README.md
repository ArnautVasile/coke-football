# Goal Impact Tracker (Python)

This project captures a camera feed (recommended `1080p @ 60fps`), supports either manual corner calibration or printed goal-frame markers, and estimates when and where the ball reaches the calibrated goal plane.

It also includes camera-shift adaptation so small camera movement can still work, with manual recalibration when needed.

## Features

- 4-corner goal calibration by mouse clicks
- Marker-based goal pose from printed goal-frame ArUco markers
- ChArUco camera calibration for camera intrinsics and 3D plane checks
- Saved calibration + reference frame
- Auto adaptation for slight camera movement (ORB feature matching + homography)
- Two detector modes:
  - `motion` (fast CPU baseline)
  - `yolo` (YOLO26 via Ultralytics, more robust but heavier)
- Hit detection modes:
  - `entry`: validated ball crosses from outside to inside goal polygon (good for slow shots)
  - `impact`: speed-drop / bounce style event inside polygon
- Live impact visualization:
  - blue ring around current ball candidate
  - red flash + pulse on impact frame
  - "Goal Hit Map" panel with recent hit dots
  - last-hit zone + coordinates (normalized + meters)
- Impact coordinates:
  - normalized (`x`, `y`) in range `[0..1]`
  - metric (`meters`) using configurable goal dimensions
- CSV log output for all detected impacts

## Client Deployment Summary

Before deployment, please read [CLIENT_SETUP.md](CLIENT_SETUP.md).

Recommended camera:

- `1920 x 1080` at `60 fps`
- good low-light performance
- larger sensor preferred, because cleaner frames and less motion blur improve detection reliability
- rigid mounting is strongly recommended

Reference hardware used during testing:

- camera: `Dell Pro Webcam WB5023`
- computer: `MacBook Pro M4` with `24 GB RAM`

Operating system:

- the current tested path is macOS
- Windows is possible through the ONNX-based runtime path, but the acceleration/runtime stack must be adapted for the target machine

Cabling:

- the original webcam cable is relatively short
- if a longer run is required and the camera uses USB Type-A, use an active USB extension cable

Goal markers:

- recommended black marker sizes:
  - `8 x 8 cm`: minimum recommended
  - `10 x 10 cm`: recommended standard
  - `12 x 12 cm`: preferred for longer distance or weaker light
- add a white quiet zone of `1.5 cm` to `2.0 cm` on each side around the black square
- if the printed black marker size changes, the software configuration must be updated to match the black square only

Camera calibration:

- a printed ChArUco board must be used to calibrate the camera before deployment
- ChArUco print scale:
  - board layout: `7 x 5` squares
  - each square: `30 mm` (`3.0 cm`)
  - each inner ArUco marker: `22 mm` (`2.2 cm`)
  - active board area: `210 x 150 mm` (`21 x 15 cm`)
- during camera calibration, hide or remove the goal-frame markers from view
- only the printed ChArUco board should be visible during camera calibration

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` now uses `opencv-contrib-python` because the ChArUco/ArUco calibration workflow depends on `cv2.aruco`.

Optional YOLO setup:

```bash
pip install -r requirements-yolo.txt
```

Optional export/benchmark tooling:

```bash
pip install -r requirements-export.txt
```

On Apple Silicon, use the current official `onnxruntime` package.
Do not use `onnxruntime-silicon` for this project.
On macOS, `coremltools` is only installed automatically for Python `< 3.14`.
On Windows, prefer `--backend dshow` when probing webcam frame-rate modes because some systems fail with OpenCV's default `MSMF` backend.

## Run

```bash
python run.py --camera 0 --width 1920 --height 1080 --fps 60
```

## Export & Benchmark

Check what acceleration paths are available in the current environment:

```bash
python tools/export_benchmark.py --check
```

Export and benchmark ONNX + CoreML:

```bash
python tools/export_benchmark.py \
  --model yolo26s.pt \
  --formats onnx,coreml \
  --imgsz 640
```

Mac-focused benchmark command:

```bash
python tools/export_benchmark.py \
  --model yolo26s.pt \
  --formats onnx,coreml \
  --imgsz 640 \
  --device cpu
```

If you are on Python `3.14`, the script will skip `CoreML` and benchmark `ONNX` only.
For CoreML export on macOS, use a Python `3.13` environment.

Results are written to `data/benchmarks/*.json` and `data/benchmarks/*.csv`.

Notes:

- On Apple Silicon, `CoreML` is usually the best acceleration target when PyTorch `mps` is unavailable.
- In Python `3.14` macOS environments, expect `ONNX` to work first and `CoreML` to require a Python `3.13` venv.
- If `onnx`, `onnxruntime`, or `coremltools` are missing, the script records the error per format instead of failing the whole run.
- On modern macOS ARM setups, install `onnxruntime`, not `onnxruntime-silicon`.
- Project default is now `yolo26s.pt`.
- If you need a lighter debugging baseline or a faster export benchmark, fall back to `yolo26n.pt`.
- If an exported ONNX model is unstable at runtime, try `--opset 17 --no-simplify` during export.

## Native Apple Helper

A native macOS helper is included at `apple/BallVisionHelper`. It uses:

- `Core ML` for detection
- `Vision` object tracking between detection frames
- `AVFoundation` camera capture
- JSON lines on stdout for detections

Build it:

```bash
cd apple/BallVisionHelper
swift build -c release
```

List cameras:

```bash
./.build/release/BallVisionHelper --list-cameras
```

Run it with a Core ML ball model:

```bash
./.build/release/BallVisionHelper \
  --model /absolute/path/to/ball.mlmodelc \
  --label ball \
  --camera 0 \
  --width 1920 \
  --height 1080 \
  --fps 60 \
  --detect-every 3 \
  --confidence 0.20
```

Export a custom YOLO26 ball model to Core ML with the working Python `3.12` setup:

```bash
python3.12 -m venv .venv-coreml312
source .venv-coreml312/bin/activate
pip install ultralytics coremltools onnx numpy<=2.3.5 torch==2.7.0 torchvision==0.22.0

python - <<'PY'
from ultralytics import YOLO
model = YOLO('/absolute/path/to/best.pt')
path = model.export(format='coreml', imgsz=640)
print(path)
PY
```

Optional: compile the exported `.mlpackage` to `.mlmodelc`:

```bash
xcrun coremlcompiler compile /absolute/path/to/best.mlpackage /absolute/path/to/output_dir
```

Output format:

- one JSON object per detected/tracked ball
- fields include `timestamp`, `frameIndex`, `source`, `confidence`, `x`, `y`, `width`, `height`, `radius`
- coordinates are normalized to `[0..1]`

Notes:

- this helper is a standalone macOS helper binary
- it can now be used from `run.py` with `--detector vision`
- to use this path, export your custom detector to `Core ML` from a Python `3.12`/`3.13` Core ML-capable toolchain
- the bundled example model at `data/coreml/best.mlmodelc` is an older `YOLO26n` example, so replace it with a fresh export of your production detector before deployment

Run the main tracker with the Apple helper as detector:

```bash
python run.py \
  --camera 0 \
  --backend avfoundation \
  --detector vision \
  --vision-model data/coreml/best.mlmodelc \
  --vision-helper-bin apple/BallVisionHelper/.build/release/BallVisionHelper \
  --vision-camera 0 \
  --vision-detect-every 3 \
  --vision-confidence 0.20 \
  --vision-local-search-scale 2.6 \
  --vision-full-recover-every 4 \
  --vision-compute-units all \
  --vision-identity-source data/identity/ball_identity.onnx \
  --vision-max-age 0.08 \
  --no-detect-full-frame \
  --detect-roi-margin 180 \
  --global-search-every 2 \
  --event-motion-fallback \
  --trajectory-bridge \
  --trajectory-max-gap-frames 3 \
  --event-source candidate \
  --event-min-consecutive 1 \
  --process-every 1
```

Caveat:

- in `--detector vision` mode, the helper now owns the camera and streams preview frames + detections into Python
- this keeps the Python goal UI, calibration, and hit logic while avoiding dual camera ownership on macOS

To reject clutter that does not look like your exact ball, point the vision detector at a learned exact-ball verifier model:

```bash
--vision-identity-source /absolute/path/to/data/identity/ball_identity.onnx
```

This adds a second-stage learned exact-ball check on the detector crop before the blue marker is accepted.

## Train Your Ball

If you want the detector to focus on your exact football and camera setup, collect a 1-class custom dataset and train a `ball` model.

Fastest workflow: capture and label in one step from the live camera.

```bash
python tools/capture_label_ball_live.py \
  --camera 0 \
  --backend avfoundation \
  --width 1920 \
  --height 1080 \
  --fps 60 \
  --dataset-root datasets/custom_ball \
  --save-split train \
  --target-count 300
```

Live-label flow:

- live camera stays running
- press `space` to freeze the current frame
- drag a box around the ball
- press `enter` or `space` to confirm the box
- the tool saves the image and YOLO label, then returns to the live camera automatically
- it stops when it reaches the target count

Notes:

- safest workflow: capture one session with `--save-split train`, then a separate session with `--save-split val`
- by default, `--save-split train` is now the safer production default
- `--save-split auto` still exists only as a legacy convenience mode where every 5th image goes into `val`
- `--val-every` is convenient, but it mixes near-neighbor frames across `train` and `val`, so validation can look better than real-life performance
- labels are written directly in YOLO format, so no separate annotation pass is needed

Capture frames:

```bash
python tools/capture_ball_dataset.py \
  --camera 0 \
  --backend avfoundation \
  --width 1920 \
  --height 1080 \
  --fps 60 \
  --dataset-root datasets/custom_ball
```

Capture keys:

- `space`: save current frame
- `r`: toggle burst capture
- `1`: save to `train`
- `2`: save to `val`
- `q`: quit

Annotate the ball:

```bash
python tools/annotate_ball_dataset.py --dataset-root datasets/custom_ball --split all
```

Annotation keys:

- `enter` or `space`: draw/edit the ball box
- `n`: mark frame as no ball
- `s`: skip
- `b`: go back
- `q`: quit

## Fair Unseen Metrics (Holdout Test Split)

To get honest metrics, evaluate on frames the model never saw in training.

1. Extract a `test` split from an unseen video:

```bash
python tools/create_unseen_test_split.py \
  --video data/identity/raw_video/ball_identity_000008.mp4 \
  --dataset-root datasets/custom_ball_holdout \
  --split test \
  --target-count 180
```

2. Annotate the extracted test frames:

```bash
python tools/annotate_ball_dataset.py \
  --dataset-root datasets/custom_ball_holdout \
  --split test
```

3. Evaluate your full runtime system (detector + optional verifier):

```bash
python tools/evaluate_ball_system.py \
  --dataset-root datasets/custom_ball_holdout \
  --split test \
  --model runs/custom_ball/ball_yolo26s4/weights/best.pt \
  --class-id 0 \
  --conf 0.10 \
  --imgsz 960 \
  --identity-source data/identity/ball_identity_goal_v2.onnx \
  --iou-threshold 0.50 \
  --output-json data/metrics/holdout_system_eval.json \
  --output-csv data/metrics/holdout_system_eval.csv
```

This reports fair holdout metrics such as `precision`, `recall`, `f1`, false alarms on no-ball frames, and localization quality (`mean_iou_tp`, center error).

To compare two detectors fairly, run the same command twice with different `--model` values against the same holdout labels.

Train a 1-class model:

```bash
python tools/train_ball_detector.py \
  --dataset-root datasets/custom_ball \
  --model yolo26s.pt \
  --imgsz 640 \
  --epochs 80 \
  --batch 16 \
  --export-onnx
```

Train for accuracy at `960`, but export a faster runtime ONNX at `640`:

```bash
python tools/train_ball_detector.py \
  --dataset-root datasets/custom_ball \
  --model yolo26s.pt \
  --imgsz 960 \
  --epochs 80 \
  --batch 8 \
  --export-onnx \
  --export-imgsz 640 \
  --export-opset 17 \
  --export-no-simplify
```

Alternative starter model for Mac testing:

```bash
python tools/train_ball_detector.py \
  --dataset-root datasets/custom_ball \
  --model yolo11n.pt \
  --imgsz 640 \
  --epochs 80 \
  --batch 16 \
  --export-onnx
```

Safer ONNX export when a model crashes at runtime:

```bash
python tools/train_ball_detector.py \
  --dataset-root datasets/custom_ball \
  --model yolo26s.pt \
  --imgsz 640 \
  --epochs 80 \
  --batch 16 \
  --export-onnx \
  --export-opset 17 \
  --export-no-simplify
```

After training, run the tracker with your custom model and set the class id to `0`:

```bash
python run.py \
  --camera 0 \
  --detector yolo \
  --yolo-model runs/custom_ball/ball_yolo26s/weights/best.pt \
  --yolo-class-id 0
```

Tip: prefer the exact absolute `--yolo-model` path printed by `tools/train_ball_detector.py` after training, because that is the real saved location for your run.

## Train Exact-Ball Verifier

If detector false positives are the main pain point, keep the detector and add a second-stage verifier that learns the identity of your exact ball.

1. Record a clean moving-ball video from the same camera:

```bash
python tools/record_ball_identity_video.py \
  --camera 0 \
  --backend avfoundation \
  --width 1920 \
  --height 1080 \
  --fps 60 \
  --duration-seconds 60 \
  --countdown-seconds 3
```

2. Capture verifier crops from that video:

```bash
python tools/capture_ball_identity_video.py \
  --source /absolute/path/to/ball_video.mp4 \
  --output-root datasets/ball_identity \
  --sample-every 8 \
  --max-samples 220
```

Suggested workflow:

- film the real ball for about a minute
- rotate it, move it closer/farther, include a little blur and partial hand occlusion
- on each paused sample frame, press `space`, draw the ball box, and save the crop
- press `s` to skip weak/unhelpful frames

3. Train the verifier model.

Recommended:

```bash
python tools/train_ball_identity.py \
  --source datasets/ball_identity \
  --output data/identity/ball_identity.onnx \
  --format onnx \
  --device auto
```

Fallback classic verifier:

```bash
python tools/train_ball_identity.py \
  --source datasets/ball_identity \
  --output data/identity/ball_identity.npz \
  --format classic
```

You can also train directly from the labeled detector dataset:

```bash
python tools/train_ball_identity.py \
  --source datasets/custom_ball/prepared \
  --output data/identity/ball_identity.onnx \
  --format onnx \
  --device auto
```

4. Use the trained verifier in the tracker:

```bash
python run.py \
  --camera 0 \
  --backend avfoundation \
  --detector vision \
  --vision-model /absolute/path/to/best.mlmodelc \
  --vision-helper-bin /absolute/path/to/BallVisionHelper \
  --vision-camera 0 \
  --vision-confidence 0.08 \
  --vision-identity-source /absolute/path/to/data/identity/ball_identity.onnx \
  --vision-local-search-scale 2.6 \
  --vision-full-recover-every 4 \
  --detect-full-frame
```

Notes:

- the helper now does local detector search around the tracked ball between periodic global recoveries
- the Python side only trusts helper track updates after a verified ball hit started the sequence
- the ONNX verifier trains on Mac with `mps` when available, exports one cross-platform model, and runs on macOS or Windows through `onnxruntime`
- old `--vision-appearance-dataset` and `--vision-appearance-threshold` flags are still accepted as aliases for compatibility

Recommended one-camera reliability flags:

```bash
python run.py \
  --camera 0 \
  --backend avfoundation \
  --detector yolo \
  --yolo-model runs/custom_ball/ball_yolo26s/weights/best.pt \
  --yolo-class-id 0 \
  --no-yolo-track \
  --no-detect-full-frame \
  --detect-roi-margin 180 \
  --global-search-every 2 \
  --event-motion-fallback \
  --trajectory-bridge \
  --trajectory-max-gap-frames 3 \
  --yolo-imgsz 640 \
  --event-source candidate \
  --event-min-consecutive 1 \
  --process-every 1
```

Optional arguments:

```bash
python run.py \
  --camera 0 \
  --probe-camera \
  --probe-seconds 2.5
```

Then run with the best mode reported by probe, for example:

```bash
python run.py \
  --camera 0 \
  --width 1920 \
  --height 1080 \
  --fps 60 \
  --backend avfoundation \
  --fourcc MJPG \
  --force-resize-input \
  --detect-full-frame \
  --detector motion \
  --motion-scale 0.6 \
  --motion-warmup-frames 45 \
  --motion-static-fallback \
  --motion-static-every 3 \
  --motion-static-min-radius 7 \
  --motion-static-max-radius 90 \
  --motion-static-param2 20 \
  --track-min-consecutive 3 \
  --track-max-step 260 \
  --track-max-misses 2 \
  --track-radius-ratio 1.7 \
  --process-every 1 \
  --stats-every 120 \
  --display-scale 1.0 \
  --goal-width-m 7.32 \
  --goal-height-m 2.44 \
  --adapt-every 8 \
  --adapt-scale 0.65 \
  --adapt-min-confidence 0.5 \
  --impact-enable-entry \
  --impact-entry-min-speed 25 \
  --impact-min-displacement 90 \
  --impact-min-speed 350 \
  --impact-arm-seconds 1.2 \
  --hit-overlay-ttl 0.9 \
  --min-area 120 \
  --max-area 6000 \
  --min-circularity 0.42
```

YOLO26 mode:

```bash
python run.py \
  --camera 0 \
  --width 1920 \
  --height 1080 \
  --fps 60 \
  --detector yolo \
  --no-detect-full-frame \
  --detect-roi-margin 140 \
  --yolo-track \
  --yolo-tracker configs/bytetrack_ball.yaml \
  --yolo-model yolo26s.pt \
  --yolo-device mps \
  --yolo-conf 0.15 \
  --yolo-imgsz 640 \
  --event-motion-fallback \
  --process-every 1 \
  --event-source auto \
  --event-min-consecutive 1
```

Hybrid mode (recommended for high-speed shots with blur):

```bash
python run.py \
  --camera 0 \
  --width 1920 \
  --height 1080 \
  --fps 60 \
  --detector hybrid \
  --no-detect-full-frame \
  --detect-roi-margin 180 \
  --global-search-every 1 \
  --yolo-model yolo26s.pt \
  --yolo-device mps \
  --yolo-conf 0.10 \
  --yolo-imgsz 960 \
  --yolo-track \
  --yolo-tracker configs/bytetrack_ball.yaml \
  --motion-scale 0.6 \
  --event-motion-fallback \
  --event-source auto \
  --event-min-consecutive 1 \
  --process-every 1 \
  --candidate-overlay-ttl 0.25
```

High-reliability football profile (single camera):

```bash
python run.py \
  --camera 0 \
  --width 1920 \
  --height 1080 \
  --fps 60 \
  --detector yolo \
  --no-detect-full-frame \
  --detect-roi-margin 180 \
  --global-search-every 2 \
  --yolo-model yolo26s.pt \
  --yolo-device mps \
  --yolo-conf 0.10 \
  --yolo-imgsz 960 \
  --yolo-track \
  --yolo-tracker configs/bytetrack_ball.yaml \
  --event-motion-fallback \
  --event-source candidate \
  --event-min-consecutive 1 \
  --process-every 1 \
  --candidate-overlay-ttl 0.25
```

If the ball starts outside the white ROI box and enters fast, increase fallback search:

```bash
--global-search-every 1
```

(`1` = full-frame search every detector frame, highest reliability but lower FPS)

## Controls

- `c`: freeze the current marker pose when marker pose is active; otherwise recalibrate manual goal corners
- `g`: refresh marker pose once and keep it frozen (`goal markers` mode only)
- `a`: resume automatic marker-pose updates (`goal markers` mode only)
- `Shift+C`: force manual 4-corner recalibration even when marker pose mode is active
- `b`: manually select/lock ball ROI for tracker (only when `--ball-tracker` is not `none`)
- `x`: unlock/reset ball tracker
- `q` or `Esc`: quit

## Calibration & Recalibration

1. If you are using manual goal calibration, click the 4 goal corners and press `Enter`.
2. If you are using marker-based goal pose, mount the printed goal markers and let the app solve the goal automatically; manual 4-corner clicks are not required in this mode.
3. Small camera movement is auto-adjusted in manual/adaptation mode; marker mode can refresh the goal pose from the visible markers.
4. If overlays drift:
   - use `c` to freeze the current marker pose
   - use `g` to refresh marker pose once while staying frozen
   - use `a` to resume automatic marker updates
   - use `Shift+C` for a manual 4-corner recalibration

### ChArUco Camera Calibration

Before marker-based goal pose or 3D plane checks, calibrate the camera/lens once with a printed ChArUco board.

Generate a printable board:

```bash
python tools/generate_charuco_board.py \
  --output-image data/calibration/charuco_board.png \
  --output-spec data/calibration/charuco_board.json
```

Print the PNG at `100%` scale with no fit-to-page scaling.

Recommended ChArUco print scale:

- board layout: `7 x 5` squares
- each square: `30 mm` (`3.0 cm`)
- each inner ArUco marker: `22 mm` (`2.2 cm`)
- active board area: `210 x 150 mm` (`21 x 15 cm`)

Calibration note:

- during camera calibration, hide or remove the goal-frame markers from view
- only the printed ChArUco board should be visible to the camera during this step

Then capture ChArUco views from the live camera:

```bash
python tools/calibrate_camera_charuco.py \
  --camera 0 \
  --backend avfoundation \
  --board-spec data/calibration/charuco_board.json \
  --output data/calibration/camera_intrinsics.json
```

Workflow:

1. Show the printed board to the camera from many positions and angles.
2. Press `Space` to accept a good view.
3. Press `Enter` once you have enough diverse samples.

Good captures should cover:

- center, left, right, top, and bottom of the image
- near and farther distances
- different tilts / rotations

The saved intrinsics file is the foundation for later 3D goal-plane checks.

For the OpenCV camera path (`motion`, `yolo`, `hybrid`), you can already load it with:

```bash
python run.py \
  --camera 0 \
  --camera-calibration-file data/calibration/camera_intrinsics.json \
  --undistort-input
```

Note:

- in native `--detector vision` mode, the helper currently owns the camera feed, so `--undistort-input` is not applied there yet
- ChArUco calibrates the camera/lens, not the goal frame itself
- live goal-pose updates are handled by the printed goal-frame markers described below

### Goal Frame Markers

Generate printable goal-frame markers and their placement files:

```bash
python tools/generate_goal_markers.py
```

This creates:

- `data/calibration/goal_markers_sheet.png`: printable A4 marker sheet
- `data/calibration/goal_markers_layout.png`: placement diagram for the frame
- `data/calibration/goal_markers_layout.json`: marker IDs and real-world positions
- `data/calibration/goal_markers/`: individual marker PNGs
- `data/calibration/goal_markers_pages/`: printable per-page marker references

Default layout:

- dictionary: `DICT_6X6_50`
- marker size: `80 mm`
- IDs:
  - `10`: top-left
  - `11`: top-right
  - `12`: mid-left
  - `13`: mid-right
  - `14`: bottom-left
  - `15`: bottom-right

The generated default is `80 mm`, but for production installs a larger black marker size is usually more robust.

Recommended goal-marker print sizes:

- the black marker area can be larger than the default; in general, bigger markers are more robust
- minimum recommended black area: `8 x 8 cm`
- recommended standard black area: `10 x 10 cm`
- preferred black area for longer distance or weaker light: `12 x 12 cm`
- add a white quiet zone of `1.5 cm` to `2.0 cm` on each side around the black square
- the software marker size must match the black square only, not the white quiet zone

Print at `100%` scale and avoid `fill page`.

Preview the mounted markers and solve the live goal pose:

```bash
python tools/preview_goal_pose.py \
  --camera 0 \
  --backend avfoundation \
  --camera-calibration-file data/calibration/camera_intrinsics.json \
  --goal-markers-layout data/calibration/goal_markers_layout.json
```

When the pose is solved, the tool draws the reconstructed goal opening in green and reports visible marker IDs + reprojection error.

Run the main tracker with marker-based goal pose and 3D plane gating:

```bash
python run.py \
  --camera 0 \
  --backend avfoundation \
  --detector vision \
  --vision-model /absolute/path/to/best.mlmodelc \
  --vision-helper-bin /absolute/path/to/BallVisionHelper \
  --vision-camera 0 \
  --vision-detect-every 1 \
  --vision-confidence 0.08 \
  --vision-compute-units all \
  --vision-max-age 0.20 \
  --vision-max-area-ratio 0.30 \
  --vision-max-aspect-ratio 1.80 \
  --vision-min-area-ratio 0.002 \
  --detect-full-frame \
  --event-source trusted \
  --camera-calibration-file data/calibration/camera_intrinsics.json \
  --goal-markers-layout data/calibration/goal_markers_layout.json \
  --ball-diameter-m 0.22 \
  --goal-plane-contact-tolerance-m 0.10 \
  --goal-pose-alpha 0.45 \
  --goal-pose-max-age 0.35 \
  --process-every 1
```

What this adds on top of the normal 2D hit logic:

- the goal corners are updated from the live marker pose instead of only manual 4-click calibration
- the app estimates the ball's distance to the real goal plane using:
  - camera intrinsics
  - live marker pose
  - known ball diameter
- entry / impact events are rejected when the ball is still too far in front of the goal plane

Notes:

- `--goal-markers-layout` requires `--camera-calibration-file`
- when marker pose is active, `--undistort-input` is ignored because the pose solve already uses the calibrated camera model
- if marker visibility drops for a moment, the last valid pose is kept alive for `--goal-pose-max-age` seconds

### Tested Reference Command (Preferred YOLO + Marker Pose Mode)

The following command worked well in our testing and is the preferred reference mode for this project when using the YOLO detector with marker-based goal pose:

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

This command is intended as a tested reference, not a universal preset. Replace the model and calibration file paths if your deployment uses different files. For deployment-specific guidance, required printable assets, and the recommended model comparison workflow, see [CLIENT_SETUP.md](CLIENT_SETUP.md).

Parameter notes for this reference command:

- `--camera 0`: use camera index `0`
- `--backend avfoundation`: use the macOS AVFoundation camera backend
- `--detector yolo`: use the YOLO-based detector path
- `--yolo-model ...best.onnx`: load the tested ONNX detector model
- `--yolo-class-id 0`: use class `0` from the custom one-class ball model
- `--yolo-device mps`: request the Apple Silicon/CoreML-accelerated runtime path
- `--yolo-imgsz 640`: run the detector at `640 x 640`; this is a good balance of speed and reliability
- `--no-yolo-track`: disable YOLO ByteTrack mode; this reference profile relies on direct detections instead
- `--detect-full-frame`: search the full image instead of a reduced ROI
- `--event-source candidate`: allow the event logic to use candidate detections directly
- `--event-min-consecutive 1`: require only one consecutive candidate frame for event gating
- `--track-min-consecutive 1`: require only one trusted frame before a track is considered valid
- `--camera-calibration-file ...camera_intrinsics_cam_front_1.json`: load the tested camera intrinsics
- `--goal-markers-layout ...goal_markers_layout_cola_zone_209x156.json`: load the tested goal-marker geometry and placement
- `--goal-plane-depth-m 0.27`: move the scoring plane `27 cm` deeper into the goal than the front opening
- `--goal-pose-every 8`: re-solve marker pose every `8` frames while not yet stable
- `--goal-pose-every-stable 24`: once stable, re-solve marker pose every `24` frames
- `--goal-pose-stable-after 3`: switch to the stable cadence after `3` successful pose solves
- `--goal-pose-max-age 1.5`: keep the last valid goal pose alive for up to `1.5` seconds if markers disappear briefly
- `--goal-presence-margin-px 140`: allow the ball to be slightly outside the visible opening before rejecting it as unrelated
- `--ball-diameter-m 0.27`: use a real ball diameter of `27 cm` for 3D plane calculations
- `--goal-plane-contact-tolerance-m 0.16`: allow `16 cm` of extra surface slack around the plane for hit gating
- `--impact-entry-min-speed 25`: require at least `25 px/s` to arm entry-style goal events
- `--impact-entry-confirm-frames 1`: confirm entry after one accepted inside-goal frame
- `--impact-entry-point-mode deepest`: preferred mode for this project; place the hit marker at the deepest point reached inside the goal, not the first or last frame
- `--impact-max-dt 0.22`: reject event history samples that are too far apart in time
- `--impact-min-displacement 30`: require at least `30 px` of movement for impact-style events
- `--impact-entry-fallbacks`: allow permissive recovery logic when the exact crossing frame is missed
- `--impact-cooldown 2`: block a new hit for `2` seconds after a registered hit; if you see double-hit behavior after a real goal, increase this value
- `--impact-rearm-outside-ratio 1.0`: require the ball to move fully back outside before another hit can arm again
- `--impact-rearm-camera-margin-m 0.70`: require the ball to move `70 cm` back to the camera side of the plane before rearming
- `--impact-rearm-miss-seconds 1.40`: also rearm if the ball disappears for `1.4` seconds after the last event
- `--no-event-motion-after-reject`: disable extra permissive motion rescue after a rejected event candidate
- `--process-every 1`: process every frame; keep this as low as possible for best tracking, and `1` is preferred when the hardware can sustain it
- `--async-capture`: always consume the most recent camera frame to reduce latency
- `--ball-overlay-ttl 0`: show the live ball overlay only for the current frame
- `--candidate-overlay-ttl 0`: show the candidate overlay only for the current frame
- `--no-trajectory-bridge`: disable synthetic in-between ball prediction and rely on live detections only
- `--perf-breakdown`: print timing breakdowns for diagnosis and tuning
- `--stats-every 90`: print periodic stats every `90` frames

Operational notes:

- `--impact-entry-point-mode deepest` is the preferred mode for this project because it places the marker where the ball reached its maximum depth inside the goal
- if repeated or double-hit events appear after a real goal, increase `--impact-cooldown`
- for best tracking reliability, keep `--process-every` as low as possible; `--process-every 1` is preferred when performance allows
- this reference profile was tested on the macOS Apple Silicon path and should be treated as the starting point for similar installs

Saved files:

- calibration: `data/calibration/goal_calibration.json`
- camera intrinsics: `data/calibration/camera_intrinsics.json`
- ChArUco board image/spec: `data/calibration/charuco_board.png`, `data/calibration/charuco_board.json`
- reference frame: `data/calibration/reference_frame.jpg`
- impact logs: `data/logs/hits.csv`

## Notes for Better Accuracy

- Use high shutter speed and good lighting.
- Keep the background stable.
- Frame the full goal and avoid severe motion blur.
- If false detections occur, tune:
  - `--min-area`
  - `--max-area`
  - `--min-circularity`
- Recalibrate after meaningful camera repositioning.

## Notes for Better FPS

- `motion` detector is the best path to 60fps on CPU.
- Use `--backend avfoundation` on macOS cameras.
- Keep `--adapt-every` around `8` to reduce ORB cost.
- Lower detector resolution:
  - `--motion-scale 0.5` or `0.6` for motion mode
  - `--yolo-imgsz 512` or `640` for YOLO mode
- Run YOLO every 2nd frame with `--process-every 2`.
- If preview rendering is expensive, reduce `--display-scale` (example `0.8`).
- Runtime prints camera mode and periodic perf stats (`--stats-every`).
- Blue candidate marker is detection-level and now has a short persistence window (`--candidate-overlay-ttl`) for visibility.
- If camera is outputting 4K even when 1080 is requested, use:
  - `--probe-camera` to discover working formats
  - `--fourcc` to force codec (for example `MJPG`)
  - `--force-resize-input` to keep processing at 1080p even if camera feed is larger
- To reduce false hits from noise/shadows:
  - keep `--motion-warmup-frames` around `45`
  - keep `--adapt-min-confidence` around `0.5` or higher
  - increase `--impact-min-speed` (example `500`)
  - increase `--track-min-consecutive` (example `8`) for stricter validation
  - lower `--track-max-step` (example `90`) to reject random jumps
  - increase `--impact-min-displacement` (example `120`) for stronger hit motion
  - use `--impact-arm-seconds 1.2` after startup/recalibration
- If the ball is visible but nearly static and not detected:
  - keep `--motion-static-fallback` enabled
  - lower `--motion-static-param2` (example `16`)
  - tune `--motion-static-min-radius` / `--motion-static-max-radius` to your ball size
- To detect slow valid goals, keep entry mode enabled:
  - `--impact-enable-entry`
  - lower `--impact-entry-min-speed` (example `20-35`)
- If the ball only gets detected after it stops, keep motion fallback enabled:
  - `--event-motion-fallback`
  - this runs a permissive motion detector near the goal when YOLO misses a moving ball
- For reliability, prefer full-frame detection:
  - `--detect-full-frame`
  - use `--no-detect-full-frame --detect-roi-margin ...` only if you need extra FPS
- For football throws, tracker lock is often the most reliable:
  - set `--ball-tracker mil --auto-lock-tracker`
  - press `b` before throwing and draw a box around the ball
  - this is optional; default mode does not require manual box selection

## Important Limitation

This is a single-camera 2D approach, so impact detection is heuristic-based.
For production-level reliability, consider:

- higher-speed cameras,
- per-frame ball detector model (e.g. YOLO-based),
- multi-camera triangulation or depth sensing.
