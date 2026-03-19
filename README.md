# Goal Impact Tracker (Python)

This project captures a camera feed (target `1080p @ 60fps`), lets you click the 4 goal corners, and estimates when/where the ball hits inside the calibrated goal plane.

It also includes camera-shift adaptation so small camera movement can still work, with manual recalibration when needed.

## Features

- 4-corner goal calibration by mouse clicks
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

On Apple Silicon, this uses the current official `onnxruntime` package.
`onnxruntime-silicon` is a legacy package and should not be used here.
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

- this helper is currently a standalone Apple-native executable
- it can now be used from `run.py` with `--detector vision`
- to use this path, export your custom detector to `Core ML` from a Python `3.12`/`3.13` Core ML-capable toolchain
- the checked-in [best.mlmodelc](/Users/arnautvasile/Projects/ogilvy/coke-football/data/coreml/best.mlmodelc) metadata still reflects an older `YOLO26n` export, so retrain/export a fresh detector after moving your workflow to `YOLO26s`

Run the main tracker with the Apple helper as detector:

```bash
python run.py \
  --camera 0 \
  --backend avfoundation \
  --detector vision \
  --vision-model /Users/arnautvasile/Projects/ogilvy/coke-football/data/coreml/best.mlmodelc \
  --vision-helper-bin /Users/arnautvasile/Projects/ogilvy/coke-football/apple/BallVisionHelper/.build/release/BallVisionHelper \
  --vision-camera 0 \
  --vision-detect-every 3 \
  --vision-confidence 0.20 \
  --vision-local-search-scale 2.6 \
  --vision-full-recover-every 4 \
  --vision-compute-units all \
  --vision-identity-source /Users/arnautvasile/Projects/ogilvy/coke-football/data/identity/ball_identity.onnx \
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
  --yolo-tracker bytetrack.yaml \
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

- `c`: recalibrate goal corners (also refreshes camera adaptation reference)
- `b`: manually select/lock ball ROI for tracker (only when `--ball-tracker` is not `none`)
- `x`: unlock/reset ball tracker
- `q` or `Esc`: quit

## Calibration & Recalibration

1. On first run, click 4 goal corners and press `Enter`.
2. Small camera movement is auto-adjusted.
3. If adaptation confidence drops or overlays drift, press `c` to recalibrate.

### ChArUco Camera Calibration

Before marker-based goal pose or 3D plane checks, calibrate the camera/lens once with a printed ChArUco board.

Generate a printable board:

```bash
python tools/generate_charuco_board.py \
  --output-image data/calibration/charuco_board.png \
  --output-spec data/calibration/charuco_board.json
```

Print the PNG at `100%` scale with no fit-to-page scaling.

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
- for automatic goal recalibration after camera movement, the next step is ArUco/AprilTag-style markers on the goal frame

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
