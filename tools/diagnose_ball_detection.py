from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goal_tracker.calibration import load_calibration
from goal_tracker.yolo_ball_detection import YoloBallDetector, YoloConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline ball-detection continuity diagnostics on a video. "
            "Reports detection rate and longest miss streaks for multiple YOLO settings."
        )
    )
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--model", required=True, help="YOLO model path (.pt/.onnx)")
    parser.add_argument("--class-id", type=int, default=0, help="Class id for the ball")
    parser.add_argument("--imgsz", type=int, default=960, help="Requested inference image size")
    parser.add_argument("--device", default="cpu", help="Inference device (cpu/mps/cuda:0)")
    parser.add_argument(
        "--identity-source",
        default="",
        help="Optional exact-ball verifier source for YOLO/hybrid diagnostics",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=0.0,
        help="Optional exact-ball verifier threshold override (<=0 uses learned default)",
    )
    parser.add_argument(
        "--conf-values",
        default="0.08,0.10,0.12,0.15",
        help="Comma-separated confidence values to test",
    )
    parser.add_argument(
        "--tracker-values",
        default="true,false",
        help="Comma-separated tracker toggles to test (true/false)",
    )
    parser.add_argument(
        "--tracker-cfg",
        default="configs/bytetrack_ball.yaml",
        help="Tracker config for YOLO track mode",
    )
    parser.add_argument("--max-frames", type=int, default=900, help="Max frames per config")
    parser.add_argument(
        "--calibration-file",
        default="",
        help="Optional goal calibration JSON; when set, detections run in goal ROI mode",
    )
    parser.add_argument("--roi-margin", type=int, default=180, help="ROI margin around calibrated goal")
    parser.add_argument("--follow-max-age", type=float, default=0.45, help="Seconds to keep ROI follow extension alive")
    parser.add_argument("--follow-radius-scale", type=float, default=3.4, help="Last-ball radius scale for ROI follow extension")
    parser.add_argument("--follow-min-half-size", type=int, default=90, help="Min half-size for ROI follow extension")
    parser.add_argument(
        "--global-search-every",
        type=int,
        default=0,
        help="In ROI mode, run full-frame search every N frames (0 disables)",
    )
    return parser.parse_args()


def parse_conf_values(raw: str) -> list[float]:
    vals: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("No confidence values parsed from --conf-values")
    return vals


def parse_bool_values(raw: str) -> list[bool]:
    vals: list[bool] = []
    for part in raw.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p in {"1", "true", "yes", "on"}:
            vals.append(True)
        elif p in {"0", "false", "no", "off"}:
            vals.append(False)
        else:
            raise ValueError(f"Invalid boolean value in --tracker-values: {part!r}")
    if not vals:
        raise ValueError("No tracker values parsed from --tracker-values")
    return vals


def goal_roi(corners: np.ndarray, frame_shape: tuple[int, int, int], margin: int) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1 = max(0, int(np.floor(np.min(corners[:, 0]))) - margin)
    y1 = max(0, int(np.floor(np.min(corners[:, 1]))) - margin)
    x2 = min(w, int(np.ceil(np.max(corners[:, 0]))) + margin)
    y2 = min(h, int(np.ceil(np.max(corners[:, 1]))) + margin)
    return x1, y1, x2, y2


def miss_run_lengths(sequence: Iterable[int]) -> list[int]:
    runs: list[int] = []
    cur = 0
    for v in sequence:
        if int(v) == 0:
            cur += 1
        elif cur > 0:
            runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return runs


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(np.percentile(arr, q))


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise RuntimeError(f"Video not found: {video_path}")
    if args.calibration_file:
        calibration_path = Path(args.calibration_file)
        if not calibration_path.exists():
            raise RuntimeError(f"Calibration file not found: {calibration_path}")
        calibration = load_calibration(calibration_path)
        corners = calibration.corners_px.astype(np.float32)
    else:
        corners = None

    conf_values = parse_conf_values(args.conf_values)
    tracker_values = parse_bool_values(args.tracker_values)

    print(
        "config,frames,detect_rate,first_detect_frame,longest_miss_after_first,miss_p95,mean_miss,miss_runs,avg_radius"
    )
    for conf in conf_values:
        for use_tracker in tracker_values:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            detector = YoloBallDetector(
                YoloConfig(
                    model=str(args.model),
                    conf=float(conf),
                    imgsz=int(args.imgsz),
                    device=(args.device or "").strip() or None,
                    class_id=int(args.class_id),
                    use_tracker=bool(use_tracker),
                    tracker_cfg=str(args.tracker_cfg),
                    identity_source=str(args.identity_source),
                    identity_threshold=float(args.identity_threshold),
                )
            )

            seen: list[int] = []
            radii: list[float] = []
            frame_idx = 0
            last_center: tuple[int, int] | None = None
            last_radius: float | None = None
            last_seen_t = 0.0

            while frame_idx < max(1, int(args.max_frames)):
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1

                roi = None
                if corners is not None:
                    roi = goal_roi(corners, frame.shape, margin=max(0, int(args.roi_margin)))
                    if (
                        last_center is not None
                        and last_radius is not None
                        and (frame_idx - last_seen_t) <= max(1.0, float(args.follow_max_age) * 60.0)
                    ):
                        h, w = frame.shape[:2]
                        follow_half = int(
                            max(
                                float(args.follow_min_half_size),
                                float(last_radius) * max(1.2, float(args.follow_radius_scale)),
                            )
                        )
                        fx1 = max(0, int(last_center[0]) - follow_half)
                        fy1 = max(0, int(last_center[1]) - follow_half)
                        fx2 = min(w, int(last_center[0]) + follow_half)
                        fy2 = min(h, int(last_center[1]) + follow_half)
                        if fx2 > fx1 and fy2 > fy1:
                            gx1, gy1, gx2, gy2 = roi
                            roi = (min(gx1, fx1), min(gy1, fy1), max(gx2, fx2), max(gy2, fy2))
                    if int(args.global_search_every) > 0 and frame_idx % max(1, int(args.global_search_every)) == 0:
                        roi = None

                det = detector.detect(frame, roi=roi)
                hit = det is not None
                seen.append(1 if hit else 0)
                if hit and det is not None:
                    radii.append(float(det.radius))
                    last_center = det.center
                    last_radius = det.radius
                    last_seen_t = frame_idx

            cap.release()

            total = len(seen)
            if total == 0:
                continue
            detect_rate = float(sum(seen) / total)
            first_detect = next((i for i, v in enumerate(seen) if v == 1), -1)
            if first_detect >= 0:
                post = seen[first_detect:]
            else:
                post = seen
            runs = miss_run_lengths(post)
            longest = max(runs) if runs else 0
            p95 = percentile(runs, 95.0) if runs else 0.0
            mean_run = float(np.mean(runs)) if runs else 0.0
            avg_radius = float(np.mean(radii)) if radii else 0.0
            label = f"conf={conf:.2f}|tracker={str(use_tracker).lower()}"
            print(
                f"{label},{total},{detect_rate:.4f},{first_detect},{longest},{p95:.2f},{mean_run:.2f},{len(runs)},{avg_radius:.2f}"
            )


if __name__ == "__main__":
    main()
