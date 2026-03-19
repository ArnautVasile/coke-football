from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goal_tracker.ball_detection import BallDetection
from goal_tracker.yolo_ball_detection import YoloBallDetector, YoloConfig

from ball_dataset_common import dataset_dirs, label_path_for, list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate full YOLO ball system (detector + optional identity verifier) "
            "on labeled unseen data."
        )
    )
    parser.add_argument("--dataset-root", default="datasets/custom_ball_holdout", help="Dataset root")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate (usually test)")
    parser.add_argument("--model", required=True, help="YOLO model path (.pt/.onnx)")
    parser.add_argument("--class-id", type=int, default=0, help="Ball class id in labels and model")
    parser.add_argument("--conf", type=float, default=0.10, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size")
    parser.add_argument("--device", default="", help="YOLO device (cpu/mps/cuda:0)")
    parser.add_argument("--tracker", dest="use_tracker", action="store_true", help="Use tracker mode during evaluation")
    parser.add_argument("--no-tracker", dest="use_tracker", action="store_false", help="Disable tracker mode (recommended for still-image evaluation)")
    parser.set_defaults(use_tracker=False)
    parser.add_argument("--tracker-cfg", default="configs/bytetrack_ball.yaml", help="Tracker config path")
    parser.add_argument(
        "--identity-source",
        default="",
        help="Optional verifier source (.onnx/.npz or dataset root) for end-to-end system evaluation",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=0.0,
        help="Optional verifier threshold override (<=0 uses learned default)",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.50, help="IoU threshold for TP matching")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap for quick checks")
    parser.add_argument("--output-json", default="", help="Optional summary JSON output path")
    parser.add_argument("--output-csv", default="", help="Optional per-image CSV output path")
    return parser.parse_args()


def _parse_label_boxes(label_path: Path, width: int, height: int, class_id: int) -> list[tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx = float(parts[1]) * float(width)
            cy = float(parts[2]) * float(height)
            bw = float(parts[3]) * float(width)
            bh = float(parts[4]) * float(height)
        except Exception:
            continue
        if cid != class_id:
            continue
        x1 = max(0.0, cx - 0.5 * bw)
        y1 = max(0.0, cy - 0.5 * bh)
        x2 = min(float(width), cx + 0.5 * bw)
        y2 = min(float(height), cy + 0.5 * bh)
        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue
        boxes.append((x1, y1, x2, y2))
    return boxes


def _detection_to_box_xyxy(det: BallDetection, width: int, height: int) -> tuple[float, float, float, float]:
    cx = float(det.center[0])
    cy = float(det.center[1])
    r = max(1.0, float(det.radius))
    x1 = max(0.0, cx - r)
    y1 = max(0.0, cy - r)
    x2 = min(float(width), cx + r)
    y2 = min(float(height), cy + r)
    return (x1, y1, x2, y2)


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = max(1e-9, area_a + area_b - inter)
    return float(inter / union)


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    split = args.split.strip() or "test"
    image_dir, _ = dataset_dirs(dataset_root, split)
    images = list_images(image_dir)
    if not images:
        raise SystemExit(f"No images found under {image_dir.resolve()}")
    if args.max_images > 0:
        images = images[: int(args.max_images)]

    detector = YoloBallDetector(
        YoloConfig(
            model=str(args.model),
            conf=float(args.conf),
            imgsz=int(args.imgsz),
            device=(args.device or "").strip() or None,
            class_id=int(args.class_id),
            use_tracker=bool(args.use_tracker),
            tracker_cfg=str(args.tracker_cfg),
            identity_source=str(args.identity_source),
            identity_threshold=float(args.identity_threshold),
        )
    )

    iou_thr = float(max(0.0, min(1.0, args.iou_threshold)))
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    fp_negative = 0
    fp_positive_mismatch = 0
    positive_frames = 0
    negative_frames = 0
    pred_on_positive_frames = 0
    ious_tp: list[float] = []
    center_errors_tp: list[float] = []
    inference_ms: list[float] = []
    rows: list[dict[str, str | int | float]] = []

    for idx, image_path in enumerate(images):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        label_path = label_path_for(image_path, dataset_root, split)
        gt_boxes = _parse_label_boxes(label_path, w, h, class_id=int(args.class_id))
        gt_count = len(gt_boxes)
        has_gt = gt_count > 0
        if has_gt:
            positive_frames += 1
        else:
            negative_frames += 1

        t0 = time.perf_counter()
        det = detector.detect(frame, roi=None)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        inference_ms.append(dt_ms)

        pred_box = _detection_to_box_xyxy(det, w, h) if det is not None else None
        pred_present = pred_box is not None
        if has_gt and pred_present:
            pred_on_positive_frames += 1

        best_iou = 0.0
        best_gt: tuple[float, float, float, float] | None = None
        if pred_box is not None and gt_boxes:
            for gt in gt_boxes:
                iou = _box_iou(pred_box, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

        outcome = "tn"
        if pred_box is None and not has_gt:
            tn += 1
            outcome = "tn"
        elif pred_box is None and has_gt:
            fn += gt_count
            outcome = "fn"
        elif pred_box is not None and not has_gt:
            fp += 1
            fp_negative += 1
            outcome = "fp"
        else:
            if best_iou >= iou_thr and best_gt is not None:
                tp += 1
                missed = max(0, gt_count - 1)
                fn += missed
                ious_tp.append(float(best_iou))
                px, py = _box_center(pred_box)
                gx, gy = _box_center(best_gt)
                center_errors_tp.append(float(math.hypot(px - gx, py - gy)))
                outcome = "tp"
            else:
                fp += 1
                fp_positive_mismatch += 1
                fn += gt_count
                outcome = "fp_fn"

        row = {
            "index": idx,
            "image": image_path.name,
            "gt_count": gt_count,
            "pred_present": int(pred_present),
            "best_iou": float(best_iou),
            "outcome": outcome,
            "inference_ms": float(dt_ms),
        }
        rows.append(row)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    positive_detection_rate = _safe_div(pred_on_positive_frames, positive_frames)
    false_alarm_rate = _safe_div(fp_negative, negative_frames)
    specificity = _safe_div(tn, tn + fp_negative)
    mean_iou_tp = float(np.mean(ious_tp)) if ious_tp else 0.0
    mean_center_err_tp = float(np.mean(center_errors_tp)) if center_errors_tp else 0.0
    median_center_err_tp = float(np.median(center_errors_tp)) if center_errors_tp else 0.0
    total_ms = float(np.sum(inference_ms)) if inference_ms else 0.0
    mean_inference_ms = float(np.mean(inference_ms)) if inference_ms else 0.0
    throughput_fps = _safe_div(len(rows), total_ms / 1000.0) if total_ms > 0.0 else 0.0

    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "split": split,
        "images_evaluated": int(len(rows)),
        "model": str(Path(args.model).resolve() if Path(args.model).exists() else args.model),
        "class_id": int(args.class_id),
        "settings": {
            "conf": float(args.conf),
            "imgsz": int(args.imgsz),
            "device": (args.device or "").strip() or "default",
            "use_tracker": bool(args.use_tracker),
            "tracker_cfg": str(args.tracker_cfg),
            "identity_source": str(args.identity_source),
            "identity_threshold": float(args.identity_threshold),
            "iou_threshold": float(iou_thr),
        },
        "counts": {
            "positive_frames": int(positive_frames),
            "negative_frames": int(negative_frames),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "fp_negative": int(fp_negative),
            "fp_positive_mismatch": int(fp_positive_mismatch),
        },
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity_negative_frames": float(specificity),
            "positive_frame_detection_rate": float(positive_detection_rate),
            "false_alarm_rate_negative_frames": float(false_alarm_rate),
            "mean_iou_tp": float(mean_iou_tp),
            "mean_center_error_px_tp": float(mean_center_err_tp),
            "median_center_error_px_tp": float(median_center_err_tp),
            "mean_inference_ms": float(mean_inference_ms),
            "throughput_fps": float(throughput_fps),
        },
    }

    print(f"[Eval] dataset={summary['dataset_root']} split={split} images={len(rows)}")
    print(
        f"[Eval] positives={positive_frames} negatives={negative_frames} "
        f"tp={tp} fp={fp} fn={fn} tn={tn}"
    )
    if positive_frames == 0:
        print(
            "[Eval] warning: no positive labels found in this split. "
            "Precision/recall are not meaningful until you annotate ball boxes."
        )
    print(
        f"[Eval] precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
        f"mean_iou_tp={mean_iou_tp:.4f}"
    )
    print(
        f"[Eval] pos_detect_rate={positive_detection_rate:.4f} "
        f"false_alarm_rate={false_alarm_rate:.4f} specificity={specificity:.4f}"
    )
    print(f"[Eval] mean_inference_ms={mean_inference_ms:.2f} throughput_fps={throughput_fps:.2f}")

    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "index",
                    "image",
                    "gt_count",
                    "pred_present",
                    "best_iou",
                    "outcome",
                    "inference_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Eval] per_image_csv={csv_path.resolve()}")

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(summary)
        payload["per_image"] = rows
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[Eval] summary_json={json_path.resolve()}")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
