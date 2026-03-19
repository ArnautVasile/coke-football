from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from ball_dataset_common import dataset_dirs, label_path_for


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames from an unseen video into a dataset split (default: test) "
            "for fair holdout evaluation."
        )
    )
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--dataset-root", default="datasets/custom_ball_holdout", help="Dataset root")
    parser.add_argument("--split", default="test", help="Dataset split name (usually test)")
    parser.add_argument("--prefix", default="holdout", help="Output image name prefix")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=12,
        help="When --target-count is 0, keep every Nth frame",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=0,
        help="If >0, sample this many frames uniformly in the selected time range",
    )
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end-sec", type=float, default=0.0, help="End time in seconds (0 = to video end)")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite images that already exist")
    parser.add_argument(
        "--write-empty-labels",
        dest="write_empty_labels",
        action="store_true",
        help="Create empty label files for each extracted image",
    )
    parser.add_argument(
        "--no-write-empty-labels",
        dest="write_empty_labels",
        action="store_false",
        help="Do not create empty label files",
    )
    parser.set_defaults(write_empty_labels=True)
    parser.add_argument(
        "--write-yaml",
        dest="write_yaml",
        action="store_true",
        help="Write dataset YAML with a test split for Ultralytics val",
    )
    parser.add_argument(
        "--no-write-yaml",
        dest="write_yaml",
        action="store_false",
        help="Do not write dataset YAML",
    )
    parser.set_defaults(write_yaml=True)
    return parser.parse_args()


def _uniform_indices(start_frame: int, end_frame: int, target_count: int) -> set[int]:
    if target_count <= 0:
        return set()
    if end_frame < start_frame:
        return set()
    if target_count == 1:
        return {start_frame}
    span = float(end_frame - start_frame)
    values: set[int] = set()
    for i in range(target_count):
        rel = span * (float(i) / float(target_count - 1))
        idx = int(round(float(start_frame) + rel))
        values.add(max(start_frame, min(end_frame, idx)))
    return values


def _split_rel(dataset_root: Path, split: str) -> tuple[str, str]:
    train_rel = "images/train" if (dataset_root / "images" / "train").exists() else f"images/{split}"
    val_rel = "images/val" if (dataset_root / "images" / "val").exists() else f"images/{split}"
    return train_rel, val_rel


def _write_dataset_yaml(dataset_root: Path, split: str) -> Path:
    train_rel, val_rel = _split_rel(dataset_root, split)
    yaml_path = dataset_root / "ball_dataset_test.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_root.resolve()}",
                f"train: {train_rel}",
                f"val: {val_rel}",
                f"test: images/{split}",
                "names:",
                "  0: ball",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    dataset_root = Path(args.dataset_root)
    split = args.split.strip() or "test"
    image_dir, _ = dataset_dirs(dataset_root, split)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = total_frames_raw if total_frames_raw > 0 else -1

    start_frame = 0
    if args.start_sec > 0.0:
        if fps <= 0.0:
            raise SystemExit("Video FPS is unavailable; cannot use --start-sec reliably.")
        start_frame = max(0, int(round(args.start_sec * fps)))

    if args.end_sec > 0.0:
        if fps <= 0.0:
            raise SystemExit("Video FPS is unavailable; cannot use --end-sec reliably.")
        end_frame = int(round(args.end_sec * fps))
    elif total_frames > 0:
        end_frame = total_frames - 1
    else:
        end_frame = 2**31 - 1

    if total_frames > 0:
        end_frame = min(end_frame, total_frames - 1)
    if end_frame < start_frame:
        raise SystemExit(f"Invalid range: start_frame={start_frame}, end_frame={end_frame}")

    target_count = max(0, int(args.target_count))
    sample_every = max(1, int(args.sample_every))
    keep_set: set[int] | None = None
    if target_count > 0:
        if total_frames <= 0:
            raise SystemExit(
                "CAP_PROP_FRAME_COUNT is unavailable for this video, so --target-count cannot be used. "
                "Use --sample-every instead."
            )
        keep_set = _uniform_indices(start_frame, end_frame, target_count)
        if not keep_set:
            raise SystemExit("No frame indices selected from the requested --target-count and time range.")

    saved = 0
    skipped_existing = 0
    visited = 0
    rows: list[tuple[str, int, float]] = []
    frame_idx = 0
    jpg_quality = max(60, min(100, int(args.jpg_quality)))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx < start_frame:
            frame_idx += 1
            continue
        if frame_idx > end_frame:
            break

        if keep_set is not None:
            keep = frame_idx in keep_set
        else:
            keep = ((frame_idx - start_frame) % sample_every) == 0
        if keep:
            ts = (frame_idx / fps) if fps > 0.0 else -1.0
            ts_tag = f"{ts:010.3f}".replace(".", "p") if ts >= 0.0 else "unknown"
            stem = f"{args.prefix}_f{frame_idx:07d}_t{ts_tag}"
            image_path = image_dir / f"{stem}.jpg"
            if image_path.exists() and not args.overwrite:
                skipped_existing += 1
            else:
                ok_write = cv2.imwrite(
                    str(image_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpg_quality],
                )
                if not ok_write:
                    raise RuntimeError(f"Failed to write image: {image_path}")
                if args.write_empty_labels:
                    label_path = label_path_for(image_path, dataset_root, split)
                    if args.overwrite or not label_path.exists():
                        label_path.write_text("", encoding="utf-8")
                saved += 1
                rows.append((image_path.name, frame_idx, ts))

        visited += 1
        if visited % 300 == 0:
            print(f"[Holdout] scanned={visited} saved={saved}")
        frame_idx += 1

    cap.release()

    manifest_path = dataset_root / f"{split}_{args.prefix}_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "frame_index", "time_seconds", "source_video"])
        for name, idx, ts in rows:
            writer.writerow([name, idx, f"{ts:.6f}" if ts >= 0.0 else "", str(video_path.resolve())])

    yaml_path = _write_dataset_yaml(dataset_root, split) if args.write_yaml else None

    print(f"[Holdout] video={video_path.resolve()}")
    print(f"[Holdout] dataset_root={dataset_root.resolve()} split={split}")
    print(f"[Holdout] fps={'%.3f' % fps if fps > 0.0 else 'unknown'} total_frames={total_frames_raw}")
    print(f"[Holdout] range_frames=[{start_frame}, {end_frame}]")
    if keep_set is not None:
        print(f"[Holdout] sampling=uniform target_count={target_count} selected_indices={len(keep_set)}")
    else:
        print(f"[Holdout] sampling=every_{sample_every}_frames")
    print(f"[Holdout] saved={saved} skipped_existing={skipped_existing}")
    print(f"[Holdout] manifest={manifest_path.resolve()}")
    if yaml_path is not None:
        print(f"[Holdout] dataset_yaml={yaml_path.resolve()}")
    print("[Next] Annotate extracted frames and mark no-ball frames with key 'n':")
    print(f"python tools/annotate_ball_dataset.py --dataset-root {dataset_root} --split {split}")


if __name__ == "__main__":
    main()
