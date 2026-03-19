from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from ball_dataset_common import dataset_dirs, yolo_line_from_bbox

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class IdentityItem:
    video_name: str
    frame_path: Path
    box_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy identity capture folders (frames+boxes) into a fresh YOLO detector dataset "
            "without changing the source folders."
        )
    )
    parser.add_argument("--identity-root", default="datasets/ball_identity_mod", help="Root with video_* folders or a single frames/boxes folder")
    parser.add_argument("--dataset-root", default="datasets/custom_ball_from_identity_v1", help="Output YOLO dataset root (images/labels train/val)")
    parser.add_argument("--video-glob", default="video_*", help="Folder glob under --identity-root when root has multiple videos")
    parser.add_argument(
        "--val-videos",
        default="",
        help=(
            "Comma-separated video folder names routed fully to val split "
            "(e.g. video_000010,video_000011). If empty, a random val split is used."
        ),
    )
    parser.add_argument("--val-fraction", type=float, default=0.20, help="Used only when --val-videos is empty")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for val split when --val-videos is empty")
    parser.add_argument("--class-id", type=int, default=0, help="YOLO class id for written labels")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output dataset root if it already exists")
    parser.add_argument("--write-yaml", action="store_true", help="Write ball_dataset.yaml in output root")
    parser.add_argument("--no-write-yaml", dest="write_yaml", action="store_false", help="Do not write dataset YAML")
    parser.set_defaults(write_yaml=True)
    return parser.parse_args()


def _parse_name_set(text: str) -> set[str]:
    if not text.strip():
        return set()
    return {part.strip() for part in text.split(",") if part.strip()}


def _bbox_from_identity_json(box_path: Path, image_w: int, image_h: int) -> tuple[int, int, int, int] | None:
    try:
        payload = json.loads(box_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    bbox = payload.get("bbox_xywh")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = [int(round(float(v))) for v in bbox]
    except Exception:
        return None
    if w <= 1 or h <= 1:
        return None
    x = max(0, min(image_w - 1, x))
    y = max(0, min(image_h - 1, y))
    x2 = max(x + 1, min(image_w, x + w))
    y2 = max(y + 1, min(image_h, y + h))
    w = x2 - x
    h = y2 - y
    if w <= 1 or h <= 1:
        return None
    return (x, y, w, h)


def _collect_sources(identity_root: Path, video_glob: str) -> list[Path]:
    if (identity_root / "frames").exists() and (identity_root / "boxes").exists():
        return [identity_root]
    return sorted(
        p
        for p in identity_root.glob(video_glob)
        if p.is_dir() and (p / "frames").exists() and (p / "boxes").exists()
    )


def _collect_items(sources: list[Path]) -> list[IdentityItem]:
    items: list[IdentityItem] = []
    for source in sources:
        frames_dir = source / "frames"
        boxes_dir = source / "boxes"
        for frame_path in sorted(frames_dir.iterdir()):
            if not frame_path.is_file() or frame_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            box_path = boxes_dir / f"{frame_path.stem}.json"
            if not box_path.exists():
                continue
            items.append(IdentityItem(video_name=source.name, frame_path=frame_path, box_path=box_path))
    return items


def _split_items(
    items: list[IdentityItem],
    *,
    val_videos: set[str],
    val_fraction: float,
    seed: int,
) -> tuple[list[IdentityItem], list[IdentityItem]]:
    if not items:
        return [], []

    if val_videos:
        val = [item for item in items if item.video_name in val_videos]
        train = [item for item in items if item.video_name not in val_videos]
        if not train or not val:
            raise SystemExit(
                "Invalid --val-videos split. Ensure at least one video remains in train and at least one in val."
            )
        return train, val

    shuffled = items[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []
    frac = max(0.05, min(0.5, float(val_fraction)))
    val_count = max(1, int(round(len(shuffled) * frac)))
    val_count = min(len(shuffled) - 1, val_count)
    val = shuffled[:val_count]
    train = shuffled[val_count:]
    return train, val


def _copy_items(items: list[IdentityItem], *, dataset_root: Path, split: str, class_id: int) -> tuple[int, int]:
    image_dir, label_dir = dataset_dirs(dataset_root, split)
    copied = 0
    skipped_bad = 0
    for item in items:
        image = cv2.imread(str(item.frame_path))
        if image is None:
            skipped_bad += 1
            continue
        h, w = image.shape[:2]
        bbox = _bbox_from_identity_json(item.box_path, w, h)
        if bbox is None:
            skipped_bad += 1
            continue

        out_name = f"{item.video_name}__{item.frame_path.name}"
        out_image = image_dir / out_name
        out_label = label_dir / f"{Path(out_name).stem}.txt"

        shutil.copy2(item.frame_path, out_image)
        out_label.write_text(yolo_line_from_bbox(bbox, w, h, class_id=class_id), encoding="utf-8")
        copied += 1
    return copied, skipped_bad


def _write_dataset_yaml(dataset_root: Path) -> Path:
    path = dataset_root / "ball_dataset.yaml"
    path.write_text(
        "\n".join(
            [
                f"path: {dataset_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: ball",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    args = parse_args()
    identity_root = Path(args.identity_root)
    dataset_root = Path(args.dataset_root)
    if not identity_root.exists():
        raise SystemExit(f"Identity root not found: {identity_root}")
    if dataset_root.exists():
        if args.overwrite:
            shutil.rmtree(dataset_root)
        else:
            raise SystemExit(f"Dataset root already exists: {dataset_root}. Use --overwrite to rebuild it.")

    sources = _collect_sources(identity_root, args.video_glob)
    if not sources:
        raise SystemExit(
            f"No identity sources found under {identity_root} matching {args.video_glob} with frames/boxes folders."
        )
    items = _collect_items(sources)
    if not items:
        raise SystemExit("No frame+box pairs found in selected identity sources.")

    val_videos = _parse_name_set(args.val_videos)
    train_items, val_items = _split_items(
        items,
        val_videos=val_videos,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    if not train_items or not val_items:
        raise SystemExit(
            "Split produced an empty train or val set. Adjust --val-videos or --val-fraction."
        )

    train_copied, train_bad = _copy_items(train_items, dataset_root=dataset_root, split="train", class_id=int(args.class_id))
    val_copied, val_bad = _copy_items(val_items, dataset_root=dataset_root, split="val", class_id=int(args.class_id))
    yaml_path = _write_dataset_yaml(dataset_root) if args.write_yaml else None

    print(f"[Identity->Detector] identity_root={identity_root.resolve()}")
    print(f"[Identity->Detector] dataset_root={dataset_root.resolve()}")
    print(f"[Identity->Detector] sources={len(sources)} items_found={len(items)}")
    print(f"[Identity->Detector] train_items={len(train_items)} copied={train_copied} skipped_bad={train_bad}")
    print(f"[Identity->Detector] val_items={len(val_items)} copied={val_copied} skipped_bad={val_bad}")
    if val_videos:
        print(f"[Identity->Detector] val_videos={sorted(val_videos)}")
    else:
        print(f"[Identity->Detector] split=random val_fraction={float(args.val_fraction):.2f} seed={int(args.seed)}")
    if yaml_path is not None:
        print(f"[Identity->Detector] dataset_yaml={yaml_path.resolve()}")
    print("[Next] Train detector with:")
    print(
        f"python tools/train_ball_detector.py --dataset-root {dataset_root} "
        "--model yolo26s.pt --device mps --imgsz 960 --epochs 80 --batch 8 --export-onnx"
    )


if __name__ == "__main__":
    main()
