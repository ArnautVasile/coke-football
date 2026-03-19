from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dataset_dirs(dataset_root: Path, split: str) -> tuple[Path, Path]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, label_dir


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def label_path_for(image_path: Path, dataset_root: Path, split: str) -> Path:
    _, label_dir = dataset_dirs(dataset_root, split)
    return label_dir / f"{image_path.stem}.txt"


def next_capture_name(image_dir: Path, prefix: str) -> str:
    highest = -1
    for image_path in list_images(image_dir):
        stem = image_path.stem
        if not stem.startswith(prefix):
            continue
        tail = stem[len(prefix):].lstrip("_")
        if tail.isdigit():
            highest = max(highest, int(tail))
    return f"{prefix}_{highest + 1:06d}.jpg"


def yolo_line_from_bbox(bbox: tuple[int, int, int, int], width: int, height: int, class_id: int = 0) -> str:
    x, y, w, h = bbox
    cx = (x + 0.5 * w) / max(width, 1)
    cy = (y + 0.5 * h) / max(height, 1)
    nw = w / max(width, 1)
    nh = h / max(height, 1)
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"
