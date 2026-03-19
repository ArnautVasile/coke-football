from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ball_dataset_common import dataset_dirs, label_path_for, list_images, yolo_line_from_bbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate captured ball images with 1 YOLO box per image")
    parser.add_argument("--dataset-root", default="datasets/custom_ball")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--display-scale", type=float, default=0.9)
    return parser.parse_args()



def load_existing_bbox(label_path: Path, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not label_path.exists():
        return None
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) != 5:
        return None
    _, cx, cy, bw, bh = [float(v) for v in parts]
    w = int(round(bw * width))
    h = int(round(bh * height))
    x = int(round(cx * width - 0.5 * w))
    y = int(round(cy * height - 0.5 * h))
    return x, y, w, h


def draw_preview(image, split: str, index: int, total: int, label_path: Path) -> any:
    preview = image.copy()
    existing = load_existing_bbox(label_path, image.shape[1], image.shape[0])
    if existing is not None:
        x, y, w, h = existing
        cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 200, 0), 2)
    cv2.putText(preview, f"{split} {index + 1}/{total}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    status = "labeled" if label_path.exists() and label_path.read_text(encoding="utf-8").strip() else "no label yet"
    cv2.putText(preview, f"Status: {status}", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 255, 120), 2)
    cv2.putText(preview, "Keys: enter=box  n=no ball  s=skip  b=back  q=quit", (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    return preview


def collect_items(dataset_root: Path, split: str) -> list[tuple[str, Path]]:
    splits = ["train", "val", "test"] if split == "all" else [split]
    items: list[tuple[str, Path]] = []
    for one_split in splits:
        image_dir, _ = dataset_dirs(dataset_root, one_split)
        for image_path in list_images(image_dir):
            items.append((one_split, image_path))
    return items


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    items = collect_items(dataset_root, args.split)
    if not items:
        raise SystemExit(f"No images found under {dataset_root / 'images'}")

    cv2.namedWindow("Ball Annotation", cv2.WINDOW_NORMAL)
    idx = 0
    while 0 <= idx < len(items):
        split, image_path = items[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            idx += 1
            continue
        label_path = label_path_for(image_path, dataset_root, split)
        preview = draw_preview(image, split, idx, len(items), label_path)
        show = preview
        if args.display_scale < 0.999:
            show = cv2.resize(show, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Ball Annotation", show)

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (13, 10, ord(" ")):
            roi = cv2.selectROI("Ball Annotation", image, fromCenter=False, showCrosshair=True)
            if roi[2] > 1 and roi[3] > 1:
                label_path.write_text(yolo_line_from_bbox(roi, image.shape[1], image.shape[0]), encoding="utf-8")
                print(f"[Annotate] saved {label_path}")
                idx += 1
        elif key == ord("n"):
            label_path.write_text("", encoding="utf-8")
            print(f"[Annotate] marked no-ball {label_path}")
            idx += 1
        elif key == ord("s"):
            idx += 1
        elif key == ord("b"):
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
