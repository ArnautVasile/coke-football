from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goal_tracker.yolo_ball_detection import resolve_yolo_device

from ball_dataset_common import dataset_dirs, label_path_for, list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 1-class custom ball detector with Ultralytics")
    parser.add_argument("--dataset-root", default="datasets/custom_ball")
    parser.add_argument("--prepared-root", default="", help="Optional prepared dataset output directory")
    parser.add_argument("--model", default="yolo26s.pt", help="Starting weights")
    parser.add_argument("--device", default="", help="Preferred train device (cpu, mps, cuda:0)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--multi-scale",
        type=float,
        nargs="?",
        const=0.25,
        default=0.0,
        help=(
            "Multi-scale range as a fraction of --imgsz. "
            "Example: --multi-scale 0.25 varies sizes roughly within +/-25%%. "
            "Passing --multi-scale with no value uses 0.25."
        ),
    )
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic in the last N epochs")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation augmentation in degrees")
    parser.add_argument("--translate", type=float, default=0.10, help="Translation augmentation fraction")
    parser.add_argument("--scale", type=float, default=0.50, help="Scale augmentation gain")
    parser.add_argument("--perspective", type=float, default=0.0, help="Perspective augmentation amount")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Holdout fraction when no val split exists")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/custom_ball")
    parser.add_argument("--name", default="", help="Optional run name (defaults to ball_<model-name>)")
    parser.add_argument("--export-onnx", action="store_true", help="Export best weights to ONNX after training")
    parser.add_argument(
        "--export-imgsz",
        type=int,
        default=0,
        help="Optional ONNX export image size. Defaults to --imgsz when not set.",
    )
    parser.add_argument("--export-opset", type=int, default=0, help="Optional ONNX opset override for export")
    parser.add_argument("--export-no-simplify", dest="export_simplify", action="store_false", help="Disable ONNX simplify during post-train export")
    parser.set_defaults(export_simplify=True)
    return parser.parse_args()


def copy_item(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def gather_split_items(dataset_root: Path, split: str) -> tuple[list[tuple[Path, Path]], list[Path]]:
    image_dir, _ = dataset_dirs(dataset_root, split)
    labeled: list[tuple[Path, Path]] = []
    background_images: list[Path] = []
    for image_path in list_images(image_dir):
        label_path = label_path_for(image_path, dataset_root, split)
        if not label_path.exists():
            background_images.append(image_path)
            continue
        labeled.append((image_path, label_path))
    return labeled, background_images


def split_images(
    images: list[Path],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    if not images:
        return [], []
    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []
    val_count = max(1, int(round(len(shuffled) * max(0.05, min(0.5, val_fraction)))))
    val_count = min(len(shuffled) - 1, val_count)
    val = shuffled[:val_count]
    train = shuffled[val_count:]
    return train, val


def prepare_dataset(args: argparse.Namespace) -> tuple[Path, dict[str, int]]:
    dataset_root = Path(args.dataset_root)
    prepared_root = Path(args.prepared_root) if args.prepared_root else dataset_root / "prepared"
    if prepared_root.resolve() == dataset_root.resolve():
        raise SystemExit("--prepared-root must be different from --dataset-root to avoid overwriting raw captures.")
    if prepared_root.exists():
        shutil.rmtree(prepared_root)

    train_items, train_background = gather_split_items(dataset_root, "train")
    val_items, val_background = gather_split_items(dataset_root, "val")
    if not train_items and not val_items:
        raise SystemExit("No labeled images found. Capture and annotate the ball first.")

    if not train_items and val_items:
        train_items, val_items = val_items, []
        train_background.extend(val_background)
        val_background = []

    if not val_items and not val_background:
        rng = random.Random(args.seed)
        shuffled = train_items[:]
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * max(0.05, min(0.5, args.val_fraction))))) if len(shuffled) > 4 else 1
        val_items = shuffled[:val_count]
        train_items = shuffled[val_count:]
        if not train_items:
            train_items = val_items[:-1]
            val_items = val_items[-1:]
        train_background, val_background = split_images(
            train_background,
            val_fraction=float(args.val_fraction),
            seed=int(args.seed) + 7,
        )

    summary = {
        "train": len(train_items),
        "val": len(val_items),
        "train_background": len(train_background),
        "val_background": len(val_background),
    }
    for split, items in (("train", train_items), ("val", val_items)):
        image_dir, label_dir = dataset_dirs(prepared_root, split)
        for image_path, label_path in items:
            copy_item(image_path, image_dir / image_path.name)
            copy_item(label_path, label_dir / label_path.name)
    for split, background_items in (("train", train_background), ("val", val_background)):
        image_dir, label_dir = dataset_dirs(prepared_root, split)
        for image_path in background_items:
            copy_item(image_path, image_dir / image_path.name)
            empty_label = label_dir / f"{image_path.stem}.txt"
            empty_label.parent.mkdir(parents=True, exist_ok=True)
            empty_label.write_text("", encoding="utf-8")

    dataset_yaml = prepared_root / "ball_dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {prepared_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: ball",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_yaml, summary


def main() -> None:
    args = parse_args()
    dataset_yaml, summary = prepare_dataset(args)
    device = resolve_yolo_device(args.device) or ""
    project_dir = Path(args.project).resolve()
    print(f"[Train] dataset={dataset_yaml}")
    print(
        f"[Train] train_images={summary['train']} (+bg {summary['train_background']}) "
        f"val_images={summary['val']} (+bg {summary['val_background']})"
    )
    print(f"[Train] model={args.model} device={device or 'default'} imgsz={args.imgsz} epochs={args.epochs}")
    print(f"[Train] project_dir={project_dir}")

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Install with: pip install -r requirements-yolo.txt") from exc

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(dataset_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "multi_scale": float(max(0.0, args.multi_scale)),
        "close_mosaic": int(max(0, args.close_mosaic)),
        "degrees": float(args.degrees),
        "translate": float(args.translate),
        "scale": float(args.scale),
        "perspective": float(args.perspective),
        "project": str(project_dir),
        "name": args.name or f"ball_{Path(args.model).stem}",
    }
    if device:
        train_kwargs["device"] = device

    model.train(**train_kwargs)
    save_dir = Path(model.trainer.save_dir).resolve()
    best_weights = save_dir / "weights" / "best.pt"
    print(f"[Train] save_dir={save_dir}")
    print(f"[Train] best_weights={best_weights}")

    recommended_model = best_weights
    runtime_imgsz = args.imgsz

    if args.export_onnx and best_weights.exists():
        export_imgsz = int(args.export_imgsz) if int(args.export_imgsz) > 0 else int(args.imgsz)
        export_kwargs = {
            "format": "onnx",
            "imgsz": export_imgsz,
            "simplify": args.export_simplify,
        }
        if args.export_opset > 0:
            export_kwargs["opset"] = args.export_opset
        exported = YOLO(str(best_weights)).export(**export_kwargs)
        print(f"[Train] exported_onnx={exported}")
        recommended_model = Path(str(exported))
        runtime_imgsz = export_imgsz

    if Path(recommended_model).exists():
        resolved_model = Path(recommended_model).resolve()
        print("[Next] Use the custom ball model in the tracker with:")
        print(
            f"python run.py --camera 0 --detector yolo --no-yolo-track --yolo-imgsz {runtime_imgsz} "
            f'--yolo-model "{resolved_model}" --yolo-class-id 0'
        )


if __name__ == "__main__":
    main()
