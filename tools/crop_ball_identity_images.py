from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goal_tracker.ball_identity import _crop_ball_square
from capture_ball_identity_video import draw_hud, save_identity_sample, show_frame

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label a small set of still images as exact-ball verifier samples."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing sampled frames")
    parser.add_argument("--output-root", default="datasets/ball_identity_smoke")
    parser.add_argument("--prefix", default="ball_id_smoke")
    parser.add_argument("--padding-scale", type=float, default=1.15)
    parser.add_argument("--display-scale", type=float, default=0.85)
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--jpg-quality", type=int, default=95)
    return parser.parse_args()


def list_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    images = list_images(input_dir)[: max(1, int(args.max_images))]
    if not images:
        raise SystemExit("No images found to crop.")

    output_root = Path(args.output_root)
    positives_dir = output_root / "positives"
    saved_count = sum(1 for p in positives_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)

    window_name = "Ball Identity Smoke Crop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for index, image_path in enumerate(images, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[IdentitySmoke] skipped unreadable image: {image_path}")
            continue
        while True:
            hud = frame.copy()
            draw_hud(
                hud,
                saved_count=saved_count,
                max_samples=len(images),
                sample_every=1,
                mode=f"still image {index}/{len(images)}",
                paused=True,
            )
            cv2.putText(hud, f"File: {image_path.name}", (18, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
            show_frame(window_name, hud, args.display_scale)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord('q')):
                cv2.destroyAllWindows()
                return
            if key == ord('s'):
                print(f"[IdentitySmoke] skipped {image_path.name}")
                break
            if key != ord(' '):
                continue

            select_frame = frame.copy()
            show_frame(window_name, select_frame, args.display_scale)
            cv2.waitKey(1)
            roi = cv2.selectROI(window_name, select_frame, fromCenter=False, showCrosshair=True)
            if roi[2] <= 1 or roi[3] <= 1:
                print("[IdentitySmoke] selection cancelled")
                continue
            center = (int(round(roi[0] + 0.5 * roi[2])), int(round(roi[1] + 0.5 * roi[3])))
            radius = max(roi[2], roi[3]) * 0.5
            crop = _crop_ball_square(frame, center, radius, scale=max(1.0, float(args.padding_scale)))
            if crop is None:
                print("[IdentitySmoke] crop rejected")
                continue
            saved = save_identity_sample(
                crop=crop,
                full_frame=frame,
                roi_xywh=(int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])),
                output_root=output_root,
                prefix=args.prefix,
                jpg_quality=args.jpg_quality,
                padding_scale=float(args.padding_scale),
            )
            saved_count += 1
            print(f"[IdentitySmoke] saved {saved}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
