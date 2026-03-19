from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goal_tracker.app import fourcc_to_text, open_capture, parse_camera_source

from ball_dataset_common import dataset_dirs, next_capture_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture custom ball training frames from a camera or video source")
    parser.add_argument("--camera", default="0", help="Camera index or video path/URL")
    parser.add_argument("--backend", choices=["auto", "avfoundation"], default="auto")
    parser.add_argument("--fourcc", default="", help="Optional camera codec FourCC")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--dataset-root", default="datasets/custom_ball")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--prefix", default="ball")
    parser.add_argument("--save-every", type=int, default=4, help="Burst mode save interval in frames")
    parser.add_argument("--display-scale", type=float, default=0.85, help="Preview-only scale factor")
    parser.add_argument("--jpg-quality", type=int, default=95)
    return parser.parse_args()


def save_frame(frame, image_dir: Path, prefix: str, jpg_quality: int) -> Path:
    path = image_dir / next_capture_name(image_dir, prefix)
    ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, max(60, min(100, jpg_quality))])
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")
    return path


def draw_hud(frame, split: str, saved_count: int, burst_on: bool, save_every: int, actual_mode: str) -> None:
    cv2.putText(frame, f"Dataset split: {split}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(frame, f"Saved: {saved_count}", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (120, 255, 120), 2)
    burst_text = f"Burst: {'ON' if burst_on else 'off'} every {max(1, save_every)} frame(s)"
    cv2.putText(frame, burst_text, (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 220, 120), 2)
    cv2.putText(frame, actual_mode, (18, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 2)
    cv2.putText(
        frame,
        "Keys: space=save  r=burst  1=train  2=val  q=quit",
        (18, 148),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    current_split = args.split
    image_dir, _ = dataset_dirs(dataset_root, current_split)

    source = parse_camera_source(args.camera)
    cap = open_capture(source, args.width, args.height, args.fps, backend=args.backend, fourcc=args.fourcc)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera/video source.")

    actual_mode = (
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        f" @{cap.get(cv2.CAP_PROP_FPS):.1f} codec={fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC))}"
    )
    print(f"[Capture] mode={actual_mode}")

    cv2.namedWindow("Ball Dataset Capture", cv2.WINDOW_NORMAL)
    burst_on = False
    frame_idx = 0
    saved_count = len(list(image_dir.glob("*.jpg")))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        hud = frame.copy()
        draw_hud(hud, current_split, saved_count, burst_on, args.save_every, actual_mode)

        show = hud
        if args.display_scale < 0.999:
            show = cv2.resize(show, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Ball Dataset Capture", show)

        if burst_on and frame_idx % max(1, args.save_every) == 0:
            path = save_frame(frame, image_dir, args.prefix, args.jpg_quality)
            saved_count += 1
            print(f"[Capture] saved {path}")

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord(" "):
            path = save_frame(frame, image_dir, args.prefix, args.jpg_quality)
            saved_count += 1
            print(f"[Capture] saved {path}")
        elif key == ord("r"):
            burst_on = not burst_on
            print(f"[Capture] burst={'on' if burst_on else 'off'}")
        elif key == ord("1"):
            current_split = "train"
            image_dir, _ = dataset_dirs(dataset_root, current_split)
            saved_count = len(list(image_dir.glob('*.jpg')))
            print("[Capture] switched to split=train")
        elif key == ord("2"):
            current_split = "val"
            image_dir, _ = dataset_dirs(dataset_root, current_split)
            saved_count = len(list(image_dir.glob('*.jpg')))
            print("[Capture] switched to split=val")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
