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

from ball_dataset_common import dataset_dirs, next_capture_name, yolo_line_from_bbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live capture tool: freeze frame, draw the ball box, confirm, save image+label, resume camera"
    )
    parser.add_argument("--camera", default="0", help="Camera index or video path/URL")
    parser.add_argument("--backend", choices=["auto", "avfoundation"], default="auto")
    parser.add_argument("--fourcc", default="", help="Optional camera codec FourCC")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--dataset-root", default="datasets/custom_ball")
    parser.add_argument("--target-count", type=int, default=300, help="Stop automatically after this many saved images")
    parser.add_argument("--prefix", default="ball")
    parser.add_argument("--display-scale", type=float, default=0.85, help="Preview-only scale factor")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument(
        "--save-split",
        choices=["auto", "train", "val"],
        default="train",
        help=(
            "Where to save captured frames. "
            "'train' is the production-safe default. 'auto' keeps the old every-N validation split, while 'train'/'val' "
            "lets you capture separate sessions for a more honest holdout."
        ),
    )
    parser.add_argument("--val-every", type=int, default=5, help="Send every Nth saved image to val split (0 = all train)")
    parser.add_argument("--class-id", type=int, default=0, help="YOLO class id to write into labels")
    return parser.parse_args()


def total_saved(dataset_root: Path) -> int:
    total = 0
    for split in ("train", "val"):
        image_dir, _ = dataset_dirs(dataset_root, split)
        total += sum(1 for p in image_dir.iterdir() if p.is_file())
    return total


def split_for_index(index_1_based: int, val_every: int, save_split: str) -> str:
    if save_split in {"train", "val"}:
        return save_split
    if val_every > 0 and index_1_based % val_every == 0:
        return "val"
    return "train"


def save_labeled_frame(
    frame,
    bbox: tuple[int, int, int, int],
    dataset_root: Path,
    split: str,
    prefix: str,
    jpg_quality: int,
    class_id: int,
) -> tuple[Path, Path]:
    image_dir, label_dir = dataset_dirs(dataset_root, split)
    image_path = image_dir / next_capture_name(image_dir, prefix)
    label_path = label_dir / f"{image_path.stem}.txt"
    ok = cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, max(60, min(100, jpg_quality))])
    if not ok:
        raise RuntimeError(f"Failed to write image: {image_path}")
    label_path.write_text(yolo_line_from_bbox(bbox, frame.shape[1], frame.shape[0], class_id=class_id), encoding="utf-8")
    return image_path, label_path


def draw_live_hud(frame, saved_count: int, target_count: int, actual_mode: str, next_split: str) -> None:
    remaining = max(0, target_count - saved_count)
    cv2.putText(frame, f"Saved: {saved_count}/{target_count}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (120, 255, 120), 2)
    cv2.putText(frame, f"Next split: {next_split}   Remaining: {remaining}", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 220, 120), 2)
    cv2.putText(frame, actual_mode, (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 2)
    cv2.putText(frame, "Keys: space=freeze+label  q=quit", (18, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(frame, "Freeze flow: drag the ball box, then press ENTER or SPACE to confirm", (18, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)


def draw_frozen_hud(frame, saved_count: int, target_count: int, next_split: str) -> None:
    cv2.putText(frame, f"Frozen frame  Saved: {saved_count}/{target_count}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (120, 255, 120), 2)
    cv2.putText(frame, f"Next split: {next_split}", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 220, 120), 2)
    cv2.putText(frame, "Draw the ball box, then press ENTER or SPACE to accept", (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(frame, "Press c inside the selector to cancel and return to live camera", (18, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)


def show_frame(window_name: str, frame, display_scale: float) -> None:
    show = frame
    if display_scale < 0.999:
        show = cv2.resize(show, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(window_name, show)


def draw_video_hud(
    frame,
    saved_count: int,
    target_count: int,
    actual_mode: str,
    next_split: str,
    *,
    frame_idx: int,
    total_frames: int,
    fps: float,
    paused: bool,
) -> None:
    remaining = max(0, target_count - saved_count)
    cv2.putText(frame, f"Saved: {saved_count}/{target_count}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (120, 255, 120), 2)
    cv2.putText(frame, f"Next split: {next_split}   Remaining: {remaining}", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 220, 120), 2)
    cv2.putText(frame, actual_mode, (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 2)
    status = "paused" if paused else "playing"
    frame_text = f"Playback: {status}  frame {frame_idx}"
    if total_frames > 0:
        frame_text += f"/{total_frames}"
    if fps > 0.0:
        frame_text += f"  t={(max(0, frame_idx - 1) / fps):.2f}s"
    cv2.putText(frame, frame_text, (18, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 235, 255), 2)
    cv2.putText(frame, "Keys: space=pause/resume  enter=label frame  n=next frame  q=quit", (18, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
    cv2.putText(frame, "When paused, use n to step frame-by-frame, then press ENTER to box the ball", (18, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    saved_count = total_saved(dataset_root)
    source = parse_camera_source(args.camera)
    cap = open_capture(source, args.width, args.height, args.fps, backend=args.backend, fourcc=args.fourcc)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera/video source.")

    actual_mode = (
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        f" @{cap.get(cv2.CAP_PROP_FPS):.1f} codec={fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC))}"
    )
    print(f"[LiveLabel] mode={actual_mode}")
    if args.save_split == "auto":
        print(
            "[LiveLabel] split mode=auto "
            f"(every {max(0, args.val_every)}th frame goes to val; nearby frames may make validation optimistic)"
        )
    else:
        print(f"[LiveLabel] split mode=fixed -> all new samples go to {args.save_split}")

    window_name = "Ball Live Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    video_controls_enabled = isinstance(source, str) and Path(str(source)).exists()
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if video_controls_enabled else 0
    playback_paused = False
    step_frame = False
    current_frame = None
    frame_idx = 0

    if video_controls_enabled:
        print("[LiveLabel] video controls enabled: space=pause/resume, n=next frame while paused, enter=label current frame")

    while True:
        if saved_count >= max(1, args.target_count):
            print(f"[LiveLabel] target reached: {saved_count}/{args.target_count}")
            break

        if current_frame is None or not (video_controls_enabled and playback_paused and not step_frame):
            ok, frame = cap.read()
            if not ok:
                break
            current_frame = frame
            frame_idx += 1
            step_frame = False

        frame = current_frame.copy()

        next_index = saved_count + 1
        next_split = split_for_index(next_index, args.val_every, args.save_split)
        hud = frame.copy()
        if video_controls_enabled:
            draw_video_hud(
                hud,
                saved_count,
                args.target_count,
                actual_mode,
                next_split,
                frame_idx=frame_idx,
                total_frames=video_total_frames,
                fps=float(cap.get(cv2.CAP_PROP_FPS)),
                paused=playback_paused,
            )
        else:
            draw_live_hud(hud, saved_count, args.target_count, actual_mode, next_split)
        show_frame(window_name, hud, args.display_scale)

        key = cv2.waitKey(0 if (video_controls_enabled and playback_paused) else 1) & 0xFF
        if key in (27, ord("q")):
            break
        if video_controls_enabled:
            if key == ord(" "):
                playback_paused = not playback_paused
                continue
            if playback_paused and key in (ord("n"), ord(".")):
                step_frame = True
                playback_paused = True
                continue
            if key not in (13, 10):
                continue
        else:
            if key != ord(" "):
                continue

        frozen = frame.copy()
        frozen_hud = frozen.copy()
        draw_frozen_hud(frozen_hud, saved_count, args.target_count, next_split)
        show_frame(window_name, frozen_hud, args.display_scale)
        cv2.waitKey(1)

        roi = cv2.selectROI(window_name, frozen_hud, fromCenter=False, showCrosshair=True)
        if roi[2] <= 1 or roi[3] <= 1:
            print("[LiveLabel] selection cancelled")
            continue

        image_path, label_path = save_labeled_frame(
            frozen,
            (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])),
            dataset_root=dataset_root,
            split=next_split,
            prefix=args.prefix,
            jpg_quality=args.jpg_quality,
            class_id=args.class_id,
        )
        saved_count += 1
        print(f"[LiveLabel] saved {image_path} + {label_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
