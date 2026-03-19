from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goal_tracker.app import fourcc_to_text, open_capture, parse_camera_source
from goal_tracker.ball_identity import _crop_ball_square

from ball_dataset_common import next_capture_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a video/camera feed and save manually selected exact-ball crops for the verifier."
    )
    parser.add_argument("--source", default="0", help="Camera index or video path/URL")
    parser.add_argument("--backend", choices=["auto", "avfoundation", "dshow", "msmf"], default="auto")
    parser.add_argument("--fourcc", default="", help="Optional camera codec FourCC")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output-root", default="datasets/ball_identity")
    parser.add_argument("--prefix", default="ball_id")
    parser.add_argument("--sample-every", type=int, default=8, help="Pause every Nth frame for possible labeling")
    parser.add_argument("--padding-scale", type=float, default=1.35, help="How much context to keep around the selected box")
    parser.add_argument("--display-scale", type=float, default=0.85, help="Preview-only scale factor")
    parser.add_argument("--max-samples", type=int, default=240)
    parser.add_argument("--jpg-quality", type=int, default=95)
    return parser.parse_args()


def show_frame(window_name: str, frame, display_scale: float) -> None:
    show = frame
    if display_scale < 0.999:
        show = cv2.resize(show, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(window_name, show)


def draw_hud(frame, saved_count: int, max_samples: int, sample_every: int, mode: str, paused: bool) -> None:
    status = "paused sample" if paused else "live scan"
    cv2.putText(frame, f"Saved exact-ball crops: {saved_count}/{max_samples}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (120, 255, 120), 2)
    cv2.putText(frame, f"Mode: {mode}   sampling every {sample_every} frame(s)", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 220, 130), 2)
    cv2.putText(frame, f"Status: {status}", (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(frame, "Keys: space=label crop  s=skip paused sample  q=quit", (18, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)


def save_identity_sample(
    *,
    crop,
    full_frame,
    roi_xywh: tuple[int, int, int, int],
    output_root: Path,
    prefix: str,
    jpg_quality: int,
    padding_scale: float,
) -> Path:
    positives_dir = output_root / "positives"
    frames_dir = output_root / "frames"
    boxes_dir = output_root / "boxes"
    positives_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    boxes_dir.mkdir(parents=True, exist_ok=True)

    image_name = next_capture_name(positives_dir, prefix)
    crop_path = positives_dir / image_name
    frame_path = frames_dir / image_name
    box_path = boxes_dir / f"{Path(image_name).stem}.json"

    options = [cv2.IMWRITE_JPEG_QUALITY, max(60, min(100, jpg_quality))]
    ok_crop = cv2.imwrite(str(crop_path), crop, options)
    ok_frame = cv2.imwrite(str(frame_path), full_frame, options)
    if not ok_crop or not ok_frame:
        raise RuntimeError(f"Failed to write identity sample: {crop_path}")

    x, y, w, h = [int(v) for v in roi_xywh]
    box_path.write_text(
        json.dumps(
            {
                "bbox_xywh": [x, y, w, h],
                "frame_width": int(full_frame.shape[1]),
                "frame_height": int(full_frame.shape[0]),
                "crop_padding_scale": float(padding_scale),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return crop_path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_dir = output_root / "positives"
    saved_count = sum(1 for p in output_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    source = parse_camera_source(args.source)
    cap = open_capture(source, args.width, args.height, args.fps, backend=args.backend, fourcc=args.fourcc)
    if not cap.isOpened():
        raise RuntimeError("Could not open video/camera source.")

    actual_mode = (
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        f" @{cap.get(cv2.CAP_PROP_FPS):.1f} codec={fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC))}"
    )
    print(f"[IdentityCapture] mode={actual_mode}")
    print(f"[IdentityCapture] output_dir={output_dir.resolve()}")

    window_name = "Ball Identity Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    while True:
        if saved_count >= max(1, args.max_samples):
            print(f"[IdentityCapture] target reached: {saved_count}/{args.max_samples}")
            break

        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        paused = frame_idx % max(1, args.sample_every) == 0
        hud = frame.copy()
        draw_hud(hud, saved_count, args.max_samples, args.sample_every, actual_mode, paused)
        show_frame(window_name, hud, args.display_scale)

        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key in (27, ord("q")):
            break
        if not paused:
            continue
        if key == ord("s"):
            continue
        if key != ord(" "):
            continue

        frozen = frame.copy()
        select_frame = frozen.copy()
        draw_hud(select_frame, saved_count, args.max_samples, args.sample_every, actual_mode, True)
        show_frame(window_name, select_frame, args.display_scale)
        cv2.waitKey(1)
        roi = cv2.selectROI(window_name, select_frame, fromCenter=False, showCrosshair=True)
        if roi[2] <= 1 or roi[3] <= 1:
            print("[IdentityCapture] selection cancelled")
            continue
        center = (int(round(roi[0] + 0.5 * roi[2])), int(round(roi[1] + 0.5 * roi[3])))
        radius = max(roi[2], roi[3]) * 0.5
        crop = _crop_ball_square(frozen, center, radius, scale=max(1.0, float(args.padding_scale)))
        if crop is None:
            print("[IdentityCapture] crop rejected")
            continue
        image_path = save_identity_sample(
            crop=crop,
            full_frame=frozen,
            roi_xywh=(int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])),
            output_root=output_root,
            prefix=args.prefix,
            jpg_quality=args.jpg_quality,
            padding_scale=float(args.padding_scale),
        )
        saved_count += 1
        print(f"[IdentityCapture] saved {image_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
