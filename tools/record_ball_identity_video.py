from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goal_tracker.app import fourcc_to_text, open_capture, parse_camera_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a ball-identity video from the project camera for later verifier crop capture."
    )
    parser.add_argument("--camera", default="0", help="Camera index or video path/URL")
    parser.add_argument("--backend", choices=["auto", "avfoundation", "dshow", "msmf"], default="auto")
    parser.add_argument("--fourcc", default="", help="Optional capture codec FourCC")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--duration-seconds", type=int, default=60, help="How long to record once started")
    parser.add_argument("--countdown-seconds", type=int, default=3, help="Countdown before recording starts")
    parser.add_argument("--output", default="", help="Optional explicit output video path")
    parser.add_argument("--output-dir", default="data/identity/raw_video", help="Directory for recorded videos when --output is omitted")
    parser.add_argument("--prefix", default="ball_identity", help="Filename prefix when --output is omitted")
    parser.add_argument("--display-scale", type=float, default=0.85, help="Preview-only scale factor")
    parser.add_argument("--writer-fourcc", default="mp4v", help="VideoWriter codec (default: mp4v)")
    parser.add_argument("--no-preview", action="store_true", help="Record without showing a live preview window")
    return parser.parse_args()


def next_video_path(output_dir: Path, prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob(f"{prefix}_*.mp4"))
    highest = -1
    for path in existing:
        tail = path.stem[len(prefix):].lstrip("_")
        if tail.isdigit():
            highest = max(highest, int(tail))
    return output_dir / f"{prefix}_{highest + 1:06d}.mp4"


def show_frame(window_name: str, frame, display_scale: float) -> None:
    show = frame
    if display_scale < 0.999:
        show = cv2.resize(show, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(window_name, show)


def draw_status(frame, *lines: str) -> None:
    styles = [
        (30, 0.72, (120, 255, 120)),
        (62, 0.58, (255, 255, 255)),
        (94, 0.58, (255, 220, 120)),
    ]
    for idx, line in enumerate(lines[: len(styles)]):
        if not line:
            continue
        y, scale, color = styles[idx]
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def build_writer(path: Path, width: int, height: int, fps: float, writer_fourcc: str) -> cv2.VideoWriter:
    code = (writer_fourcc or "mp4v").strip().upper()
    if len(code) != 4:
        code = "MP4V"
    fourcc = cv2.VideoWriter_fourcc(*code)
    writer = cv2.VideoWriter(str(path), fourcc, max(1.0, float(fps)), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path} with codec {code}")
    return writer


def main() -> None:
    args = parse_args()
    source = parse_camera_source(args.camera)
    cap = open_capture(source, args.width, args.height, args.fps, backend=args.backend, fourcc=args.fourcc)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera/video source.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC))
    actual_mode = f"{actual_w}x{actual_h} @{actual_fps:.1f} codec={actual_fourcc}"
    print(f"[IdentityRecord] mode={actual_mode}")

    output_path = Path(args.output) if args.output else next_video_path(Path(args.output_dir), args.prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[IdentityRecord] output={output_path.resolve()}")

    window_name = "Ball Identity Recorder"
    if not args.no_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    countdown = max(0, int(args.countdown_seconds))
    countdown_start = time.time()
    writer: cv2.VideoWriter | None = None
    record_start = 0.0
    fps_counter_start = time.time()
    fps_frame_count = 0
    measured_fps = 0.0
    last_terminal_fps_log = 0.0
    interrupted = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera stopped delivering frames while recording.")

            now = time.time()
            fps_frame_count += 1
            elapsed_fps = now - fps_counter_start
            if elapsed_fps >= 0.8:
                measured_fps = fps_frame_count / max(elapsed_fps, 1e-6)
                fps_counter_start = now
                fps_frame_count = 0
            mode_line = f"Requested {args.width}x{args.height}@{args.fps}   Actual {actual_mode}   Measured {measured_fps:04.1f} fps"
            if countdown > 0:
                remaining = countdown - int(now - countdown_start)
                hud = frame.copy()
                draw_status(
                    hud,
                    f"Recording starts in {max(0, remaining)}",
                    "Press q to cancel",
                    mode_line,
                )
                if not args.no_preview:
                    show_frame(window_name, hud, args.display_scale)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        print("[IdentityRecord] cancelled")
                        return
                if now - countdown_start >= countdown:
                    writer = build_writer(output_path, frame.shape[1], frame.shape[0], actual_fps or args.fps, args.writer_fourcc)
                    record_start = time.time()
                    break
                continue
            writer = build_writer(output_path, frame.shape[1], frame.shape[0], actual_fps or args.fps, args.writer_fourcc)
            record_start = time.time()
            break

        duration = max(1, int(args.duration_seconds))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert writer is not None
            writer.write(frame)
            elapsed = time.time() - record_start
            remaining = max(0.0, duration - elapsed)
            hud = frame.copy()
            if args.no_preview and (time.time() - last_terminal_fps_log) >= 1.0:
                print(
                    f"[IdentityRecord] elapsed={elapsed:05.1f}s remaining={remaining:04.1f}s "
                    f"measured_fps={measured_fps:04.1f} actual={actual_mode}"
                )
                last_terminal_fps_log = time.time()
            draw_status(
                hud,
                f"REC {elapsed:05.1f}s / {duration}s",
                f"Remaining: {remaining:04.1f}s   Press q to stop early",
                mode_line,
            )
            if not args.no_preview:
                show_frame(window_name, hud, args.display_scale)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("[IdentityRecord] stopped early by user")
                    break
            if elapsed >= duration:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("[IdentityRecord] interrupted by user")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    print(f"[IdentityRecord] saved={output_path.resolve()}")
    if interrupted:
        print("[IdentityRecord] video was stopped early but the saved file should still be usable.")
    print("[Next] Use that file with:")
    print(
        "python tools/capture_ball_identity_video.py "
        f'--source "{output_path.resolve()}" --output-root datasets/ball_identity --sample-every 8 --padding-scale 1.15 --max-samples 220'
    )


if __name__ == "__main__":
    main()
