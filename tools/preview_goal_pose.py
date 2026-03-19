from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goal_tracker.app import open_capture, parse_camera_source
from goal_tracker.camera_intrinsics import load_camera_intrinsics, undistort_frame
from goal_tracker.goal_markers import load_goal_marker_layout
from goal_tracker.goal_pose import solve_goal_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview goal-frame marker detection and 3D pose from the live camera")
    parser.add_argument("--camera", default="0")
    parser.add_argument("--backend", choices=["auto", "avfoundation", "msmf", "dshow"], default="auto")
    parser.add_argument("--fourcc", default="")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--camera-calibration-file", default="data/calibration/camera_intrinsics.json")
    parser.add_argument("--goal-markers-layout", default="data/calibration/goal_markers_layout.json")
    parser.add_argument("--window-name", default="Goal Marker Pose Preview")
    parser.add_argument("--undistort-display", action="store_true", help="Undistort the preview frames before marker detection")
    return parser.parse_args()


def draw_text(view: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int] = (255, 255, 255), scale: float = 0.7) -> None:
    cv2.putText(view, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(view, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    intrinsics = load_camera_intrinsics(Path(args.camera_calibration_file))
    layout = load_goal_marker_layout(Path(args.goal_markers_layout))

    cap = open_capture(
        parse_camera_source(args.camera),
        args.width,
        args.height,
        args.fps,
        backend=args.backend,
        fourcc=args.fourcc,
    )
    if not cap.isOpened():
        raise SystemExit("Could not open camera for goal marker pose preview.")

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if args.undistort_display:
                frame = undistort_frame(frame, intrinsics)

            estimate, marker_corners, marker_ids, debug = solve_goal_pose(frame, layout, intrinsics)
            view = frame.copy()

            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(view, marker_corners, marker_ids)

            if estimate is not None and np.isfinite(estimate.goal_corners_px).all():
                polygon = estimate.goal_corners_px.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(view, [polygon], True, (0, 255, 0), 3)
                for idx, pt in enumerate(estimate.goal_corners_px):
                    if not np.isfinite(pt).all():
                        continue
                    center = tuple(int(v) for v in np.round(pt).astype(int).tolist())
                    cv2.circle(view, center, 7, (0, 255, 255), -1)
                    draw_text(view, str(idx + 1), (int(pt[0]) + 10, int(pt[1]) - 10), color=(0, 255, 255), scale=0.75)
                draw_text(view, f"Visible markers: {', '.join(str(v) for v in estimate.visible_ids)}", (20, 34), color=(0, 255, 0))
                draw_text(view, f"Reprojection error: {estimate.reprojection_error_px:.2f}px", (20, 68), color=(0, 255, 0))
                draw_text(view, "Pose OK", (20, 102), color=(0, 255, 0))
            else:
                visible = [] if marker_ids is None else [str(int(v)) for v in marker_ids.reshape(-1).tolist()]
                draw_text(view, f"Visible markers: {', '.join(visible) if visible else 'none'}", (20, 34), color=(0, 200, 255))
                draw_text(view, debug.status, (20, 68), color=(0, 200, 255))
                draw_text(view, "Pose not solved yet", (20, 102), color=(0, 200, 255))

            draw_text(view, "q/Esc quit", (20, view.shape[0] - 20), color=(255, 255, 255), scale=0.65)
            cv2.imshow(args.window_name, view)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
