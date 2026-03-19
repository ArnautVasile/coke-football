from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goal_tracker.app import open_capture, parse_camera_source
from goal_tracker.camera_intrinsics import CameraIntrinsics, save_camera_intrinsics
from goal_tracker.charuco import CharucoSpec, create_board, create_detector, detect_charuco, load_spec, matched_image_object_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture ChArUco views from the live camera and calibrate intrinsics")
    parser.add_argument("--camera", default="0")
    parser.add_argument("--backend", choices=["auto", "avfoundation", "msmf", "dshow"], default="auto")
    parser.add_argument("--fourcc", default="")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--board-spec", default="data/calibration/charuco_board.json")
    parser.add_argument("--output", default="data/calibration/camera_intrinsics.json")
    parser.add_argument("--min-charuco-corners", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--target-samples", type=int, default=24)
    parser.add_argument("--window-name", default="ChArUco Camera Calibration")
    return parser.parse_args()


def calibrate_from_samples(
    board,
    obj_points_list: list[np.ndarray],
    img_points_list: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray]:
    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        objectPoints=obj_points_list,
        imagePoints=img_points_list,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )
    return float(rms), camera_matrix.astype(np.float32), dist_coeffs.astype(np.float32)


def draw_status(
    frame: np.ndarray,
    sample_count: int,
    accepted_corners: int,
    min_samples: int,
    target_samples: int,
    message: str = "",
) -> np.ndarray:
    view = frame.copy()
    lines = [
        "Show the printed ChArUco board from different angles and parts of the image.",
        f"SPACE=capture sample  ENTER=finish ({sample_count}/{min_samples} min, {target_samples} target)  R=reset  Q=quit",
        f"Current visible ChArUco corners: {accepted_corners}",
    ]
    if message:
        lines.append(message)
    y = 28
    for line in lines:
        cv2.putText(view, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
        y += 28
    return view


def main() -> None:
    args = parse_args()
    board_spec_path = Path(args.board_spec)
    if not board_spec_path.exists():
        raise SystemExit(
            f"Board spec not found: {board_spec_path}\n"
            "Generate one first with: python tools/generate_charuco_board.py"
        )

    spec: CharucoSpec = load_spec(board_spec_path)
    board = create_board(spec)
    detector = create_detector(spec)

    cap = open_capture(
        parse_camera_source(args.camera),
        args.width,
        args.height,
        args.fps,
        backend=args.backend,
        fourcc=args.fourcc,
    )
    if not cap.isOpened():
        raise SystemExit("Could not open camera for ChArUco calibration.")

    obj_points_list: list[np.ndarray] = []
    img_points_list: list[np.ndarray] = []
    last_message = ""
    image_size: tuple[int, int] | None = None

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            image_size = (frame.shape[1], frame.shape[0])
            charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco(
                frame,
                detector,
                min_corners=args.min_charuco_corners,
            )

            preview = frame.copy()
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(preview, marker_corners, marker_ids)
            visible_count = 0
            if charuco_ids is not None and charuco_corners is not None:
                visible_count = int(len(charuco_ids))
                cv2.aruco.drawDetectedCornersCharuco(preview, charuco_corners, charuco_ids, (0, 255, 0))

            preview = draw_status(
                preview,
                sample_count=len(obj_points_list),
                accepted_corners=visible_count,
                min_samples=max(4, args.min_samples),
                target_samples=max(args.min_samples, args.target_samples),
                message=last_message,
            )
            cv2.imshow(args.window_name, preview)
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q")):
                raise RuntimeError("Calibration cancelled.")
            if key == ord("r"):
                obj_points_list.clear()
                img_points_list.clear()
                last_message = "Samples reset."
                continue
            if key in (13, 10):
                if len(obj_points_list) < max(4, args.min_samples):
                    last_message = f"Need at least {args.min_samples} accepted samples before calibrating."
                    continue
                break
            if key != ord(" "):
                continue

            if charuco_ids is None or charuco_corners is None or len(charuco_ids) < max(4, args.min_charuco_corners):
                last_message = f"Capture rejected: need at least {args.min_charuco_corners} visible ChArUco corners."
                continue

            obj_points, img_points = matched_image_object_points(board, charuco_corners, charuco_ids)
            if len(obj_points) < max(4, args.min_charuco_corners):
                last_message = "Capture rejected: not enough matched object/image points."
                continue

            obj_points_list.append(obj_points.astype(np.float32))
            img_points_list.append(img_points.astype(np.float32))
            last_message = f"Accepted sample #{len(obj_points_list)} with {len(obj_points)} matched corners."

            if len(obj_points_list) >= max(args.min_samples, args.target_samples):
                break

        if image_size is None:
            raise RuntimeError("No camera frames received.")

        rms, camera_matrix, dist_coeffs = calibrate_from_samples(board, obj_points_list, img_points_list, image_size)
        output_path = Path(args.output)
        save_camera_intrinsics(
            output_path,
            CameraIntrinsics(
                image_size=image_size,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                rms_error=rms,
                sample_count=len(obj_points_list),
                board_spec=asdict(spec),
            ),
        )
        print(f"[ChArUco] samples={len(obj_points_list)}")
        print(f"[ChArUco] rms_error={rms:.4f}")
        print(f"[ChArUco] output={output_path.resolve()}")
        print("[Next] You can now load these intrinsics in the tracker for future undistortion/pose work.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
