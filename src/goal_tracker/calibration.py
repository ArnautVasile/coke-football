from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np


@dataclass
class CalibrationData:
    corners_px: np.ndarray
    reference_size: tuple[int, int]
    goal_width_m: float = 7.32
    goal_height_m: float = 2.44
    reference_frame_path: str | None = None


def reorder_clockwise(points: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start_idx = np.argmin(ordered[:, 0] + ordered[:, 1])
    ordered = np.roll(ordered, -int(start_idx), axis=0)
    return ordered.astype(np.float32)


def calibrate_goal_corners(frame: np.ndarray, window_name: str = "Goal Calibration") -> np.ndarray:
    clone = frame.copy()
    selected: List[tuple[int, int]] = []

    help_text = "Click 4 goal corners (any order). Enter=save, Backspace=undo, Esc=cancel"

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(selected) < 4:
            selected.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        view = clone.copy()
        cv2.putText(view, help_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(
            view,
            f"Points: {len(selected)}/4",
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        for idx, pt in enumerate(selected):
            cv2.circle(view, pt, 6, (0, 255, 0), -1)
            cv2.putText(
                view,
                str(idx + 1),
                (pt[0] + 8, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        if len(selected) >= 2:
            contour = np.array(selected, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(view, [contour], False, (0, 180, 255), 2)
        if len(selected) == 4:
            contour = np.array(selected, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(view, [contour], True, (0, 255, 0), 2)

        cv2.imshow(window_name, view)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Calibration cancelled.")
        if key in (8, 127) and selected:
            selected.pop()
        if key in (13, 10) and len(selected) == 4:
            break

    cv2.destroyWindow(window_name)
    return reorder_clockwise(selected)


def save_calibration(path: Path, data: CalibrationData) -> None:
    payload = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "corners_px": data.corners_px.tolist(),
        "reference_size": list(data.reference_size),
        "goal_width_m": data.goal_width_m,
        "goal_height_m": data.goal_height_m,
        "reference_frame_path": data.reference_frame_path,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_calibration(path: Path) -> CalibrationData:
    payload = json.loads(path.read_text())
    corners = np.asarray(payload["corners_px"], dtype=np.float32)
    ref_w, ref_h = payload.get("reference_size", [1920, 1080])

    return CalibrationData(
        corners_px=corners,
        reference_size=(int(ref_w), int(ref_h)),
        goal_width_m=float(payload.get("goal_width_m", 7.32)),
        goal_height_m=float(payload.get("goal_height_m", 2.44)),
        reference_frame_path=payload.get("reference_frame_path"),
    )


def scale_corners(corners: np.ndarray, from_size: tuple[int, int], to_size: tuple[int, int]) -> np.ndarray:
    from_w, from_h = from_size
    to_w, to_h = to_size
    sx = to_w / float(max(from_w, 1))
    sy = to_h / float(max(from_h, 1))
    scaled = corners.copy()
    scaled[:, 0] *= sx
    scaled[:, 1] *= sy
    return scaled

