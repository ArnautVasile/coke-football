from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CharucoSpec:
    dictionary_name: str = "DICT_6X6_50"
    squares_x: int = 7
    squares_y: int = 5
    square_length_mm: float = 30.0
    marker_length_mm: float = 22.0
    legacy_pattern: bool = False

    @property
    def square_length_m(self) -> float:
        return float(self.square_length_mm) / 1000.0

    @property
    def marker_length_m(self) -> float:
        return float(self.marker_length_mm) / 1000.0

    @property
    def board_size_mm(self) -> tuple[float, float]:
        return (
            float(self.squares_x) * float(self.square_length_mm),
            float(self.squares_y) * float(self.square_length_mm),
        )


def _dictionary_id(name: str) -> int:
    dict_id = getattr(cv2.aruco, name, None)
    if dict_id is None:
        known = sorted(n for n in dir(cv2.aruco) if n.startswith("DICT_"))
        raise RuntimeError(f"Unknown ArUco dictionary '{name}'. Known values include: {', '.join(known[:12])}")
    return int(dict_id)


def create_dictionary(name: str):
    return cv2.aruco.getPredefinedDictionary(_dictionary_id(name))


def create_board(spec: CharucoSpec):
    board = cv2.aruco.CharucoBoard(
        (int(spec.squares_x), int(spec.squares_y)),
        float(spec.square_length_m),
        float(spec.marker_length_m),
        create_dictionary(spec.dictionary_name),
    )
    if spec.legacy_pattern and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    return board


def create_detector(spec: CharucoSpec):
    return cv2.aruco.CharucoDetector(create_board(spec))


def generate_board_image(spec: CharucoSpec, pixels_per_mm: float = 12.0, margin_mm: float = 10.0) -> np.ndarray:
    board = create_board(spec)
    board_w_mm, board_h_mm = spec.board_size_mm
    image_size = (
        max(32, int(round(board_w_mm * float(pixels_per_mm)))),
        max(32, int(round(board_h_mm * float(pixels_per_mm)))),
    )
    margin_px = max(8, int(round(float(margin_mm) * float(pixels_per_mm))))
    return board.generateImage(image_size, marginSize=margin_px)


def detect_charuco(frame_bgr: np.ndarray, detector, min_corners: int = 4) -> tuple[np.ndarray | None, np.ndarray | None, tuple, np.ndarray | None]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if charuco_ids is None or charuco_corners is None or len(charuco_ids) < max(1, int(min_corners)):
        return None, None, marker_corners, marker_ids
    return charuco_corners, charuco_ids, marker_corners, marker_ids


def matched_image_object_points(board, charuco_corners: np.ndarray, charuco_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    obj = np.asarray(obj_points, dtype=np.float32).reshape(-1, 3)
    img = np.asarray(img_points, dtype=np.float32).reshape(-1, 2)
    return obj, img


def save_spec(path: Path, spec: CharucoSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")


def load_spec(path: Path) -> CharucoSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CharucoSpec(
        dictionary_name=str(payload.get("dictionary_name", "DICT_6X6_50")),
        squares_x=int(payload.get("squares_x", 7)),
        squares_y=int(payload.get("squares_y", 5)),
        square_length_mm=float(payload.get("square_length_mm", 30.0)),
        marker_length_mm=float(payload.get("marker_length_mm", 22.0)),
        legacy_pattern=bool(payload.get("legacy_pattern", False)),
    )
