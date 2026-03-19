from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms_error: float | None = None
    sample_count: int = 0
    board_spec: dict | None = None


def save_camera_intrinsics(path: Path, intrinsics: CameraIntrinsics) -> None:
    payload = {
        "image_size": [int(intrinsics.image_size[0]), int(intrinsics.image_size[1])],
        "camera_matrix": np.asarray(intrinsics.camera_matrix, dtype=float).tolist(),
        "dist_coeffs": np.asarray(intrinsics.dist_coeffs, dtype=float).reshape(-1).tolist(),
        "rms_error": None if intrinsics.rms_error is None else float(intrinsics.rms_error),
        "sample_count": int(intrinsics.sample_count),
        "board_spec": intrinsics.board_spec or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_camera_intrinsics(path: Path) -> CameraIntrinsics:
    payload = json.loads(path.read_text(encoding="utf-8"))
    image_size = tuple(int(v) for v in payload["image_size"])
    camera_matrix = np.asarray(payload["camera_matrix"], dtype=np.float32).reshape(3, 3)
    dist_coeffs = np.asarray(payload["dist_coeffs"], dtype=np.float32).reshape(1, -1)
    return CameraIntrinsics(
        image_size=(int(image_size[0]), int(image_size[1])),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms_error=None if payload.get("rms_error") is None else float(payload["rms_error"]),
        sample_count=int(payload.get("sample_count", 0)),
        board_spec=payload.get("board_spec") or None,
    )


def scaled_camera_matrix(intrinsics: CameraIntrinsics, frame_size: tuple[int, int]) -> np.ndarray:
    width, height = frame_size
    base_w, base_h = intrinsics.image_size
    sx = float(width) / float(max(base_w, 1))
    sy = float(height) / float(max(base_h, 1))
    scaled = intrinsics.camera_matrix.copy().astype(np.float32)
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def undistort_frame(frame_bgr: np.ndarray, intrinsics: CameraIntrinsics, alpha: float = 0.0) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    camera_matrix = scaled_camera_matrix(intrinsics, (w, h))
    new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        intrinsics.dist_coeffs,
        (w, h),
        float(alpha),
        (w, h),
    )
    return cv2.undistort(frame_bgr, camera_matrix, intrinsics.dist_coeffs, None, new_camera_matrix)
