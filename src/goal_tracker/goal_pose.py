from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .camera_intrinsics import CameraIntrinsics, scaled_camera_matrix
from .charuco import create_dictionary
from .goal_markers import GoalMarkerLayout


@dataclass
class GoalPoseEstimate:
    rvec: np.ndarray
    tvec: np.ndarray
    visible_ids: list[int]
    goal_corners_px: np.ndarray
    goal_object_corners: np.ndarray
    scoring_plane_depth_m: float
    reprojection_error_px: float


@dataclass
class GoalPoseDebug:
    status: str


@dataclass
class BallPlaneEstimate:
    center_distance_m: float
    signed_distance_m: float
    absolute_distance_m: float
    surface_distance_m: float
    signed_surface_distance_m: float
    camera_signed_distance_m: float


def marker_object_corners(layout: GoalMarkerLayout, marker_id: int) -> np.ndarray:
    marker = next((m for m in layout.markers if m.marker_id == int(marker_id)), None)
    if marker is None:
        raise KeyError(f"Marker id {marker_id} is not present in the goal marker layout.")
    half = layout.marker_length_m / 2.0
    cx, cy, cz = marker.center_m
    return np.asarray(
        [
            [cx - half, cy - half, cz],
            [cx + half, cy - half, cz],
            [cx + half, cy + half, cz],
            [cx - half, cy + half, cz],
        ],
        dtype=np.float32,
    )


def create_goal_board(layout: GoalMarkerLayout):
    obj_points = [marker_object_corners(layout, marker.marker_id) for marker in layout.markers]
    ids = np.asarray([marker.marker_id for marker in layout.markers], dtype=np.int32)
    dictionary = create_dictionary(layout.dictionary_name)
    return cv2.aruco.Board(obj_points, dictionary, ids)


def create_marker_detector():
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 43
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.015
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.minCornerDistanceRate = 0.03
    return params


def detect_goal_markers(frame_bgr: np.ndarray, layout: GoalMarkerLayout, intrinsics: CameraIntrinsics):
    dictionary = create_dictionary(layout.dictionary_name)
    params = create_marker_detector()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    marker_corners, marker_ids, _rejected = detector.detectMarkers(frame_bgr)
    if marker_ids is None or len(marker_ids) == 0:
        return [], None

    h, w = frame_bgr.shape[:2]
    camera_matrix = scaled_camera_matrix(intrinsics, (w, h))
    board = create_goal_board(layout)
    raw_corners = list(marker_corners)
    raw_ids = marker_ids.copy()
    try:
        marker_corners, marker_ids, _rejected, _recovered = cv2.aruco.refineDetectedMarkers(
            frame_bgr,
            board,
            marker_corners,
            marker_ids,
            _rejected,
            cameraMatrix=camera_matrix,
            distCoeffs=intrinsics.dist_coeffs,
            parameters=params,
        )
    except cv2.error:
        pass
    if marker_ids is None or len(marker_ids) == 0:
        marker_corners = raw_corners
        marker_ids = raw_ids

    allowed = {marker.marker_id for marker in layout.markers}
    filtered_corners: list[np.ndarray] = []
    filtered_ids: list[int] = []
    for corners, marker_id in zip(marker_corners, marker_ids.reshape(-1)):
        if int(marker_id) not in allowed:
            continue
        filtered_corners.append(np.asarray(corners, dtype=np.float32))
        filtered_ids.append(int(marker_id))
    if not filtered_ids:
        return [], None
    return filtered_corners, np.asarray(filtered_ids, dtype=np.int32).reshape(-1, 1)


def goal_plane_object_corners(layout: GoalMarkerLayout) -> np.ndarray:
    x1 = max(0.0, float(layout.opening_inset_left_m))
    y1 = max(0.0, float(layout.opening_inset_top_m))
    x2 = max(x1 + 0.01, float(layout.goal_width_m) - max(0.0, float(layout.opening_inset_right_m)))
    y2 = max(y1 + 0.01, float(layout.goal_height_m) - max(0.0, float(layout.opening_inset_bottom_m)))
    return np.asarray(
        [
            [x1, y1, 0.0],
            [x2, y1, 0.0],
            [x2, y2, 0.0],
            [x1, y2, 0.0],
        ],
        dtype=np.float32,
    )


def scoring_plane_camera_geometry(
    pose: GoalPoseEstimate,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float] | None:
    rotation, _ = cv2.Rodrigues(pose.rvec)
    plane_corners_obj = np.asarray(pose.goal_object_corners, dtype=np.float64).reshape(-1, 3)
    if plane_corners_obj.shape[0] < 4:
        return None

    plane_origin_obj = plane_corners_obj[0]
    x_axis_obj = plane_corners_obj[1] - plane_origin_obj
    y_axis_obj = plane_corners_obj[3] - plane_origin_obj
    width_m = float(np.linalg.norm(x_axis_obj))
    height_m = float(np.linalg.norm(y_axis_obj))
    if width_m <= 1e-6 or height_m <= 1e-6:
        return None

    x_axis_obj /= width_m
    y_axis_obj /= height_m
    plane_origin_cam = (rotation @ plane_origin_obj.reshape(3, 1) + pose.tvec.reshape(3, 1)).reshape(3)
    x_axis_cam = (rotation @ x_axis_obj.reshape(3, 1)).reshape(3)
    y_axis_cam = (rotation @ y_axis_obj.reshape(3, 1)).reshape(3)
    plane_normal_cam = np.cross(x_axis_cam, y_axis_cam)
    normal_norm = float(np.linalg.norm(plane_normal_cam))
    if normal_norm <= 1e-9:
        return None
    plane_normal_cam /= normal_norm

    raw_camera_signed_distance = float(np.dot(plane_normal_cam, -plane_origin_cam))
    if abs(raw_camera_signed_distance) > 1e-9 and pose.scoring_plane_depth_m > 0.0:
        through_goal_dir = -math.copysign(1.0, raw_camera_signed_distance) * plane_normal_cam
        plane_origin_cam = plane_origin_cam + through_goal_dir * float(pose.scoring_plane_depth_m)

    camera_signed_distance = float(np.dot(plane_normal_cam, -plane_origin_cam))
    return (
        plane_origin_cam,
        x_axis_cam,
        y_axis_cam,
        plane_normal_cam,
        width_m,
        height_m,
        camera_signed_distance,
    )


def solve_goal_pose(
    frame_bgr: np.ndarray,
    layout: GoalMarkerLayout,
    intrinsics: CameraIntrinsics,
) -> tuple[GoalPoseEstimate | None, list[np.ndarray], np.ndarray | None, GoalPoseDebug]:
    marker_corners, marker_ids = detect_goal_markers(frame_bgr, layout, intrinsics)
    if marker_ids is None or len(marker_ids) < 3:
        seen = 0 if marker_ids is None else int(len(marker_ids))
        return None, marker_corners, marker_ids, GoalPoseDebug(
            status=f"Need at least 3 markers, currently seeing {seen}.",
        )

    visible_markers = [marker for marker in layout.markers if int(marker.marker_id) in set(int(v) for v in marker_ids.reshape(-1))]
    xs = [float(marker.center_m[0]) for marker in visible_markers]
    ys = [float(marker.center_m[1]) for marker in visible_markers]
    if not xs or not ys:
        return None, marker_corners, marker_ids, GoalPoseDebug(
            status="Marker IDs were detected, but none matched the active layout.",
        )
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    if x_span < (0.35 * layout.goal_width_m) and y_span < (0.35 * layout.goal_height_m):
        return None, marker_corners, marker_ids, GoalPoseDebug(
            status=(
                f"Markers are too clustered for a stable pose "
                f"(x span {x_span:.2f}m, y span {y_span:.2f}m)."
            ),
        )

    board = create_goal_board(layout)
    obj_points, img_points = board.matchImagePoints(marker_corners, marker_ids)
    obj_points = np.asarray(obj_points, dtype=np.float32).reshape(-1, 3)
    img_points = np.asarray(img_points, dtype=np.float32).reshape(-1, 2)
    if len(obj_points) < 8:
        return None, marker_corners, marker_ids, GoalPoseDebug(
            status=f"Only {len(obj_points)} matched corner points; need at least 8.",
        )

    h, w = frame_bgr.shape[:2]
    camera_matrix = scaled_camera_matrix(intrinsics, (w, h))
    flags = cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=obj_points,
        imagePoints=img_points,
        cameraMatrix=camera_matrix,
        distCoeffs=intrinsics.dist_coeffs,
        flags=flags,
    )
    if not ok:
        return None, marker_corners, marker_ids, GoalPoseDebug(
            status="solvePnP failed to find a stable pose from the current marker geometry.",
        )

    if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return None, marker_corners, marker_ids, GoalPoseDebug(
            status="solvePnP returned non-finite rotation or translation values.",
        )

    rotation, _ = cv2.Rodrigues(rvec)
    goal_object_corners = goal_plane_object_corners(layout)
    goal_camera = (rotation @ goal_object_corners.T + tvec.reshape(3, 1)).T
    if not np.isfinite(goal_camera).all():
        return None, marker_corners, marker_ids, GoalPoseDebug(
            status="Projected goal corners became non-finite after pose solve.",
        )
    # Reject poses that place the goal behind the camera or almost on the camera center.
    if float(np.min(goal_camera[:, 2])) <= 0.05:
        return None, marker_corners, marker_ids, GoalPoseDebug(
            status=f"Pose places the goal behind or too close to the camera (min z {float(np.min(goal_camera[:, 2])):.3f}m).",
        )

    reprojected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, intrinsics.dist_coeffs)
    reproj_pts = reprojected.reshape(-1, 2)
    if not np.isfinite(reproj_pts).all():
        return None, marker_corners, marker_ids, GoalPoseDebug(
            status="Marker reprojection became non-finite after pose solve.",
        )
    reprojection_error = float(
        np.mean(np.linalg.norm(reproj_pts - img_points.reshape(-1, 2), axis=1))
    )
    if not np.isfinite(reprojection_error) or reprojection_error > 15.0:
        return None, marker_corners, marker_ids, GoalPoseDebug(
            status=(
                f"Reprojection error is too high ({reprojection_error:.2f}px). "
                "This usually means the marker layout does not match the real frame geometry."
            ),
        )

    goal_corners_2d, _ = cv2.projectPoints(
        goal_object_corners,
        rvec,
        tvec,
        camera_matrix,
        intrinsics.dist_coeffs,
    )
    goal_corners_px = goal_corners_2d.reshape(-1, 2).astype(np.float32)
    if not np.isfinite(goal_corners_px).all():
        return None, marker_corners, marker_ids, GoalPoseDebug(
            status="Goal corner projection became non-finite after a valid pose solve.",
        )
    estimate = GoalPoseEstimate(
        rvec=rvec.astype(np.float32),
        tvec=tvec.astype(np.float32),
        visible_ids=[int(v) for v in marker_ids.reshape(-1).tolist()],
        goal_corners_px=goal_corners_px,
        goal_object_corners=goal_object_corners.astype(np.float32),
        scoring_plane_depth_m=float(layout.scoring_plane_depth_m),
        reprojection_error_px=reprojection_error,
    )
    return estimate, marker_corners, marker_ids, GoalPoseDebug(
        status=f"Pose OK (reprojection error {reprojection_error:.2f}px).",
    )


def estimate_ball_plane_distance(
    center_px: tuple[int, int],
    ball_radius_px: float,
    ball_radius_m: float,
    frame_size: tuple[int, int],
    pose: GoalPoseEstimate,
    intrinsics: CameraIntrinsics,
) -> BallPlaneEstimate | None:
    if ball_radius_px <= 1.0 or ball_radius_m <= 0.0:
        return None

    camera_matrix = scaled_camera_matrix(intrinsics, frame_size)
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    if fx <= 1e-6 or fy <= 1e-6:
        return None

    inv_camera = np.linalg.inv(camera_matrix.astype(np.float64))
    pixel_h = np.array([float(center_px[0]), float(center_px[1]), 1.0], dtype=np.float64)
    ray = inv_camera @ pixel_h
    ray_norm = float(np.linalg.norm(ray))
    if ray_norm <= 1e-9:
        return None
    ray /= ray_norm

    angular_radius = 0.5 * (
        math.atan(float(ball_radius_px) / fx) + math.atan(float(ball_radius_px) / fy)
    )
    sin_alpha = math.sin(angular_radius)
    if sin_alpha <= 1e-6:
        return None

    center_distance_m = float(ball_radius_m / sin_alpha)
    center_camera = ray * center_distance_m

    scoring_geometry = scoring_plane_camera_geometry(pose)
    if scoring_geometry is None:
        return None
    plane_point, _x_axis_cam, _y_axis_cam, plane_normal, _width_m, _height_m, camera_signed_distance_m = scoring_geometry
    signed_distance_m = float(np.dot(plane_normal, center_camera - plane_point))
    signed_surface_distance_m = math.copysign(
        max(0.0, abs(signed_distance_m) - float(ball_radius_m)),
        signed_distance_m,
    )

    return BallPlaneEstimate(
        center_distance_m=center_distance_m,
        signed_distance_m=signed_distance_m,
        absolute_distance_m=abs(signed_distance_m),
        surface_distance_m=abs(signed_surface_distance_m),
        signed_surface_distance_m=signed_surface_distance_m,
        camera_signed_distance_m=camera_signed_distance_m,
    )


def project_pixel_to_goal_plane(
    point_px: tuple[int, int] | tuple[float, float],
    frame_size: tuple[int, int],
    pose: GoalPoseEstimate,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float] | None:
    camera_matrix = scaled_camera_matrix(intrinsics, frame_size)
    if camera_matrix.shape != (3, 3):
        return None

    scoring_geometry = scoring_plane_camera_geometry(pose)
    if scoring_geometry is None:
        return None
    plane_origin_cam, x_axis_cam, y_axis_cam, plane_normal_cam, width_m, height_m, _camera_signed_distance = scoring_geometry

    inv_camera = np.linalg.inv(camera_matrix.astype(np.float64))
    pixel_h = np.array([float(point_px[0]), float(point_px[1]), 1.0], dtype=np.float64)
    ray = inv_camera @ pixel_h
    ray_norm = float(np.linalg.norm(ray))
    if ray_norm <= 1e-9:
        return None
    ray /= ray_norm

    denom = float(np.dot(plane_normal_cam, ray))
    if abs(denom) <= 1e-9:
        return None
    t = float(np.dot(plane_normal_cam, plane_origin_cam) / denom)
    if t <= 0.0:
        return None

    hit_cam = ray * t
    rel = hit_cam - plane_origin_cam
    nx = float(np.dot(rel, x_axis_cam) / width_m)
    ny = float(np.dot(rel, y_axis_cam) / height_m)
    return nx, ny
