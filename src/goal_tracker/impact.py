from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque

import cv2
import numpy as np

from .camera_intrinsics import CameraIntrinsics
from .goal_pose import GoalPoseEstimate, project_pixel_to_goal_plane


@dataclass
class ImpactEvent:
    timestamp: float
    frame_index: int
    event_type: str
    pixel_point: tuple[int, int]
    normalized_point: tuple[float, float]
    meters_point: tuple[float, float]
    speed_before: float
    speed_after: float
    angle_change_deg: float


def build_goal_homography(corners: np.ndarray) -> np.ndarray:
    dst = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(corners.astype(np.float32), dst)


def project_to_goal(point_xy: tuple[int, int], homography: np.ndarray) -> tuple[float, float]:
    src = np.array([[point_xy]], dtype=np.float32)
    projected = cv2.perspectiveTransform(src, homography)[0, 0]
    return float(projected[0]), float(projected[1])


def point_inside_polygon(point_xy: tuple[int, int], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon.astype(np.float32), point_xy, False) >= 0


def signed_distance_to_polygon(point_xy: tuple[float, float], polygon: np.ndarray) -> float:
    return float(cv2.pointPolygonTest(polygon.astype(np.float32), point_xy, True))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_angle = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def segment_intersection(
    a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray
) -> np.ndarray | None:
    x1, y1 = float(a1[0]), float(a1[1])
    x2, y2 = float(a2[0]), float(a2[1])
    x3, y3 = float(b1[0]), float(b1[1])
    x4, y4 = float(b2[0]), float(b2[1])

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
    if not (0.0 <= t <= 1.0 and 0.0 <= u <= 1.0):
        return None

    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dtype=np.float32)


def closest_point_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        return a.astype(np.float32)
    t = float(np.dot(point - a, ab) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    return (a + t * ab).astype(np.float32)


def closest_point_on_polygon(point: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    poly = polygon.astype(np.float32)
    best = poly[0].astype(np.float32)
    best_dist = float("inf")
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        c = closest_point_on_segment(point, a, b)
        d = float(np.linalg.norm(point - c))
        if d < best_dist:
            best = c
            best_dist = d
    return best


def find_entry_point(prev_xy: np.ndarray, cur_xy: np.ndarray, polygon: np.ndarray) -> np.ndarray | None:
    poly = polygon.astype(np.float32)
    best: np.ndarray | None = None
    best_dist = float("inf")
    n = len(poly)
    for i in range(n):
        q1 = poly[i]
        q2 = poly[(i + 1) % n]
        inter = segment_intersection(prev_xy, cur_xy, q1, q2)
        if inter is None:
            continue
        d = float(np.linalg.norm(inter - prev_xy))
        if d < best_dist:
            best = inter
            best_dist = d
    return best


class ImpactDetector:
    def __init__(
        self,
        goal_width_m: float,
        goal_height_m: float,
        min_pre_impact_speed: float = 450.0,
        speed_drop_ratio: float = 0.45,
        min_direction_change_deg: float = 55.0,
        cooldown_s: float = 0.5,
        max_dt_s: float = 0.12,
        min_displacement_px: float = 90.0,
        enable_entry_event: bool = True,
        min_entry_speed_px_s: float = 35.0,
        entry_confirm_frames: int = 1,
        allow_entry_fallbacks: bool = True,
        rearm_outside_ratio: float = 0.60,
        rearm_camera_margin_m: float = 0.14,
        rearm_miss_seconds: float = 0.75,
    ) -> None:
        self.goal_width_m = goal_width_m
        self.goal_height_m = goal_height_m
        self.min_pre_impact_speed = min_pre_impact_speed
        self.speed_drop_ratio = speed_drop_ratio
        self.min_direction_change_deg = min_direction_change_deg
        self.cooldown_s = cooldown_s
        self.max_dt_s = max(0.02, float(max_dt_s))
        self.min_displacement_px = max(8.0, float(min_displacement_px))
        self.enable_entry_event = bool(enable_entry_event)
        self.min_entry_speed_px_s = max(0.0, float(min_entry_speed_px_s))
        self.entry_confirm_frames = max(1, int(entry_confirm_frames))
        self.allow_entry_fallbacks = bool(allow_entry_fallbacks)
        self.rearm_outside_ratio = max(0.10, float(rearm_outside_ratio))
        self.rearm_camera_margin_m = max(0.01, float(rearm_camera_margin_m))
        self.rearm_miss_seconds = max(0.0, float(rearm_miss_seconds))
        self.first_inside_edge_px = 30.0
        self.first_inside_inward_px = 2.0

        self.history: Deque[tuple[float, np.ndarray, float | None]] = deque(maxlen=8)
        self.last_event_time = 0.0
        self.event_latched = False
        self.last_seen_time = 0.0
        # Dropout-resilient entry fallback: remember when the ball was clearly
        # on the camera side of the plane so we can still score one entry if
        # the exact crossing frame is missed.
        self.last_camera_side_time = -1e9
        self.last_camera_side_center: np.ndarray | None = None
        self.pending_entry_active = False
        self.pending_entry_count = 0
        self.pending_entry_point_xy: tuple[int, int] | None = None
        self.pending_entry_timestamp = 0.0
        self.pending_entry_frame_index = 0
        self.pending_entry_speed = 0.0
        self.pending_entry_last_seen_time = 0.0
        self.pending_entry_missing_grace_s = max(0.08, float(self.max_dt_s) * 2.0)

    def reset_history(self) -> None:
        self.history.clear()
        self.event_latched = False
        self.last_seen_time = 0.0
        self.last_camera_side_time = -1e9
        self.last_camera_side_center = None
        self._clear_pending_entry()

    def _clear_pending_entry(self) -> None:
        self.pending_entry_active = False
        self.pending_entry_count = 0
        self.pending_entry_point_xy = None
        self.pending_entry_timestamp = 0.0
        self.pending_entry_frame_index = 0
        self.pending_entry_speed = 0.0
        self.pending_entry_last_seen_time = 0.0

    def _release_latch_if_ready(
        self,
        *,
        now_s: float,
        center: np.ndarray | None,
        ball_radius_px: float | None,
        goal_corners: np.ndarray,
        plane_signed_distance_m: float | None,
        camera_signed_distance_m: float | None,
        ball_radius_m: float | None,
        plane_contact_tolerance_m: float,
    ) -> None:
        if not self.event_latched:
            return

        if center is None:
            if self.rearm_miss_seconds > 0.0 and (now_s - self.last_seen_time) >= self.rearm_miss_seconds:
                self.event_latched = False
                self.history.clear()
            return

        radius_margin_px = max(0.0, float(ball_radius_px or 0.0))
        outside_release_px = max(8.0, radius_margin_px * self.rearm_outside_ratio)
        signed_cur = signed_distance_to_polygon((float(center[0]), float(center[1])), goal_corners)
        if signed_cur < -outside_release_px:
            self.event_latched = False
            self.history.clear()
            return

        if (
            plane_signed_distance_m is not None
            and camera_signed_distance_m is not None
        ):
            plane_release_margin = max(
                self.rearm_camera_margin_m,
                float(ball_radius_m or 0.0) + max(0.0, float(plane_contact_tolerance_m)),
            )
            same_as_camera_side = float(plane_signed_distance_m) * float(camera_signed_distance_m) >= 0.0
            if same_as_camera_side and abs(float(plane_signed_distance_m)) >= plane_release_margin:
                self.event_latched = False
                self.history.clear()

    @staticmethod
    def _plane_contact_ok(
        prev_signed_distance_m: float | None,
        cur_signed_distance_m: float | None,
        ball_radius_m: float | None,
        tolerance_m: float,
    ) -> bool:
        if prev_signed_distance_m is None or cur_signed_distance_m is None or ball_radius_m is None:
            return True
        margin = max(0.01, (2.0 * float(ball_radius_m)) + max(0.0, float(tolerance_m)))
        if abs(prev_signed_distance_m) <= margin or abs(cur_signed_distance_m) <= margin:
            return True
        return (prev_signed_distance_m * cur_signed_distance_m) < 0.0

    def update(
        self,
        center_px: tuple[int, int] | None,
        ball_radius_px: float | None,
        plane_signed_distance_m: float | None,
        camera_signed_distance_m: float | None,
        ball_radius_m: float | None,
        plane_contact_tolerance_m: float,
        now_s: float,
        frame_index: int,
        goal_corners: np.ndarray,
        goal_homography: np.ndarray,
        pose: GoalPoseEstimate | None = None,
        intrinsics: CameraIntrinsics | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> ImpactEvent | None:
        if center_px is None:
            self._release_latch_if_ready(
                now_s=now_s,
                center=None,
                ball_radius_px=ball_radius_px,
                goal_corners=goal_corners,
                plane_signed_distance_m=plane_signed_distance_m,
                camera_signed_distance_m=camera_signed_distance_m,
                ball_radius_m=ball_radius_m,
                plane_contact_tolerance_m=plane_contact_tolerance_m,
            )
            if self.pending_entry_active:
                if (now_s - float(self.pending_entry_last_seen_time)) > float(self.pending_entry_missing_grace_s):
                    self._clear_pending_entry()
            return None

        center = np.asarray(center_px, dtype=np.float32)
        self.last_seen_time = now_s
        self._release_latch_if_ready(
            now_s=now_s,
            center=center,
            ball_radius_px=ball_radius_px,
            goal_corners=goal_corners,
            plane_signed_distance_m=plane_signed_distance_m,
            camera_signed_distance_m=camera_signed_distance_m,
            ball_radius_m=ball_radius_m,
            plane_contact_tolerance_m=plane_contact_tolerance_m,
        )
        if self.event_latched:
            self.history.append((now_s, center, plane_signed_distance_m))
            return None

        self.history.append((now_s, center, plane_signed_distance_m))
        if len(self.history) < 2:
            return None

        t1, p1, d1 = self.history[-2]
        t2, p2, d2 = self.history[-1]
        if (t2 - t1) > self.max_dt_s:
            return None
        d0 = self.history[-3][2] if len(self.history) >= 3 else None

        dt1 = max(1e-4, t2 - t1)
        entry_speed = float(np.linalg.norm(p2 - p1) / dt1)
        # Approximate circle-vs-polygon overlap using the ball radius instead
        # of only the center point. The previous 0.35*r and 60px cap could miss
        # real close-range entries when the ball edge crossed first.
        radius_margin = max(0.0, float(ball_radius_px or 0.0))
        signed_prev = signed_distance_to_polygon((float(p1[0]), float(p1[1])), goal_corners)
        signed_cur = signed_distance_to_polygon((float(p2[0]), float(p2[1])), goal_corners)
        prev_inside = signed_prev >= -radius_margin
        cur_inside = signed_cur >= -radius_margin

        camera_side_sign: float | None = None
        oriented_d2: float | None = None
        if d2 is not None and camera_signed_distance_m is not None:
            camera_side_sign = 1.0 if float(camera_signed_distance_m) >= 0.0 else -1.0
            oriented_d2 = float(d2) * camera_side_sign
            if ball_radius_m is not None:
                camera_side_margin = max(0.01, float(ball_radius_m) * 0.20)
                if oriented_d2 >= camera_side_margin:
                    self.last_camera_side_time = now_s
                    self.last_camera_side_center = center.copy()

        if now_s - self.last_event_time < self.cooldown_s:
            return None

        def project_event_point(point_xy: tuple[int, int]) -> tuple[float, float]:
            if pose is not None and intrinsics is not None and frame_size is not None:
                projected = project_pixel_to_goal_plane(point_xy, frame_size, pose, intrinsics)
                if projected is not None:
                    return projected
            return project_to_goal(point_xy, goal_homography)

        def entry_display_point(default_xy: tuple[int, int]) -> tuple[int, int]:
            # For doorway goal-entry validation, operators expect the map/crosshair
            # to follow the ball center going through the opening, not the first
            # 2D edge intersection with the polygon. When the center is already
            # near the plane, using the center is much more intuitive and avoids
            # large apparent left/right offsets on the hit map.
            if center_px is None:
                return default_xy
            if plane_signed_distance_m is not None and ball_radius_m is not None:
                plane_margin = max(0.02, float(ball_radius_m) + max(0.0, float(plane_contact_tolerance_m)))
                if abs(float(plane_signed_distance_m)) <= plane_margin:
                    return center_px
            # If we're already deep through the plane (rebound/late recovery),
            # keep the geometric crossing/default point to avoid plotting hits
            # at the later rebound location.
            return default_xy

        def emit_entry_event(
            *,
            event_point_xy: tuple[int, int],
            event_timestamp_s: float,
            event_frame_index: int,
            event_speed: float,
        ) -> ImpactEvent:
            nx, ny = project_event_point(event_point_xy)
            nx = float(np.clip(nx, 0.0, 1.0))
            ny = float(np.clip(ny, 0.0, 1.0))
            mx = nx * self.goal_width_m
            my = ny * self.goal_height_m
            self.last_event_time = now_s
            self.event_latched = True
            self.history.clear()
            self.last_camera_side_time = -1e9
            self.last_camera_side_center = None
            self._clear_pending_entry()
            return ImpactEvent(
                timestamp=event_timestamp_s,
                frame_index=event_frame_index,
                event_type="entry",
                pixel_point=event_point_xy,
                normalized_point=(nx, ny),
                meters_point=(mx, my),
                speed_before=event_speed,
                speed_after=event_speed,
                angle_change_deg=0.0,
            )

        def arm_entry_confirmation(
            *,
            event_point_xy: tuple[int, int],
            event_timestamp_s: float,
            event_frame_index: int,
            event_speed: float,
        ) -> ImpactEvent | None:
            if self.entry_confirm_frames <= 1:
                return emit_entry_event(
                    event_point_xy=event_point_xy,
                    event_timestamp_s=event_timestamp_s,
                    event_frame_index=event_frame_index,
                    event_speed=event_speed,
                )
            if not self.pending_entry_active:
                self.pending_entry_active = True
                self.pending_entry_count = 1
                self.pending_entry_point_xy = event_point_xy
                self.pending_entry_timestamp = float(event_timestamp_s)
                self.pending_entry_frame_index = int(event_frame_index)
                self.pending_entry_speed = float(event_speed)
                self.pending_entry_last_seen_time = float(now_s)
            return None

        if self.pending_entry_active:
            if not cur_inside:
                self._clear_pending_entry()
            else:
                self.pending_entry_count += 1
                self.pending_entry_last_seen_time = float(now_s)
                if self.pending_entry_count >= max(1, int(self.entry_confirm_frames)):
                    if self.pending_entry_point_xy is None:
                        self._clear_pending_entry()
                    else:
                        return emit_entry_event(
                            event_point_xy=self.pending_entry_point_xy,
                            event_timestamp_s=self.pending_entry_timestamp,
                            event_frame_index=self.pending_entry_frame_index,
                            event_speed=self.pending_entry_speed,
                        )

        if self.enable_entry_event and (not prev_inside) and cur_inside and entry_speed >= self.min_entry_speed_px_s:
            if not self._plane_contact_ok(d1, d2, ball_radius_m, plane_contact_tolerance_m):
                return None
            entry_point = find_entry_point(p1, p2, goal_corners)
            if entry_point is None:
                edge_point = closest_point_on_polygon(p2, goal_corners.astype(np.float32))
                entry_point_xy = (int(edge_point[0]), int(edge_point[1]))
            else:
                entry_point_xy = (int(entry_point[0]), int(entry_point[1]))

            event_point_xy = entry_display_point(entry_point_xy)
            return arm_entry_confirmation(
                event_point_xy=event_point_xy,
                event_timestamp_s=now_s,
                event_frame_index=frame_index,
                event_speed=entry_speed,
            )

        # Fallback for partial-visibility setups:
        # if the ball first appears already inside the goal polygon, infer an entry event
        # only when it is near the edge and moving inward.
        if self.allow_entry_fallbacks and self.enable_entry_event and prev_inside and cur_inside and entry_speed >= self.min_entry_speed_px_s:
            poly = goal_corners.astype(np.float32)
            signed1 = float(cv2.pointPolygonTest(poly, (float(p1[0]), float(p1[1])), True))
            signed2 = float(cv2.pointPolygonTest(poly, (float(p2[0]), float(p2[1])), True))
            if signed1 >= -radius_margin and signed2 >= -radius_margin:
                centroid = np.mean(poly, axis=0)
                moving_inward = float(np.dot((p2 - p1), (centroid - p1))) > 0.0
                edge_near = abs(signed1) <= (self.first_inside_edge_px + radius_margin)
                deeper_inside = (signed2 - signed1) >= max(0.5, self.first_inside_inward_px)
                if moving_inward and edge_near and deeper_inside:
                    if not self._plane_contact_ok(d1, d2, ball_radius_m, plane_contact_tolerance_m):
                        return None
                    edge_point = closest_point_on_polygon(p1, poly)
                    entry_point_xy = (int(edge_point[0]), int(edge_point[1]))
                    event_point_xy = entry_display_point(entry_point_xy)
                    return arm_entry_confirmation(
                        event_point_xy=event_point_xy,
                        event_timestamp_s=now_s,
                        event_frame_index=frame_index,
                        event_speed=entry_speed,
                    )

        # Marker-based 3D gating can be noisy by a few frames, so keep a
        # plane-aware fallback when the ball is clearly inside the opening and
        # has just reached the goal plane.
        if (
            self.allow_entry_fallbacks
            and
            self.enable_entry_event
            and cur_inside
            and entry_speed >= self.min_entry_speed_px_s
            and d2 is not None
            and ball_radius_m is not None
        ):
            plane_margin = max(0.02, (2.0 * float(ball_radius_m)) + max(0.0, float(plane_contact_tolerance_m)))
            near_plane_now = abs(float(d2)) <= plane_margin
            prior_distances = [
                abs(float(v))
                for v in (d0, d1)
                if v is not None
            ]
            farther_before = (not prev_inside) or any(v > (plane_margin * 1.10) for v in prior_distances)
            centroid = np.mean(goal_corners.astype(np.float32), axis=0)
            moving_inward = float(np.dot((p2 - p1), (centroid - p1))) > -0.05 * max(entry_speed, 1.0)
            deeper_inside = signed_cur >= (signed_prev - max(2.0, radius_margin * 0.25))
            if near_plane_now and farther_before and moving_inward and deeper_inside:
                edge_point = closest_point_on_polygon(p2, goal_corners.astype(np.float32))
                entry_point_xy = (int(edge_point[0]), int(edge_point[1]))
                event_point_xy = entry_display_point(entry_point_xy)
                return arm_entry_confirmation(
                    event_point_xy=event_point_xy,
                    event_timestamp_s=now_s,
                    event_frame_index=frame_index,
                    event_speed=entry_speed,
                )

        # Recovery fallback for detector/pose dropouts:
        # if the exact crossing frame was missed but the ball is now clearly
        # through the goal plane and we saw it on the camera side shortly
        # before, emit a single entry event.
        if (
            self.allow_entry_fallbacks
            and
            self.enable_entry_event
            and cur_inside
            and oriented_d2 is not None
            and ball_radius_m is not None
            and camera_side_sign is not None
        ):
            recovery_window_s = max(0.60, self.max_dt_s * 10.0)
            recently_camera_side = (now_s - self.last_camera_side_time) <= recovery_window_s
            through_surface_now = oriented_d2 <= (-max(0.01, float(ball_radius_m) * 0.55))
            centroid = np.mean(goal_corners.astype(np.float32), axis=0)
            moving_not_away = float(np.dot((p2 - p1), (centroid - p1))) > -0.20 * max(entry_speed, 1.0)
            if recently_camera_side and through_surface_now and moving_not_away:
                recovery_default_xy = center_px
                if self.last_camera_side_center is not None:
                    entry_point = find_entry_point(self.last_camera_side_center.astype(np.float32), p2, goal_corners)
                    if entry_point is not None:
                        recovery_default_xy = (int(entry_point[0]), int(entry_point[1]))
                    else:
                        edge_point = closest_point_on_polygon(p2, goal_corners.astype(np.float32))
                        recovery_default_xy = (int(edge_point[0]), int(edge_point[1]))
                event_point_xy = entry_display_point(recovery_default_xy)
                return arm_entry_confirmation(
                    event_point_xy=event_point_xy,
                    event_timestamp_s=now_s,
                    event_frame_index=frame_index,
                    event_speed=entry_speed,
                )

        if len(self.history) < 3:
            return None

        t0, p0, _d0 = self.history[-3]
        if (t1 - t0) > self.max_dt_s:
            return None

        dt0 = max(1e-4, t1 - t0)
        v0 = (p1 - p0) / dt0
        v1 = (p2 - p1) / dt1
        speed_before = float(np.linalg.norm(v0))
        speed_after = float(np.linalg.norm(v1))
        direction_change = angle_between(v0, v1)
        displacement = float(np.linalg.norm(p2 - p0))

        if displacement < self.min_displacement_px:
            return None

        if not cur_inside:
            return None

        speed_drop = speed_before >= self.min_pre_impact_speed and speed_after <= speed_before * self.speed_drop_ratio
        direction_bounce = speed_before >= self.min_pre_impact_speed and direction_change >= self.min_direction_change_deg
        if not (speed_drop or direction_bounce):
            return None

        if not self._plane_contact_ok(d1, d2, ball_radius_m, plane_contact_tolerance_m):
            return None

        # For the live hit map/UI, use the ball center projected onto the goal
        # plane instead of an inferred first-contact point shifted by velocity.
        # The contact estimate can look noticeably left/right of the visible ball
        # when the incoming trajectory has lateral motion, which feels wrong to
        # operators validating the system by eye.
        impact_point_xy = center_px

        nx, ny = project_event_point(impact_point_xy)
        nx = float(np.clip(nx, 0.0, 1.0))
        ny = float(np.clip(ny, 0.0, 1.0))
        mx = nx * self.goal_width_m
        my = ny * self.goal_height_m
        self.last_event_time = now_s
        self.event_latched = True
        self.history.clear()
        self.last_camera_side_time = -1e9
        self.last_camera_side_center = None

        return ImpactEvent(
            timestamp=now_s,
            frame_index=frame_index,
            event_type="impact",
            pixel_point=impact_point_xy,
            normalized_point=(nx, ny),
            meters_point=(mx, my),
            speed_before=speed_before,
            speed_after=speed_after,
            angle_change_deg=direction_change,
        )
