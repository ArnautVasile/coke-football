from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BallDetection:
    center: tuple[int, int]
    radius: float
    area: float
    circularity: float


class MotionBallDetector:
    def __init__(
        self,
        min_area: int = 120,
        max_area: int = 6000,
        min_circularity: float = 0.42,
        process_scale: float = 1.0,
        warmup_frames: int = 45,
        learning_rate: float = 0.002,
        enable_static_fallback: bool = True,
        static_every_n: int = 3,
        static_hough_param2: float = 20.0,
        static_min_radius: int = 7,
        static_max_radius: int = 90,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.process_scale = float(np.clip(process_scale, 0.2, 1.0))
        self.warmup_frames = max(0, int(warmup_frames))
        self.learning_rate = float(np.clip(learning_rate, 0.0, 1.0))
        self.enable_static_fallback = bool(enable_static_fallback)
        self.static_every_n = max(1, int(static_every_n))
        self.static_hough_param2 = float(np.clip(static_hough_param2, 8.0, 60.0))
        self.static_min_radius = max(2, int(static_min_radius))
        self.static_max_radius = max(self.static_min_radius + 1, int(static_max_radius))
        self.frame_count = 0
        self.no_motion_frames = 0
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=24, detectShadows=False)

    def _resolve_roi(self, frame_bgr: np.ndarray, roi: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
        if roi is None:
            x1, y1, x2, y2 = 0, 0, frame_bgr.shape[1], frame_bgr.shape[0]
        else:
            x1, y1, x2, y2 = roi
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_bgr.shape[1], x2)
            y2 = min(frame_bgr.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _full_detection_from_proc(
        self,
        cx: float,
        cy: float,
        radius: float,
        area_proc: float,
        circularity: float,
        x1: int,
        y1: int,
    ) -> BallDetection:
        inv = 1.0 / self.process_scale
        scale_sq = self.process_scale * self.process_scale
        return BallDetection(
            center=(int(cx * inv + x1), int(cy * inv + y1)),
            radius=float(radius * inv),
            area=float(area_proc / max(scale_sq, 1e-6)),
            circularity=float(circularity),
        )

    def _detect_motion_candidate(self, proc: np.ndarray, x1: int, y1: int) -> BallDetection | None:
        scale_sq = self.process_scale * self.process_scale
        min_area = max(8.0, self.min_area * scale_sq)
        max_area = max(min_area + 1.0, self.max_area * scale_sq)

        fg_mask = self.bg_subtractor.apply(proc, learningRate=self.learning_rate)
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        _, fg_mask = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), dtype=np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best: BallDetection | None = None
        best_score = -1.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter < 1e-3:
                continue

            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            score = float(circularity * area)
            if score > best_score:
                best = self._full_detection_from_proc(cx, cy, radius, area, circularity, x1, y1)
                best_score = score

        return best

    def _detect_static_circle_candidate(self, proc: np.ndarray, x1: int, y1: int) -> BallDetection | None:
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.2)

        min_r = max(2, int(self.static_min_radius * self.process_scale))
        max_r = max(min_r + 1, int(self.static_max_radius * self.process_scale))
        min_dist = max(10, int(min(gray.shape[0], gray.shape[1]) * 0.18))

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=110,
            param2=self.static_hough_param2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is None:
            return None

        best: BallDetection | None = None
        best_score = -1.0

        for c in np.round(circles[0]).astype(np.int32):
            cx, cy, radius = int(c[0]), int(c[1]), int(c[2])
            area_proc = float(np.pi * radius * radius)
            area_full = area_proc / max(self.process_scale * self.process_scale, 1e-6)
            if area_full < self.min_area or area_full > self.max_area * 1.5:
                continue

            # Simple score: larger circles are usually the ball in this constrained ROI.
            score = float(radius)
            if score > best_score:
                best = self._full_detection_from_proc(
                    float(cx),
                    float(cy),
                    float(radius),
                    area_proc,
                    1.0,
                    x1,
                    y1,
                )
                best_score = score

        return best

    def detect(self, frame_bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> BallDetection | None:
        bounds = self._resolve_roi(frame_bgr, roi)
        if bounds is None:
            return None
        x1, y1, x2, y2 = bounds

        crop = frame_bgr[y1:y2, x1:x2]
        if self.process_scale < 0.999:
            proc = cv2.resize(
                crop,
                None,
                fx=self.process_scale,
                fy=self.process_scale,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            proc = crop

        self.frame_count += 1
        if self.frame_count <= self.warmup_frames:
            # Let the background model settle before trusting detections.
            self.bg_subtractor.apply(proc, learningRate=0.10)
            return None

        motion = self._detect_motion_candidate(proc, x1, y1)
        if motion is not None:
            self.no_motion_frames = 0
            return motion

        if not self.enable_static_fallback:
            return None

        self.no_motion_frames += 1
        if self.no_motion_frames % self.static_every_n != 0:
            return None

        return self._detect_static_circle_candidate(proc, x1, y1)
