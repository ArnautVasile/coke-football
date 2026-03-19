from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AdaptationResult:
    corners: np.ndarray | None
    confidence: float
    matches: int
    inliers: int


class CameraAdapter:
    def __init__(
        self,
        reference_frame_bgr: np.ndarray,
        reference_corners: np.ndarray,
        process_scale: float = 0.65,
    ) -> None:
        self.orb = cv2.ORB_create(nfeatures=1200, scaleFactor=1.2, nlevels=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.process_scale = float(np.clip(process_scale, 0.3, 1.0))

        self.reference_corners = reference_corners.astype(np.float32)
        self.reference_gray, self.reference_corners_scaled = self._to_scaled_gray_and_corners(
            reference_frame_bgr, self.reference_corners
        )
        self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(self.reference_gray, None)

    def _to_scaled_gray_and_corners(
        self, frame_bgr: np.ndarray, corners: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if self.process_scale < 0.999:
            frame_scaled = cv2.resize(
                frame_bgr,
                None,
                fx=self.process_scale,
                fy=self.process_scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            frame_scaled = frame_bgr

        gray = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2GRAY)
        if corners is None:
            return gray, None

        corners_scaled = corners.astype(np.float32).copy()
        corners_scaled[:, 0] *= self.process_scale
        corners_scaled[:, 1] *= self.process_scale
        return gray, corners_scaled

    def reset_reference(self, reference_frame_bgr: np.ndarray, reference_corners: np.ndarray) -> None:
        self.reference_corners = reference_corners.astype(np.float32)
        self.reference_gray, self.reference_corners_scaled = self._to_scaled_gray_and_corners(
            reference_frame_bgr, self.reference_corners
        )
        self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(self.reference_gray, None)

    def adapt(self, frame_bgr: np.ndarray, min_good_matches: int = 18) -> AdaptationResult:
        if self.ref_descriptors is None or len(self.ref_descriptors) == 0:
            return AdaptationResult(corners=None, confidence=0.0, matches=0, inliers=0)

        gray, _ = self._to_scaled_gray_and_corners(frame_bgr, None)
        cur_keypoints, cur_descriptors = self.orb.detectAndCompute(gray, None)
        if cur_descriptors is None or len(cur_descriptors) < 2:
            return AdaptationResult(corners=None, confidence=0.0, matches=0, inliers=0)

        knn_matches = self.matcher.knnMatch(self.ref_descriptors, cur_descriptors, k=2)
        good = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < min_good_matches:
            return AdaptationResult(corners=None, confidence=0.0, matches=len(good), inliers=0)

        src_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([cur_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        homography, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        if homography is None or inlier_mask is None:
            return AdaptationResult(corners=None, confidence=0.0, matches=len(good), inliers=0)

        inliers = int(inlier_mask.sum())
        confidence = inliers / float(max(1, len(good)))
        if confidence < 0.35:
            return AdaptationResult(corners=None, confidence=confidence, matches=len(good), inliers=inliers)

        corners_scaled = cv2.perspectiveTransform(self.reference_corners_scaled.reshape(-1, 1, 2), homography).reshape(-1, 2)
        inv = 1.0 / self.process_scale
        corners = corners_scaled.copy()
        corners[:, 0] *= inv
        corners[:, 1] *= inv
        return AdaptationResult(corners=corners.astype(np.float32), confidence=confidence, matches=len(good), inliers=inliers)
