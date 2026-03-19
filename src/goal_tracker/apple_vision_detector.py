from __future__ import annotations

import base64
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .ball_identity import _crop_ball_square, load_identity_verifier
from .ball_detection import BallDetection
from .charuco import create_dictionary


@dataclass
class AppleVisionConfig:
    helper_bin: str
    model_path: str
    label: str = ""
    camera_index: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 60
    detect_every: int = 3
    confidence: float = 0.20
    compute_units: str = "all"
    local_search_scale: float = 2.6
    full_recover_every: int = 4
    max_age_s: float = 0.15
    max_area_ratio: float = 0.45
    max_aspect_ratio: float = 1.8
    min_area_ratio: float = 0.004
    identity_source: str = ""
    identity_threshold: float = 0.0


class AppleVisionBallDetector:
    def __init__(self, cfg: AppleVisionConfig) -> None:
        self.cfg = cfg
        self._process: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_message: dict | None = None
        self._latest_time = 0.0
        self._frames: deque[tuple[int, float, np.ndarray]] = deque(maxlen=4)
        self._detections_by_frame: dict[int, tuple[float, dict]] = {}
        self._active_frame_index: int | None = None
        self._stop = False
        self._identity_verifier = None
        self._load_identity_verifier()
        self._last_verified_frame_index: int | None = None
        self._last_verified_detection: BallDetection | None = None
        self._marker_dictionary = create_dictionary("DICT_6X6_50")
        self._marker_detector = cv2.aruco.ArucoDetector(self._marker_dictionary, cv2.aruco.DetectorParameters())
        self._last_debug_reason = ""
        self._last_debug_time = 0.0
        self._start_process()

    def _set_debug_reason(self, reason: str) -> None:
        self._last_debug_reason = reason
        self._last_debug_time = time.time()

    def _clear_debug_reason(self) -> None:
        self._last_debug_reason = ""
        self._last_debug_time = time.time()

    def get_debug_reason(self, max_age_s: float = 0.9) -> str:
        if not self._last_debug_reason:
            return ""
        if (time.time() - self._last_debug_time) > max(0.05, float(max_age_s)):
            return ""
        return self._last_debug_reason

    def _load_identity_verifier(self) -> None:
        source_value = self.cfg.identity_source.strip()
        if not source_value:
            return
        source_root = Path(source_value)
        if not source_root.exists():
            raise RuntimeError(f"Apple Vision identity source not found: {source_root}")
        verifier = load_identity_verifier(
            source=source_root,
            threshold=float(self.cfg.identity_threshold),
        )
        if verifier is None:
            raise RuntimeError(
                "Apple Vision identity verifier could not be built from the provided source. "
                "Use a labeled dataset root, a verifier positives directory, or a saved .npz/.onnx verifier model."
            )
        self._identity_verifier = verifier
        stats = verifier.describe()
        threshold_value = float(stats.get("threshold", 0.0))
        if "probability_threshold" in stats:
            print(
                "[Vision] learned exact-ball verifier loaded "
                f"(source={source_root}, threshold={threshold_value:.3f}, "
                f"prob={float(stats.get('probability_threshold', 0.0)):.3f}, "
                f"provider={stats.get('provider', 'unknown')})"
            )
        else:
            print(
                "[Vision] learned exact-ball verifier loaded "
                f"(source={source_root}, threshold={threshold_value:.3f}, "
                f"pos95={float(stats.get('positive_distance_p95', float('nan'))):.3f}, "
                f"impostor05={float(stats.get('impostor_distance_p05', float('nan'))):.3f})"
            )

    def _start_process(self) -> None:
        helper_path = Path(self.cfg.helper_bin)
        if not helper_path.exists():
            raise RuntimeError(
                f"Apple Vision helper not found: {helper_path}\n"
                "Build it first with: cd apple/BallVisionHelper && swift build -c release"
            )
        model_path = Path(self.cfg.model_path)
        if not model_path.exists():
            raise RuntimeError(f"Apple Vision model not found: {model_path}")

        cmd = [
            str(helper_path),
            "--model",
            str(model_path),
            "--camera",
            str(self.cfg.camera_index),
            "--width",
            str(self.cfg.width),
            "--height",
            str(self.cfg.height),
            "--fps",
            str(self.cfg.fps),
            "--detect-every",
            str(max(1, self.cfg.detect_every)),
            "--confidence",
            f"{float(self.cfg.confidence):.3f}",
            "--compute-units",
            str(self.cfg.compute_units or "all"),
            "--local-search-scale",
            f"{float(self.cfg.local_search_scale):.2f}",
            "--full-recover-every",
            str(max(1, int(self.cfg.full_recover_every))),
            "--emit-frames",
        ]
        if self.cfg.label.strip():
            cmd.extend(["--label", self.cfg.label.strip()])

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._stdout_thread = threading.Thread(target=self._read_stdout, name="AppleVisionStdout", daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, name="AppleVisionStderr", daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for raw_line in self._process.stdout:
            if self._stop:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            with self._lock:
                payload_type = str(payload.get("type", "detection")).strip().lower()
                frame_index = int(payload.get("frameIndex", -1))
                seen_at = time.time()
                if payload_type == "frame":
                    jpeg_b64 = payload.get("jpeg")
                    if isinstance(jpeg_b64, str) and jpeg_b64:
                        frame = self._decode_frame(jpeg_b64)
                        if frame is not None:
                            self._frames.append((frame_index, seen_at, frame))
                    continue
                self._latest_message = payload
                self._latest_time = seen_at
                if frame_index >= 0:
                    self._detections_by_frame[frame_index] = (seen_at, payload)
                    stale = [idx for idx in self._detections_by_frame.keys() if idx < frame_index - 8]
                    for idx in stale:
                        self._detections_by_frame.pop(idx, None)

    def _decode_frame(self, jpeg_b64: str) -> np.ndarray | None:
        try:
            encoded = base64.b64decode(jpeg_b64)
        except Exception:
            return None
        arr = np.frombuffer(encoded, dtype=np.uint8)
        if arr.size == 0:
            return None
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for raw_line in self._process.stderr:
            if self._stop:
                break
            line = raw_line.rstrip()
            if not line:
                continue
            if "input_fps=" in line:
                continue
            print(line)

    def _message_to_detection(
        self,
        payload: dict,
        frame_shape: tuple[int, int, int],
        roi: tuple[int, int, int, int] | None,
    ) -> tuple[BallDetection | None, str]:
        h, w = frame_shape[:2]
        width_norm = float(payload.get("width", 0.0))
        height_norm = float(payload.get("height", 0.0))
        if width_norm <= 0.0 or height_norm <= 0.0:
            return None, "reject: helper invalid box"
        width = width_norm * w
        height = height_norm * h
        if width < 10.0 or height < 10.0:
            return None, "reject: helper tiny"
        area_ratio = width_norm * height_norm
        aspect_ratio = max(width, height) / max(1e-6, min(width, height))
        if area_ratio > max(0.02, float(self.cfg.max_area_ratio)):
            return None, "reject: helper large"
        if aspect_ratio > max(1.0, float(self.cfg.max_aspect_ratio)):
            return None, "reject: helper aspect"
        cx = int(round(float(payload.get("x", 0.0)) * w))
        cy = int(round(float(payload.get("y", 0.0)) * h))
        # The helper emits width/height normalized to frame width/height, so
        # reconstruct pixel geometry first and derive a stable on-screen radius
        # from the real pixel box instead of mixing normalized values with the
        # frame's max dimension.
        radius = 0.25 * (width + height)
        area = max(1.0, width * height)
        detection = BallDetection(center=(cx, cy), radius=max(4.0, radius), area=area, circularity=1.0)
        filter_reason = ""
        if area_ratio < max(1e-5, float(self.cfg.min_area_ratio)):
            filter_reason = "reject: helper small"

        if roi is None:
            return detection, filter_reason
        x1, y1, x2, y2 = roi
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            return None, "reject: helper roi"
        return detection, filter_reason

    def _remember_verified_detection(self, detection: BallDetection, frame_index: int) -> None:
        if frame_index >= 0:
            self._last_verified_frame_index = frame_index
        self._last_verified_detection = detection

    def _matches_recent_verified_ball(
        self,
        detection: BallDetection,
        *,
        frame_index: int,
        max_gap_frames: int | None = None,
    ) -> bool:
        if frame_index < 0 or self._last_verified_frame_index is None:
            return False
        gap_frames = frame_index - self._last_verified_frame_index
        allowed_gap = max(2, int(self.cfg.detect_every) + 2) if max_gap_frames is None else max(1, int(max_gap_frames))
        if gap_frames > allowed_gap:
            return False
        prev = self._last_verified_detection
        if prev is None:
            return False
        prev_center = np.asarray(prev.center, dtype=np.float32)
        curr_center = np.asarray(detection.center, dtype=np.float32)
        center_distance = float(np.linalg.norm(curr_center - prev_center))
        # Allow larger motion jumps when detector cadence is sparse and the ball is small.
        motion_scale = 3.6 + (0.70 * max(1, int(self.cfg.detect_every)))
        if gap_frames > 1:
            motion_scale += 0.35 * min(4, gap_frames - 1)
        max_distance = max(36.0, motion_scale * max(prev.radius, detection.radius))
        if center_distance > max_distance:
            return False
        prev_radius = max(1.0, float(prev.radius))
        curr_radius = max(1.0, float(detection.radius))
        radius_ratio = curr_radius / prev_radius
        if gap_frames > 1:
            min_ratio = 0.42
            max_ratio = 2.30
        else:
            min_ratio = 0.50
            max_ratio = 2.05
        if radius_ratio < min_ratio or radius_ratio > max_ratio:
            return False
        return True

    def _can_identity_rescue(
        self,
        detection: BallDetection,
        *,
        frame_index: int,
        helper_confidence: float,
        identity_score: float,
        identity_threshold: float,
        source: str,
        frame_bgr: np.ndarray,
    ) -> bool:
        # Keep rescue path intentionally narrow: local continuation only.
        # This preserves real moving-ball continuity while avoiding fresh
        # full-frame promotions of round-ish clutter.
        if source != "detect_local":
            return False
        if helper_confidence < max(0.18, float(self.cfg.confidence) + 0.06):
            return False
        soft_threshold = float(identity_threshold) * 1.02
        if identity_score > soft_threshold:
            return False
        if not self._matches_recent_verified_ball(detection, frame_index=frame_index, max_gap_frames=2):
            return False
        shape_score = self._estimate_shape_roundness(frame_bgr, detection)
        if shape_score is not None and shape_score < 0.54:
            return False
        return True

    def _estimate_shape_roundness(self, frame_bgr: np.ndarray, detection: BallDetection) -> float | None:
        crop = _crop_ball_square(frame_bgr, detection.center, detection.radius, scale=1.12)
        if crop is None or crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 140)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        crop_h, crop_w = crop.shape[:2]
        crop_area = float(crop_h * crop_w)
        crop_center = np.asarray([crop_w * 0.5, crop_h * 0.5], dtype=np.float32)
        max_side = float(max(crop_w, crop_h))
        best_score: float | None = None

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < crop_area * 0.04:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter < 1.0:
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-6:
                continue
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
            center_distance = float(np.linalg.norm(np.asarray([cx, cy], dtype=np.float32) - crop_center))
            if center_distance > (0.35 * max_side):
                continue

            (_, _), enclosing_radius = cv2.minEnclosingCircle(contour)
            if enclosing_radius < 3.0:
                continue
            circularity = float((4.0 * np.pi * area) / max(1e-6, perimeter * perimeter))
            fill_ratio = float(area / max(1e-6, np.pi * enclosing_radius * enclosing_radius))

            points = contour.reshape(-1, 2).astype(np.float32)
            radial = np.linalg.norm(points - np.asarray([cx, cy], dtype=np.float32), axis=1)
            radial_std = float(np.std(radial) / max(1e-6, np.mean(radial)))
            center_align = float(np.clip(1.0 - center_distance / max(1.0, 0.35 * max_side), 0.0, 1.0))

            score = (
                0.40 * float(np.clip(circularity, 0.0, 1.0))
                + 0.35 * float(np.clip((fill_ratio - 0.45) / 0.50, 0.0, 1.0))
                + 0.15 * float(np.clip(1.0 - radial_std * 4.0, 0.0, 1.0))
                + 0.10 * center_align
            )
            if best_score is None or score > best_score:
                best_score = float(score)
        return best_score

    def _crop_contains_marker(self, frame_bgr: np.ndarray, detection: BallDetection) -> bool:
        crop = _crop_ball_square(frame_bgr, detection.center, detection.radius, scale=1.22)
        if crop is None or crop.size == 0:
            return False
        try:
            marker_corners, marker_ids, _ = self._marker_detector.detectMarkers(crop)
        except cv2.error:
            return False
        if marker_ids is None or len(marker_ids) == 0:
            return False
        crop_h, crop_w = crop.shape[:2]
        crop_center = np.asarray([crop_w * 0.5, crop_h * 0.5], dtype=np.float32)
        max_side = float(max(crop_w, crop_h))
        for corners in marker_corners:
            pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] < 4:
                continue
            marker_center = np.mean(pts, axis=0)
            center_distance = float(np.linalg.norm(marker_center - crop_center))
            marker_w = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
            marker_h = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
            marker_side = max(marker_w, marker_h)
            if marker_side < max(8.0, 0.16 * max_side):
                continue
            if center_distance <= (0.38 * max_side):
                return True
        return False

    def _passes_detection_hysteresis(
        self,
        detection: BallDetection,
        *,
        frame_bgr: np.ndarray,
        frame_index: int,
        helper_confidence: float,
        source: str,
        identity_score: float,
        identity_threshold: float,
    ) -> tuple[bool, str]:
        has_anchor = self._last_verified_detection is not None and self._last_verified_frame_index is not None
        recent_match = self._matches_recent_verified_ball(
            detection,
            frame_index=frame_index,
            max_gap_frames=max(2, int(self.cfg.detect_every) + 2),
        )
        strong_identity = identity_score <= max(0.02, identity_threshold * 0.82)
        very_strong_identity = identity_score <= max(0.02, identity_threshold * 0.68)
        shape_score = self._estimate_shape_roundness(frame_bgr, detection)
        if shape_score is not None:
            detection.circularity = float(shape_score)

        base_conf = float(self.cfg.confidence)
        fresh_conf_floor = max(0.10, base_conf + 0.02)
        continuity_conf_floor = max(0.06, base_conf * 0.75)
        bootstrap_conf_floor = max(0.20, base_conf + 0.08)

        if source == "track":
            if not recent_match:
                # Block tracker drift onto wall text/clutter unless we have
                # exceptionally strong evidence.
                if helper_confidence < max(0.24, base_conf + 0.08):
                    return False, "reject: track drift"
                if not strong_identity:
                    return False, "reject: track drift"
                if shape_score is not None and shape_score < 0.58:
                    return False, "reject: track shape"
            if helper_confidence < continuity_conf_floor and not strong_identity:
                return False, "reject: track weak"
            return True, ""

        if source == "detect_local":
            if recent_match:
                if helper_confidence < continuity_conf_floor and not strong_identity:
                    return False, "reject: local weak"
                return True, ""
            if helper_confidence < max(0.12, base_conf + 0.06):
                return False, "reject: fresh weak"
        elif source in {"detect", "detect_full"}:
            # Fresh full-frame proposals must pass confidence floor regardless of
            # identity score to avoid accepting low-confidence round clutter.
            if not recent_match and helper_confidence < fresh_conf_floor:
                return False, "reject: fresh weak"

        if not has_anchor and source in {"detect", "detect_full", "detect_local"}:
            # First lock must be clean: this prevents bootstrapping on wall text or
            # round-ish clutter before we ever see a trusted ball.
            if helper_confidence < bootstrap_conf_floor:
                return False, "reject: bootstrap weak"
            if identity_threshold > 0.0:
                bootstrap_identity_floor = max(0.02, float(identity_threshold) * 0.78)
                if identity_score > bootstrap_identity_floor:
                    return False, "reject: bootstrap identity"
            if shape_score is None:
                return False, "reject: bootstrap shape"
            if shape_score < 0.60:
                return False, "reject: bootstrap shape"

        if not recent_match and source in {"detect", "detect_full", "detect_local"}:
            # Precision/recall balance: unknown shape alone should not kill a very
            # strong identity+confidence candidate, but low-confidence unknown
            # shapes are usually clutter.
            if shape_score is None and helper_confidence < max(0.16, base_conf + 0.02) and not very_strong_identity:
                return False, "reject: shape unknown"
            if shape_score is not None and shape_score < 0.54 and not very_strong_identity:
                return False, "reject: shape"
        return True, ""

    def detect(self, frame_bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> BallDetection | None:
        if self._process is None:
            self._set_debug_reason("reject: helper unavailable")
            return None
        if self._process.poll() is not None:
            self._set_debug_reason("reject: helper exited")
            raise RuntimeError("Apple Vision helper exited unexpectedly.")

        with self._lock:
            active_idx = self._active_frame_index
            payload: dict | None = None
            seen_at = 0.0
            if active_idx is not None:
                entry = self._detections_by_frame.get(active_idx)
                if entry is not None:
                    seen_at, raw_payload = entry
                    payload = dict(raw_payload)
            if payload is None and self._latest_message is not None:
                latest_payload = dict(self._latest_message)
                latest_idx = int(latest_payload.get("frameIndex", -1))
                # Helper detection and frame streams are asynchronous; accept a
                # tiny frame-index mismatch to avoid "waiting helper" stalls.
                if active_idx is None or latest_idx == active_idx or (active_idx >= 0 and abs(latest_idx - active_idx) <= 1):
                    payload = latest_payload
                    seen_at = self._latest_time
        if payload is None:
            self._set_debug_reason("reject: waiting helper")
            return None
        if (time.time() - seen_at) > max(0.05, float(self.cfg.max_age_s)):
            self._set_debug_reason("reject: stale")
            return None
        detection, filter_reason = self._message_to_detection(payload, frame_bgr.shape, roi)
        if detection is None:
            self._set_debug_reason(filter_reason or "reject: helper filter")
            return None
        helper_confidence = float(payload.get("confidence", 0.0))
        source = str(payload.get("source", "detect")).strip().lower()
        frame_index = int(payload.get("frameIndex", -1))
        if source == "track" and not self._track_sequence_is_verified(frame_index):
            self._set_debug_reason("reject: unverified track")
            return None
        if self._identity_verifier is None:
            if filter_reason:
                self._set_debug_reason(filter_reason)
                return None
            ok_hysteresis, hysteresis_reason = self._passes_detection_hysteresis(
                detection,
                frame_bgr=frame_bgr,
                frame_index=frame_index,
                helper_confidence=helper_confidence,
                source=source,
                identity_score=0.0,
                identity_threshold=0.0,
            )
            if not ok_hysteresis:
                self._set_debug_reason(hysteresis_reason)
                return None
            self._remember_verified_detection(detection, frame_index)
            self._clear_debug_reason()
            return detection
        identity = self._identity_verifier.verify(frame_bgr, detection)
        if identity.accepted:
            recent_match = self._matches_recent_verified_ball(
                detection,
                frame_index=frame_index,
                max_gap_frames=max(2, int(self.cfg.detect_every) + 2),
            )
            if filter_reason == "reject: helper small" and not recent_match:
                self._set_debug_reason("reject: helper small")
                return None
            ok_hysteresis, hysteresis_reason = self._passes_detection_hysteresis(
                detection,
                frame_bgr=frame_bgr,
                frame_index=frame_index,
                helper_confidence=helper_confidence,
                source=source,
                identity_score=identity.score,
                identity_threshold=identity.threshold,
            )
            if not ok_hysteresis:
                self._set_debug_reason(hysteresis_reason)
                return None
            if self._crop_contains_marker(frame_bgr, detection):
                self._set_debug_reason("reject: marker")
                return None
            self._remember_verified_detection(detection, frame_index)
            if filter_reason:
                self._set_debug_reason("accept: helper small verified")
            else:
                self._clear_debug_reason()
            return detection
        if filter_reason:
            self._set_debug_reason(filter_reason)
            return None
        if self._crop_contains_marker(frame_bgr, detection):
            self._set_debug_reason("reject: marker")
            return None
        # Keep a very small rescue window only for continuity of a recently
        # verified ball. This avoids promoting unrelated fresh detections such
        # as goal markers when they happen to score close to the verifier
        # threshold.
        if self._can_identity_rescue(
            detection,
            frame_index=frame_index,
            helper_confidence=helper_confidence,
            identity_score=identity.score,
            identity_threshold=identity.threshold,
            source=source,
            frame_bgr=frame_bgr,
        ):
            self._remember_verified_detection(detection, frame_index)
            self._set_debug_reason("accept: identity rescue")
            return detection
        self._set_debug_reason("reject: identity")
        return None

    def _track_sequence_is_verified(self, frame_index: int) -> bool:
        if frame_index < 0:
            return False
        if self._last_verified_frame_index is None:
            return False
        return (frame_index - self._last_verified_frame_index) <= 10

    def read_frame(self, timeout_s: float = 2.0) -> np.ndarray | None:
        deadline = time.time() + max(0.01, timeout_s)
        last_seen_idx = self._active_frame_index
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError("Apple Vision helper exited unexpectedly.")
            with self._lock:
                if self._frames:
                    chosen: tuple[int, float, np.ndarray] | None = None
                    latest = self._frames[-1]
                    for idx, ts, frame in list(self._frames):
                        if last_seen_idx is None or idx > last_seen_idx:
                            chosen = (idx, ts, frame)
                    if chosen is None and last_seen_idx is None:
                        chosen = latest
                    if chosen is None:
                        pass
                    else:
                        idx, _, frame = chosen
                        self._active_frame_index = idx
                        return frame.copy()
            time.sleep(0.005)
        return None

    def close(self) -> None:
        self._stop = True
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
