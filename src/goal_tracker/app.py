from __future__ import annotations

import argparse
import csv
import platform
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .apple_vision_detector import AppleVisionBallDetector, AppleVisionConfig
from .ball_detection import BallDetection, MotionBallDetector
from .calibration import CalibrationData, calibrate_goal_corners, load_calibration, save_calibration, scale_corners
from .camera_adaptation import CameraAdapter
from .camera_intrinsics import CameraIntrinsics, load_camera_intrinsics, undistort_frame
from .goal_markers import GoalMarkerLayout, load_goal_marker_layout
from .goal_pose import BallPlaneEstimate, GoalPoseEstimate, estimate_ball_plane_distance, project_pixel_to_goal_plane, solve_goal_pose
from .impact import ImpactDetector, ImpactEvent, build_goal_homography, signed_distance_to_polygon
from .yolo_ball_detection import YoloBallDetector, YoloConfig


class BallDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> BallDetection | None: ...


class HybridBallDetector:
    """YOLO-first detector with motion fallback for blur-heavy fast shots."""

    def __init__(self, primary: BallDetector, fallback: BallDetector) -> None:
        self.primary = primary
        self.fallback = fallback

    def detect(self, frame_bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> BallDetection | None:
        detection = self.primary.detect(frame_bgr, roi=roi)
        if detection is not None:
            return detection
        return self.fallback.detect(frame_bgr, roi=roi)

    def get_debug_reason(self, max_age_s: float = 0.9) -> str:
        primary_reason = getattr(self.primary, "get_debug_reason", None)
        if callable(primary_reason):
            reason = primary_reason(max_age_s=max_age_s)
            if reason:
                return reason
        fallback_reason = getattr(self.fallback, "get_debug_reason", None)
        if callable(fallback_reason):
            return fallback_reason(max_age_s=max_age_s)
        return ""


class AsyncLatestCapture:
    """Background camera reader that always exposes the latest frame."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_seq = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="AsyncLatestCapture", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.002)
                continue
            with self._lock:
                self._latest_frame = frame
                self._latest_seq += 1

    def seed(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest_frame = frame
            if self._latest_seq <= 0:
                self._latest_seq = 1

    def read(
        self,
        timeout_s: float = 0.20,
        *,
        after_seq: int | None = None,
    ) -> tuple[bool, np.ndarray | None, int]:
        start = time.time()
        with self._lock:
            latest = self._latest_frame
            latest_seq = self._latest_seq
        if latest is not None and (after_seq is None or latest_seq != after_seq):
            return True, latest, latest_seq
        while (time.time() - start) <= max(0.01, float(timeout_s)):
            with self._lock:
                latest = self._latest_frame
                latest_seq = self._latest_seq
                if latest is not None and (after_seq is None or latest_seq != after_seq):
                    return True, latest, latest_seq
            time.sleep(0.001)
        with self._lock:
            latest = self._latest_frame
            latest_seq = self._latest_seq
            return (latest is not None), latest, latest_seq

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.35)
            self._thread = None


def hit_zone_name(nx: float, ny: float) -> str:
    if nx < 0.33:
        col = "Left"
    elif nx < 0.66:
        col = "Center"
    else:
        col = "Right"

    if ny < 0.33:
        row = "Top"
    elif ny < 0.66:
        row = "Middle"
    else:
        row = "Bottom"
    return f"{row}-{col}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Goal corner calibration + ball impact detector")
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--camera", default="0", help="Camera index or stream URL/path")
    parser.add_argument("--backend", choices=["auto", "avfoundation", "msmf", "dshow"], default="auto")
    parser.add_argument("--fourcc", default="", help="Optional camera codec FourCC (e.g. MJPG, H264)")
    parser.add_argument("--probe-camera", action="store_true", help="Probe camera modes and exit")
    parser.add_argument("--probe-seconds", type=float, default=2.5, help="Seconds per mode in camera probe")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--async-capture",
        dest="async_capture",
        action="store_true",
        help="Use a background latest-frame camera reader for live camera input",
    )
    parser.add_argument(
        "--no-async-capture",
        dest="async_capture",
        action="store_false",
        help="Disable background latest-frame camera reader",
    )
    parser.set_defaults(async_capture=True)
    parser.add_argument("--force-resize-input", action="store_true", help="Resize incoming frames to --width/--height")
    parser.add_argument("--goal-width-m", type=float, default=7.32)
    parser.add_argument("--goal-height-m", type=float, default=2.44)
    parser.add_argument("--calibration-file", default="data/calibration/goal_calibration.json")
    parser.add_argument(
        "--camera-calibration-file",
        default="",
        help="Optional camera intrinsics JSON from the ChArUco calibration tool",
    )
    parser.add_argument(
        "--goal-markers-layout",
        default="",
        help="Optional goal marker layout JSON for marker-based goal pose tracking",
    )
    parser.add_argument("--reference-frame", default="data/calibration/reference_frame.jpg")
    parser.add_argument("--log-file", default="data/logs/hits.csv")
    parser.add_argument("--adapt-every", type=int, default=8, help="Run camera adaptation every N frames")
    parser.add_argument("--adapt-alpha", type=float, default=0.25, help="Corner smoothing factor [0..1]")
    parser.add_argument("--adapt-scale", type=float, default=0.65, help="Image scale for camera adaptation")
    parser.add_argument("--adapt-min-confidence", type=float, default=0.5, help="Min confidence to apply adapted corners")
    parser.add_argument("--no-auto-adapt", action="store_true", help="Disable ORB-based camera shift adaptation")
    parser.add_argument(
        "--undistort-input",
        action="store_true",
        help="Undistort OpenCV camera frames using --camera-calibration-file before calibration/detection",
    )
    parser.add_argument("--process-every", type=int, default=1, help="Run ball detector every N frames")
    parser.add_argument(
        "--ball-only-mode",
        action="store_true",
        help="Ball-only debug mode: no goal-gate filtering, no hit logic, and no goal-frame overlay panel",
    )
    parser.add_argument("--detect-full-frame", dest="detect_full_frame", action="store_true", help="Run detector on full frame for reliability")
    parser.add_argument("--no-detect-full-frame", dest="detect_full_frame", action="store_false", help="Run detector only near calibrated goal ROI")
    parser.set_defaults(detect_full_frame=True)
    parser.add_argument(
        "--fixed-detect-roi",
        default="",
        help="Optional fixed detector ROI as x1,y1,x2,y2 in pixels. When set, detector search is constrained to this box.",
    )
    parser.add_argument("--detect-roi-margin", type=int, default=100, help="ROI margin (px) when not using full-frame detection")
    parser.add_argument(
        "--global-search-every",
        type=int,
        default=2,
        help="In ROI mode, run a periodic full-frame search every N detector frames (0 disables)",
    )
    parser.add_argument(
        "--background-gate",
        action="store_true",
        help="Reject detections that look too similar to a static background reference image",
    )
    parser.add_argument(
        "--background-reference",
        default="",
        help="Path to a background reference image (same camera/view). Used with --background-gate.",
    )
    parser.add_argument(
        "--background-diff-threshold",
        type=float,
        default=16.0,
        help="Per-pixel grayscale difference threshold for background-gate active-ratio test",
    )
    parser.add_argument(
        "--background-active-ratio",
        type=float,
        default=0.08,
        help="Min changed-pixel ratio in the local candidate patch for background-gate pass",
    )
    parser.add_argument(
        "--background-patch-scale",
        type=float,
        default=1.22,
        help="Scale factor for local candidate patch used by background-gate",
    )
    parser.add_argument("--ball-tracker", choices=["none", "mil"], default="none", help="Optional tracker for robust ball lock")
    parser.add_argument("--manual-ball-select-on-start", action="store_true", help="Select ball ROI at startup to lock tracker")
    parser.add_argument("--tracker-fail-max", type=int, default=6, help="Consecutive failed tracker updates before unlock")
    parser.add_argument("--auto-lock-tracker", dest="auto_lock_tracker", action="store_true", help="Auto-start tracker once trusted ball appears")
    parser.add_argument("--no-auto-lock-tracker", dest="auto_lock_tracker", action="store_false", help="Do not auto-lock tracker from detections")
    parser.set_defaults(auto_lock_tracker=False)
    parser.add_argument("--tracker-lock-scale", type=float, default=1.9, help="Tracker init box size scale from detected ball radius")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Display-only resize factor to reduce GUI cost")
    parser.add_argument("--display-every", type=int, default=1, help="Refresh OpenCV preview every N frames (higher improves FPS)")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV preview/UI loop for maximum processing FPS")
    parser.add_argument("--stats-every", type=int, default=120, help="Print runtime stats every N frames")
    parser.add_argument(
        "--perf-breakdown",
        action="store_true",
        help="Print per-stage timing breakdown every --stats-every frames to diagnose FPS bottlenecks",
    )
    parser.add_argument(
        "--perf-breakdown-top",
        type=int,
        default=6,
        help="How many top stages to print when --perf-breakdown is enabled",
    )
    parser.add_argument("--detector", choices=["motion", "yolo", "hybrid", "vision"], default="motion")
    parser.add_argument("--min-area", type=int, default=120, help="Min contour area for ball candidates")
    parser.add_argument("--max-area", type=int, default=6000, help="Max contour area for ball candidates")
    parser.add_argument("--min-circularity", type=float, default=0.42, help="Ball circularity threshold")
    parser.add_argument("--motion-scale", type=float, default=0.6, help="Image scale for motion detector")
    parser.add_argument("--motion-warmup-frames", type=int, default=45, help="Ignore motion detections for first N frames")
    parser.add_argument("--motion-static-fallback", dest="motion_static_fallback", action="store_true", help="Enable circle-based fallback when motion is weak")
    parser.add_argument("--no-motion-static-fallback", dest="motion_static_fallback", action="store_false", help="Disable circle-based fallback")
    parser.set_defaults(motion_static_fallback=True)
    parser.add_argument("--motion-static-every", type=int, default=3, help="Run static circle fallback every N no-motion frames")
    parser.add_argument("--motion-static-min-radius", type=int, default=7, help="Min radius for static circle fallback")
    parser.add_argument("--motion-static-max-radius", type=int, default=90, help="Max radius for static circle fallback")
    parser.add_argument("--motion-static-param2", type=float, default=20.0, help="Hough circle sensitivity (lower=more detections)")
    parser.add_argument("--yolo-model", default="yolo26s.pt", help="YOLO model path or name")
    parser.add_argument("--yolo-conf", type=float, default=0.10)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-device", default="", help="Example: cpu, mps, cuda:0")
    parser.add_argument("--yolo-class-id", type=int, default=32, help="COCO sports ball class id is 32")
    parser.add_argument("--yolo-track", dest="yolo_track", action="store_true", help="Use YOLO tracking mode (ByteTrack/BOTSort) for stability")
    parser.add_argument("--no-yolo-track", dest="yolo_track", action="store_false", help="Use frame-by-frame YOLO detect without tracker")
    parser.set_defaults(yolo_track=True)
    parser.add_argument("--yolo-tracker", default="configs/bytetrack_ball.yaml", help="Tracker config for YOLO track mode")
    parser.add_argument(
        "--yolo-identity-source",
        default="",
        help=(
            "Optional exact-ball verifier source for YOLO/hybrid mode. Can be a labeled dataset root, "
            "a positives directory, or a saved .npz/.onnx verifier model."
        ),
    )
    parser.add_argument(
        "--yolo-identity-threshold",
        type=float,
        default=0.0,
        help="Optional exact-ball verifier threshold override for YOLO/hybrid mode (<=0 uses the learned default)",
    )
    parser.add_argument(
        "--vision-helper-bin",
        default=str(repo_root / "apple" / "BallVisionHelper" / ".build" / "release" / "BallVisionHelper"),
        help="Path to the native Apple Vision helper binary",
    )
    parser.add_argument("--vision-model", default="", help="Path to Core ML model (.mlmodelc or .mlpackage) for native Apple detector")
    parser.add_argument("--vision-label", default="", help="Optional label filter for the Apple Vision helper")
    parser.add_argument("--vision-camera", default="", help="Optional numeric camera index for the helper (defaults to --camera)")
    parser.add_argument("--vision-detect-every", type=int, default=2, help="Run Core ML detection every N helper frames and track between them")
    parser.add_argument("--vision-confidence", type=float, default=0.12, help="Min confidence for the Apple Vision helper")
    parser.add_argument("--vision-local-search-scale", type=float, default=2.6, help="When the helper already has the ball, search around the tracked position by this scale factor before full recovery")
    parser.add_argument("--vision-full-recover-every", type=int, default=2, help="Force a periodic full-frame detector recovery every N helper detect passes while tracking")
    parser.add_argument(
        "--vision-compute-units",
        choices=["all", "cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine"],
        default="all",
        help="Core ML compute units for the native Apple detector",
    )
    parser.add_argument("--vision-max-age", type=float, default=0.15, help="Max age in seconds for helper detections before they are considered stale")
    parser.add_argument("--vision-max-area-ratio", type=float, default=0.45, help="Reject Vision detections whose box covers more than this fraction of the frame")
    parser.add_argument("--vision-max-aspect-ratio", type=float, default=1.8, help="Reject Vision detections more elongated than this width/height ratio")
    parser.add_argument("--vision-min-area-ratio", type=float, default=0.0018, help="Reject Vision detections whose box covers less than this fraction of the frame")
    parser.add_argument(
        "--roi-follow-max-age",
        type=float,
        default=0.45,
        help="When ROI mode is enabled, keep extending detector ROI around the last seen ball for this long (seconds)",
    )
    parser.add_argument(
        "--roi-follow-radius-scale",
        type=float,
        default=3.4,
        help="In ROI mode, expand last-seen-ball radius by this factor to build a follow ROI",
    )
    parser.add_argument(
        "--roi-follow-min-half-size",
        type=int,
        default=90,
        help="Minimum half-size (px) of follow ROI around the last seen ball in ROI mode",
    )
    parser.add_argument(
        "--vision-identity-source",
        default="",
        help=(
            "Optional exact-ball verifier source. Can be a labeled dataset root, "
            "a positives directory, or a saved .npz verifier model."
        ),
    )
    parser.add_argument(
        "--vision-identity-threshold",
        type=float,
        default=0.0,
        help="Optional exact-ball verifier threshold override (<=0 uses the learned default)",
    )
    parser.add_argument("--vision-appearance-dataset", dest="vision_identity_source", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--vision-appearance-threshold", dest="vision_identity_threshold", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--event-source",
        choices=["auto", "trusted", "candidate"],
        default="auto",
        help="Point source for hit logic (auto uses candidate for YOLO, trusted for motion)",
    )
    parser.add_argument(
        "--event-min-consecutive",
        type=int,
        default=1,
        help="Min consecutive detections to allow candidate-based event updates",
    )
    parser.add_argument("--impact-min-speed", type=float, default=450.0, help="Min pre-impact speed in px/s")
    parser.add_argument("--impact-speed-drop-ratio", type=float, default=0.45, help="Speed-after/speed-before ratio threshold")
    parser.add_argument("--impact-dir-change", type=float, default=55.0, help="Min direction change in degrees")
    parser.add_argument("--impact-cooldown", type=float, default=0.5, help="Min seconds between hit events")
    parser.add_argument("--impact-min-displacement", type=float, default=90.0, help="Min 2-frame travel (px) to allow impact")
    parser.add_argument("--impact-enable-entry", dest="impact_enable_entry", action="store_true", help="Count outside->inside goal crossing as hit event")
    parser.add_argument("--no-impact-entry", dest="impact_enable_entry", action="store_false", help="Disable outside->inside crossing events")
    parser.set_defaults(impact_enable_entry=True)
    parser.add_argument("--impact-entry-min-speed", type=float, default=35.0, help="Min speed for entry hit event (px/s)")
    parser.add_argument(
        "--impact-entry-fallbacks",
        dest="impact_entry_fallbacks",
        action="store_true",
        help="Allow permissive entry fallbacks (first-inside, near-plane, recovery) when exact crossing frame is missed",
    )
    parser.add_argument(
        "--no-impact-entry-fallbacks",
        dest="impact_entry_fallbacks",
        action="store_false",
        help="Strict entry mode: require clean outside->inside crossing confirmation only",
    )
    parser.set_defaults(impact_entry_fallbacks=True)
    parser.add_argument(
        "--impact-entry-confirm-frames",
        type=int,
        default=1,
        help="Require N consecutive inside-goal frames before confirming an entry; event location stays at the first inside frame",
    )
    parser.add_argument("--impact-arm-seconds", type=float, default=1.2, help="Ignore impacts during startup/recalibration")
    parser.add_argument(
        "--impact-rearm-outside-ratio",
        type=float,
        default=0.60,
        help="Require the ball center to move this many radii outside the goal opening before another hit can arm",
    )
    parser.add_argument(
        "--impact-rearm-camera-margin-m",
        type=float,
        default=0.14,
        help="Require the ball to return at least this far to the camera side of the goal plane before another hit can arm",
    )
    parser.add_argument(
        "--impact-rearm-miss-seconds",
        type=float,
        default=0.75,
        help="If the ball disappears for this long after a hit, allow the next hit to arm again",
    )
    parser.add_argument(
        "--miss-detect",
        dest="miss_detect",
        action="store_true",
        help="Emit explicit miss events (0 points) when a shot passes near the goal plane without producing a hit event",
    )
    parser.add_argument(
        "--no-miss-detect",
        dest="miss_detect",
        action="store_false",
        help="Disable explicit miss-event generation",
    )
    parser.set_defaults(miss_detect=True)
    parser.add_argument(
        "--miss-near-plane-m",
        type=float,
        default=0.30,
        help="Candidate miss window: ball surface must come within this many meters of the goal plane",
    )
    parser.add_argument(
        "--miss-timeout-s",
        type=float,
        default=0.90,
        help="How long to wait (seconds) before finalizing a near-goal shot as MISS when no hit was emitted",
    )
    parser.add_argument(
        "--miss-cooldown-s",
        type=float,
        default=1.20,
        help="Cooldown after a hit/miss event before a new miss candidate can arm",
    )
    parser.add_argument("--ball-diameter-m", type=float, default=0.22, help="Real ball diameter in meters for 3D plane checks")
    parser.add_argument(
        "--goal-plane-contact-tolerance-m",
        type=float,
        default=0.18,
        help="Extra margin in meters allowed beyond the real ball radius before rejecting a hit",
    )
    parser.add_argument(
        "--goal-pose-alpha",
        type=float,
        default=0.45,
        help="Smoothing factor [0..1] for marker-based projected goal corners",
    )
    parser.add_argument(
        "--goal-pose-max-age",
        type=float,
        default=0.35,
        help="Keep the last solved marker pose alive for this many seconds when tags momentarily disappear",
    )
    parser.add_argument(
        "--goal-pose-every",
        type=int,
        default=1,
        help="Solve marker-based goal pose every N frames and reuse the cached pose in between",
    )
    parser.add_argument(
        "--goal-presence-margin-px",
        type=float,
        default=220.0,
        help="Reject detections whose center is farther than this many pixels outside the goal polygon",
    )
    parser.add_argument(
        "--event-motion-fallback",
        dest="event_motion_fallback",
        action="store_true",
        help="Use a permissive motion detector near the goal when the primary detector misses a moving ball",
    )
    parser.add_argument(
        "--no-event-motion-fallback",
        dest="event_motion_fallback",
        action="store_false",
        help="Disable motion fallback for goal-entry detection",
    )
    parser.set_defaults(event_motion_fallback=True)
    parser.add_argument(
        "--event-motion-after-reject",
        dest="event_motion_after_reject",
        action="store_true",
        help="After YOLO candidate is rejected (goal margin/background), run one extra motion fallback pass in the same frame",
    )
    parser.add_argument(
        "--no-event-motion-after-reject",
        dest="event_motion_after_reject",
        action="store_false",
        help="Do not run extra motion fallback after a rejected YOLO candidate (safer, less permissive)",
    )
    parser.set_defaults(event_motion_after_reject=False)
    parser.add_argument("--candidate-overlay-ttl", type=float, default=0.35, help="How long to keep blue candidate marker visible")
    parser.add_argument("--ball-overlay-ttl", type=float, default=0.45, help="How long to keep ball marker visible after detection")
    parser.add_argument(
        "--ball-overlay-radius-scale",
        type=float,
        default=1.0,
        help="Visual-only trusted-ball ring radius scale (e.g. 1.10 draws the ring 10%% larger)",
    )
    parser.add_argument("--hit-overlay-ttl", type=float, default=0.9, help="How long to keep red HIT text/marker visible")
    parser.add_argument(
        "--minimal-overlay",
        action="store_true",
        help="Show only the trusted ball ring (hide HUD text, goal/panels, candidate ring, bridge, and hit markers)",
    )
    parser.add_argument(
        "--trajectory-bridge",
        dest="trajectory_bridge",
        action="store_true",
        help="Predict a few missed near-goal ball positions so entry events survive short detector dropouts",
    )
    parser.add_argument(
        "--no-trajectory-bridge",
        dest="trajectory_bridge",
        action="store_false",
        help="Disable missed-frame trajectory bridging",
    )
    parser.set_defaults(trajectory_bridge=True)
    parser.add_argument("--trajectory-max-gap-frames", type=int, default=3, help="Max missed frames to bridge with trajectory prediction")
    parser.add_argument("--trajectory-min-speed", type=float, default=140.0, help="Min observed speed (px/s) before trajectory bridging is allowed")
    parser.add_argument("--trajectory-goal-margin", type=float, default=180.0, help="How close the last seen ball must be to the goal to allow bridging")
    parser.add_argument("--track-min-consecutive", type=int, default=3, help="Min consecutive detections before tracking is trusted")
    parser.add_argument("--track-max-step", type=float, default=260.0, help="Max px jump between consecutive detections")
    parser.add_argument("--track-max-misses", type=int, default=6, help="Max missing frames before track reset")
    parser.add_argument("--track-radius-ratio", type=float, default=1.7, help="Max allowed radius change ratio between detections")
    return parser.parse_args()


def parse_camera_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def create_tracker(tracker_name: str):
    if tracker_name == "none":
        return None
    if tracker_name == "mil" and hasattr(cv2, "TrackerMIL_create"):
        return cv2.TrackerMIL_create()
    raise RuntimeError(f"Tracker '{tracker_name}' is not available in this OpenCV build.")


def sanitize_bbox(bbox: tuple[float, float, float, float], frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    h, w = frame_shape[:2]
    x, y, bw, bh = [int(round(float(v))) for v in bbox]
    if bw <= 1 or bh <= 1:
        return None
    x = max(0, min(x, w - 2))
    y = max(0, min(y, h - 2))
    bw = max(2, min(bw, w - x))
    bh = max(2, min(bh, h - y))
    if bw <= 1 or bh <= 1:
        return None
    return (x, y, bw, bh)


def init_tracker_with_bbox(tracker, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> bool:
    bb = sanitize_bbox(bbox, frame.shape)
    if bb is None:
        return False
    try:
        result = tracker.init(frame, bb)
    except cv2.error:
        return False
    # Some OpenCV tracker bindings return None on success.
    if isinstance(result, bool):
        return result
    return True


def ball_from_bbox(bbox: tuple[float, float, float, float]) -> BallDetection:
    x, y, w, h = [float(v) for v in bbox]
    cx = int(x + 0.5 * w)
    cy = int(y + 0.5 * h)
    radius = max(4.0, 0.5 * max(w, h))
    area = max(1.0, w * h)
    return BallDetection(center=(cx, cy), radius=radius, area=area, circularity=1.0)


def tracker_bbox_from_ball(ball: BallDetection, frame_shape: tuple[int, int, int], scale: float = 1.9) -> tuple[float, float, float, float]:
    h, w = frame_shape[:2]
    half = int(max(8.0, ball.radius * max(1.1, scale)))
    x1 = max(0, int(ball.center[0] - half))
    y1 = max(0, int(ball.center[1] - half))
    x2 = min(w, int(ball.center[0] + half))
    y2 = min(h, int(ball.center[1] + half))
    bw = max(2, x2 - x1)
    bh = max(2, y2 - y1)
    return (x1, y1, bw, bh)


def backend_flag(backend: str) -> int | None:
    if backend == "avfoundation":
        return cv2.CAP_AVFOUNDATION
    if backend == "msmf":
        return cv2.CAP_MSMF if hasattr(cv2, "CAP_MSMF") else None
    if backend == "dshow":
        return cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else None
    return None


def open_capture(source: int | str, width: int, height: int, fps: int, backend: str, fourcc: str = "") -> cv2.VideoCapture:
    api = backend_flag(backend)
    if api is None:
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source, api)

    code = fourcc.strip().upper()
    if len(code) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def fourcc_to_text(value: float) -> str:
    v = int(value)
    if v <= 0:
        return "N/A"
    chars = [chr((v >> (8 * i)) & 0xFF) for i in range(4)]
    text = "".join(chars)
    return text if text.strip("\x00") else "N/A"


def print_perf_hints(args: argparse.Namespace) -> None:
    if args.detector not in ("yolo", "hybrid"):
        return

    try:
        import platform
        import torch  # type: ignore
    except Exception:
        platform = None  # type: ignore
        torch = None  # type: ignore

    if platform is not None and torch is not None:
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend) and bool(mps_backend.is_available())
        requested = (args.yolo_device or "").strip().lower()
        if is_apple_silicon and not mps_available and requested in ("", "mps", "cpu"):
            print(
                "[PerfHint] Apple Silicon detected but MPS is unavailable in this Python/Torch environment. "
                "YOLO is running on CPU, so ~14 FPS with yolo26s is expected."
            )

    if args.yolo_model.endswith("yolo26s.pt") and args.yolo_imgsz >= 960 and args.process_every == 1:
        print(
            "[PerfHint] Current settings are heavy for real-time: yolo26s + imgsz>=960 + process_every=1. "
            "If you need ~30 FPS, try yolo26n or imgsz 512/640 or process_every 2."
        )
    if args.yolo_track and args.process_every == 1:
        print(
            "[PerfHint] YOLO track mode adds extra overhead every frame. Disable --yolo-track first when testing raw detector FPS."
        )


def add_perf_sample(
    perf_sums_s: dict[str, float],
    key: str,
    dt_s: float,
) -> None:
    perf_sums_s[key] = perf_sums_s.get(key, 0.0) + max(0.0, float(dt_s))


def yolo_runtime_provider_name(ball_detector: BallDetector) -> str:
    detector_obj = ball_detector
    if isinstance(detector_obj, HybridBallDetector):
        detector_obj = detector_obj.primary
    if not isinstance(detector_obj, YoloBallDetector):
        return "n/a"
    predictor = getattr(detector_obj.model, "predictor", None)
    backend = getattr(predictor, "model", None)
    session = getattr(backend, "session", None)
    if session is None or not hasattr(session, "get_providers"):
        return "unknown"
    providers = session.get_providers()
    return providers[0] if providers else "unknown"


def print_perf_breakdown(
    *,
    perf_sums_s: dict[str, float],
    frames_done: int,
    top_n: int,
    detector_name: str,
    args: argparse.Namespace,
    ball_detector: BallDetector,
) -> None:
    if frames_done <= 0:
        return
    total_s = max(1e-9, perf_sums_s.get("loop_total", 0.0))
    stage_rows: list[tuple[str, float, float, float]] = []
    for key, stage_s in perf_sums_s.items():
        if key == "loop_total":
            continue
        avg_ms = (stage_s * 1000.0) / max(1, frames_done)
        pct = (stage_s / total_s) * 100.0
        stage_rows.append((key, stage_s, avg_ms, pct))
    stage_rows.sort(key=lambda x: x[1], reverse=True)
    keep_n = max(1, int(top_n))
    if not stage_rows:
        return

    detail = ", ".join(
        f"{name}={avg_ms:.2f}ms({pct:.0f}%)"
        for name, _stage_s, avg_ms, pct in stage_rows[:keep_n]
    )
    print(f"[PerfDetail] {detail}")

    if detector_name in ("yolo", "hybrid"):
        provider = yolo_runtime_provider_name(ball_detector)
        print(f"[PerfDetail] yolo_runtime_provider={provider}")

    top_name, _top_stage_s, _top_avg_ms, top_pct = stage_rows[0]
    top_pct = float(top_pct)
    if top_name == "detect_track" and top_pct >= 45.0:
        print(
            "[PerfDetailHint] detector dominates frame time. Try lower --yolo-imgsz, higher --process-every, "
            "and confirm CoreML provider is active (not CPU fallback)."
        )
    elif top_name == "pose_adapt" and top_pct >= 30.0:
        print(
            "[PerfDetailHint] pose/adaptation is heavy. Try higher --goal-pose-every (e.g. 5-8) "
            "and/or disable auto-adapt when camera is fixed."
        )
    elif top_name == "overlay_show" and top_pct >= 25.0:
        print(
            "[PerfDetailHint] overlay/display is heavy. Try --minimal-overlay and/or lower --display-scale."
        )
    elif top_name == "capture_preproc" and top_pct >= 35.0:
        print(
            "[PerfDetailHint] camera read/preprocess dominates. Check camera backend/FPS mode and avoid extra resize/undistort where possible."
        )


def read_ui_key(*, video_controls_enabled: bool, playback_paused: bool) -> int:
    if video_controls_enabled and playback_paused:
        return cv2.waitKey(0) & 0xFF
    poll_key = getattr(cv2, "pollKey", None)
    if callable(poll_key):
        raw = int(poll_key())
        return (raw & 0xFF) if raw >= 0 else 0xFF
    return cv2.waitKey(1) & 0xFF


def maybe_resize_input(frame: np.ndarray, width: int, height: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return frame
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def run_camera_probe(source: int | str, args: argparse.Namespace) -> None:
    if args.backend != "auto":
        backends = [args.backend]
    else:
        system = platform.system()
        if system == "Darwin":
            backends = ["auto", "avfoundation"]
        elif system == "Windows":
            backends = ["dshow", "msmf", "auto"]
        else:
            backends = ["auto"]
    fourccs = [args.fourcc.strip().upper()] if args.fourcc.strip() else ["", "MJPG", "H264", "YUY2"]
    modes = [
        (1920, 1080, 60),
        (1920, 1080, 30),
        (1280, 720, 60),
        (1280, 720, 30),
        (3840, 2160, 30),
    ]

    print("[Probe] testing camera modes...")
    best: tuple[float, str] | None = None

    for backend in backends:
        for fourcc in fourccs:
            for w, h, fps_req in modes:
                cap = open_capture(source, w, h, fps_req, backend=backend, fourcc=fourcc)
                if not cap.isOpened():
                    print(f"[Probe] backend={backend:11} fourcc={fourcc or 'AUTO':4} req={w}x{h}@{fps_req:2} -> open failed")
                    continue

                # Warm up camera settings application.
                for _ in range(8):
                    ok, _ = cap.read()
                    if not ok:
                        break

                start = time.time()
                frames = 0
                while time.time() - start < max(0.8, args.probe_seconds):
                    ok, _ = cap.read()
                    if not ok:
                        break
                    frames += 1
                elapsed = max(1e-6, time.time() - start)
                measured = frames / elapsed

                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                actual_fourcc = fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC))
                cap.release()

                row = (
                    f"[Probe] backend={backend:11} fourcc={fourcc or 'AUTO':4} "
                    f"req={w}x{h}@{fps_req:2} -> actual={actual_w}x{actual_h}@{actual_fps:4.1f} "
                    f"codec={actual_fourcc:4} measured={measured:5.1f}fps"
                )
                print(row)
                if best is None or measured > best[0]:
                    best = (measured, row)

    if best is None:
        print("[Probe] no working mode found.")
    else:
        print(f"[Probe] best: {best[1]}")


def create_ball_detector(args: argparse.Namespace) -> BallDetector:
    if args.detector == "vision":
        if not args.vision_model.strip():
            raise RuntimeError("Vision detector selected but --vision-model was not provided.")
        camera_raw = args.vision_camera.strip() if args.vision_camera.strip() else str(args.camera)
        if not camera_raw.isdigit():
            raise RuntimeError("Vision helper currently supports numeric camera indices only. Use --vision-camera 0 style values.")
        return AppleVisionBallDetector(
            AppleVisionConfig(
                helper_bin=args.vision_helper_bin,
                model_path=args.vision_model,
                label=args.vision_label,
                camera_index=int(camera_raw),
                width=args.width,
                height=args.height,
                fps=args.fps,
                detect_every=args.vision_detect_every,
                confidence=args.vision_confidence,
                compute_units=args.vision_compute_units,
                local_search_scale=args.vision_local_search_scale,
                full_recover_every=args.vision_full_recover_every,
                max_age_s=args.vision_max_age,
                max_area_ratio=args.vision_max_area_ratio,
                max_aspect_ratio=args.vision_max_aspect_ratio,
                min_area_ratio=args.vision_min_area_ratio,
                identity_source=args.vision_identity_source,
                identity_threshold=args.vision_identity_threshold,
            )
        )

    if args.detector in ("yolo", "hybrid"):
        device = args.yolo_device.strip() or None
        yolo_detector: BallDetector = YoloBallDetector(
            YoloConfig(
                model=args.yolo_model,
                conf=args.yolo_conf,
                imgsz=args.yolo_imgsz,
                device=device,
                class_id=args.yolo_class_id,
                use_tracker=args.yolo_track,
                tracker_cfg=args.yolo_tracker,
                identity_source=args.yolo_identity_source,
                identity_threshold=args.yolo_identity_threshold,
            )
        )
        if args.detector == "yolo":
            return yolo_detector

        motion_fallback: BallDetector = MotionBallDetector(
            min_area=args.min_area,
            max_area=args.max_area,
            min_circularity=args.min_circularity,
            process_scale=args.motion_scale,
            warmup_frames=args.motion_warmup_frames,
            enable_static_fallback=args.motion_static_fallback,
            static_every_n=args.motion_static_every,
            static_hough_param2=args.motion_static_param2,
            static_min_radius=args.motion_static_min_radius,
            static_max_radius=args.motion_static_max_radius,
        )
        return HybridBallDetector(primary=yolo_detector, fallback=motion_fallback)

    return MotionBallDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        min_circularity=args.min_circularity,
        process_scale=args.motion_scale,
        warmup_frames=args.motion_warmup_frames,
        enable_static_fallback=args.motion_static_fallback,
        static_every_n=args.motion_static_every,
        static_hough_param2=args.motion_static_param2,
        static_min_radius=args.motion_static_min_radius,
        static_max_radius=args.motion_static_max_radius,
    )


def create_event_motion_fallback(args: argparse.Namespace) -> MotionBallDetector | None:
    if not args.event_motion_fallback or args.detector == "motion":
        return None
    # Looser motion-only detector to catch blurred balls crossing the goal when
    # class-based detection momentarily misses.
    return MotionBallDetector(
        min_area=max(40, args.min_area // 2),
        max_area=max(args.max_area * 2, 12000),
        min_circularity=min(args.min_circularity, 0.18),
        process_scale=max(0.45, min(0.75, args.motion_scale)),
        warmup_frames=max(10, min(args.motion_warmup_frames, 30)),
        enable_static_fallback=False,
    )


def create_impact_detector_for_goal(
    args: argparse.Namespace,
    goal_width_m: float,
    goal_height_m: float,
) -> ImpactDetector:
    return ImpactDetector(
        goal_width_m=goal_width_m,
        goal_height_m=goal_height_m,
        min_pre_impact_speed=args.impact_min_speed,
        speed_drop_ratio=args.impact_speed_drop_ratio,
        min_direction_change_deg=args.impact_dir_change,
        cooldown_s=args.impact_cooldown,
        min_displacement_px=args.impact_min_displacement,
        enable_entry_event=args.impact_enable_entry,
        min_entry_speed_px_s=args.impact_entry_min_speed,
        entry_confirm_frames=args.impact_entry_confirm_frames,
        allow_entry_fallbacks=args.impact_entry_fallbacks,
        rearm_outside_ratio=args.impact_rearm_outside_ratio,
        rearm_camera_margin_m=args.impact_rearm_camera_margin_m,
        rearm_miss_seconds=args.impact_rearm_miss_seconds,
    )


def calibrate_and_persist(
    frame_bgr: np.ndarray,
    calibration_path: Path,
    reference_frame_path: Path,
    goal_width_m: float,
    goal_height_m: float,
) -> CalibrationData:
    corners = calibrate_goal_corners(frame_bgr)
    reference_frame_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(reference_frame_path), frame_bgr)

    data = CalibrationData(
        corners_px=corners.astype(np.float32),
        reference_size=(frame_bgr.shape[1], frame_bgr.shape[0]),
        goal_width_m=goal_width_m,
        goal_height_m=goal_height_m,
        reference_frame_path=str(reference_frame_path),
    )
    save_calibration(calibration_path, data)
    return data


def goal_roi(corners: np.ndarray, frame_shape: tuple[int, int, int], margin: int = 90) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1 = max(0, int(np.floor(np.min(corners[:, 0]))) - margin)
    y1 = max(0, int(np.floor(np.min(corners[:, 1]))) - margin)
    x2 = min(w, int(np.ceil(np.max(corners[:, 0]))) + margin)
    y2 = min(h, int(np.ceil(np.max(corners[:, 1]))) + margin)
    return x1, y1, x2, y2


def parse_fixed_detect_roi(value: str, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise RuntimeError("--fixed-detect-roi must be x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = [int(round(float(p))) for p in parts]
    except ValueError as exc:
        raise RuntimeError("--fixed-detect-roi must contain numeric values") from exc
    h, w = frame_shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        raise RuntimeError("--fixed-detect-roi is invalid after clipping to frame bounds")
    return (x1, y1, x2, y2)


def _candidate_patch(
    image_bgr: np.ndarray,
    center: tuple[int, int],
    radius: float,
    scale: float,
) -> np.ndarray | None:
    h, w = image_bgr.shape[:2]
    half = int(round(max(12.0, float(radius) * max(1.0, float(scale)))))
    cx, cy = int(center[0]), int(center[1])
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = image_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return patch


def candidate_differs_from_background(
    frame_bgr: np.ndarray,
    background_bgr: np.ndarray,
    detection: BallDetection,
    *,
    diff_threshold: float,
    min_active_ratio: float,
    patch_scale: float,
) -> bool:
    patch_now = _candidate_patch(frame_bgr, detection.center, detection.radius, patch_scale)
    patch_bg = _candidate_patch(background_bgr, detection.center, detection.radius, patch_scale)
    if patch_now is None or patch_bg is None:
        return True
    if patch_now.shape[:2] != patch_bg.shape[:2]:
        patch_bg = cv2.resize(
            patch_bg,
            (patch_now.shape[1], patch_now.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    gray_now = cv2.cvtColor(patch_now, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_bg = cv2.cvtColor(patch_bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_now = cv2.GaussianBlur(gray_now, (5, 5), 0)
    gray_bg = cv2.GaussianBlur(gray_bg, (5, 5), 0)
    # Compensate local illumination/contrast drift first; otherwise static wall
    # textures under changing sunlight can look falsely "different."
    now_mean, now_std = cv2.meanStdDev(gray_now)
    bg_mean, bg_std = cv2.meanStdDev(gray_bg)
    now_mean_f = float(now_mean.reshape(-1)[0])
    bg_mean_f = float(bg_mean.reshape(-1)[0])
    now_std_f = max(4.0, float(now_std.reshape(-1)[0]))
    bg_std_f = max(4.0, float(bg_std.reshape(-1)[0]))
    gray_bg_matched = (gray_bg - bg_mean_f) * (now_std_f / bg_std_f) + now_mean_f
    gray_bg_matched = np.clip(gray_bg_matched, 0.0, 255.0)
    diff = np.abs(gray_now - gray_bg_matched)
    pixel_thresh = max(2.0, float(diff_threshold))
    h, w = diff.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    center_y = 0.5 * (h - 1.0)
    center_x = 0.5 * (w - 1.0)
    radius_px = max(4.0, min(h, w) * 0.34)
    core_mask = ((yy - center_y) ** 2 + (xx - center_x) ** 2) <= (radius_px * radius_px)
    if not bool(np.any(core_mask)):
        return True
    core_diff = diff[core_mask]
    active_ratio = float(np.mean(core_diff >= pixel_thresh))
    mean_diff = float(np.mean(core_diff))
    mean_floor = max(3.0, pixel_thresh * 0.30)
    return (active_ratio >= max(0.0, float(min_active_ratio))) and (mean_diff >= mean_floor)


def detection_near_goal(
    ball: BallDetection | None,
    goal_corners: np.ndarray,
    margin_px: float,
) -> bool:
    if ball is None:
        return False
    signed = signed_distance_to_polygon((float(ball.center[0]), float(ball.center[1])), goal_corners)
    # Use the ball radius here, not just the center point, otherwise a real
    # ball can overlap the goal opening while its center is still just outside.
    overlap_margin = max(0.0, float(margin_px)) + max(0.0, float(ball.radius))
    return signed >= -overlap_margin


def detection_outside_goal_opening(
    ball: BallDetection | None,
    goal_corners: np.ndarray,
    *,
    center_inside_margin_px: float = 0.0,
) -> bool:
    if ball is None:
        return False
    signed = signed_distance_to_polygon((float(ball.center[0]), float(ball.center[1])), goal_corners)
    return signed < -max(0.0, float(center_inside_margin_px))


def project_point_to_goal_normalized(
    point_xy: tuple[int, int],
    *,
    goal_homography: np.ndarray,
    pose: GoalPoseEstimate | None,
    intrinsics: CameraIntrinsics | None,
    frame_size: tuple[int, int],
) -> tuple[float, float]:
    if pose is not None and intrinsics is not None:
        projected = project_pixel_to_goal_plane(point_xy, frame_size, pose, intrinsics)
        if projected is not None:
            return (
                float(np.clip(projected[0], 0.0, 1.0)),
                float(np.clip(projected[1], 0.0, 1.0)),
            )
    src = np.array([[point_xy]], dtype=np.float32)
    projected = cv2.perspectiveTransform(src, goal_homography)[0, 0]
    return (
        float(np.clip(projected[0], 0.0, 1.0)),
        float(np.clip(projected[1], 0.0, 1.0)),
    )


def predict_bridge_ball(
    observed_history: deque[tuple[int, float, BallDetection]],
    now_s: float,
    frame_index: int,
    goal_corners: np.ndarray,
    max_gap_frames: int,
    min_speed_px_s: float,
    goal_margin_px: float,
) -> BallDetection | None:
    if len(observed_history) < 2:
        return None

    _, prev_t, prev_ball = observed_history[-2]
    last_frame_idx, last_t, last_ball = observed_history[-1]
    gap_frames = frame_index - last_frame_idx
    if gap_frames <= 0 or gap_frames > max(1, max_gap_frames):
        return None

    dt_obs = max(1e-4, last_t - prev_t)
    dt_pred = now_s - last_t
    if dt_pred <= 0.0 or dt_pred > 0.25:
        return None

    prev_center = np.asarray(prev_ball.center, dtype=np.float32)
    last_center = np.asarray(last_ball.center, dtype=np.float32)
    velocity = (last_center - prev_center) / dt_obs
    speed = float(np.linalg.norm(velocity))
    if speed < max(0.0, min_speed_px_s):
        return None

    signed_last = signed_distance_to_polygon((float(last_center[0]), float(last_center[1])), goal_corners)
    if signed_last < -max(0.0, goal_margin_px):
        return None

    goal_center = np.mean(goal_corners.astype(np.float32), axis=0)
    moving_toward_goal = float(np.dot(velocity, goal_center - last_center)) > (-0.05 * max(speed, 1.0))
    predicted_center = last_center + velocity * dt_pred
    signed_pred = signed_distance_to_polygon((float(predicted_center[0]), float(predicted_center[1])), goal_corners)
    if (not moving_toward_goal) and signed_pred < signed_last:
        return None

    return BallDetection(
        center=(int(round(float(predicted_center[0]))), int(round(float(predicted_center[1])))),
        radius=last_ball.radius,
        area=last_ball.area,
        circularity=last_ball.circularity,
    )


def draw_overlay(
    frame_bgr: np.ndarray,
    corners: np.ndarray,
    candidate_center: tuple[int, int] | None,
    candidate_radius: int | None,
    ball_center: tuple[int, int] | None,
    ball_radius: int | None,
    ball_radius_scale: float,
    bridge_center: tuple[int, int] | None,
    bridge_radius: int | None,
    bridge_active: bool,
    event: ImpactEvent | None,
    recent_hits: list[ImpactEvent],
    fps: float,
    adapt_conf: float,
    auto_adapt: bool,
    detector_name: str,
    flash_strength: float,
    track_consecutive: int,
    track_min_consecutive: int,
    event_source_label: str,
    event_min_consecutive: int,
    tracker_name: str,
    tracker_locked: bool,
    pose_status: str | None = None,
    pose_reprojection_error_px: float | None = None,
    plane_estimate: BallPlaneEstimate | None = None,
    ball_radius_m: float | None = None,
    reject_reason: str | None = None,
    key_hint_override: str | None = None,
    playback_status: str | None = None,
    goal_width_m: float | None = None,
    goal_height_m: float | None = None,
    show_goal_overlay: bool = True,
    minimal_overlay: bool = False,
) -> None:
    if show_goal_overlay and not minimal_overlay:
        polygon = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame_bgr, [polygon], True, (0, 255, 0), 2)

        for idx, corner in enumerate(corners):
            p = (int(corner[0]), int(corner[1]))
            cv2.circle(frame_bgr, p, 5, (0, 200, 255), -1)
            cv2.putText(frame_bgr, str(idx + 1), (p[0] + 7, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # Blue ring shows current detector candidate (cheap visual debugging).
    if (not minimal_overlay) and candidate_center is not None:
        cand_r = max(3, int(candidate_radius or 6))
        cv2.circle(frame_bgr, candidate_center, cand_r, (255, 0, 0), 2)

    # Cyan ring is the validated/trusted ball track.
    if ball_center is not None:
        scaled_radius = float(ball_radius or 8) * max(0.1, float(ball_radius_scale))
        radius = max(4, int(round(scaled_radius)))
        cv2.circle(frame_bgr, ball_center, radius, (255, 255, 0), 2)
        cv2.circle(frame_bgr, ball_center, 2, (0, 255, 255), -1)

    if (not minimal_overlay) and bridge_center is not None:
        radius = max(4, int(bridge_radius or 8))
        cv2.circle(frame_bgr, bridge_center, radius, (0, 165, 255), 2)
        cv2.circle(frame_bgr, bridge_center, 2, (0, 165, 255), -1)

    if minimal_overlay:
        return

    if event is not None:
        event_color = (0, 0, 255)
        if event.event_type == "miss":
            event_color = (0, 165, 255)
        cv2.circle(frame_bgr, event.pixel_point, 14, event_color, 3)
        cv2.drawMarker(frame_bgr, event.pixel_point, event_color, cv2.MARKER_CROSS, 22, 2)
        if event.event_type == "impact":
            prefix = "HIT"
        elif event.event_type == "miss":
            prefix = "MISS"
        else:
            prefix = "GOAL-IN"
        label = f"{prefix} {hit_zone_name(*event.normalized_point)}  x={event.normalized_point[0]:.3f}, y={event.normalized_point[1]:.3f}"
        cv2.putText(frame_bgr, label, (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.72, event_color, 2)

    # Short red flash for immediate visual feedback when impact is detected.
    if flash_strength > 0.0 and event is not None:
        overlay = frame_bgr.copy()
        pulse_radius = int(20 + 90 * flash_strength)
        cv2.circle(overlay, event.pixel_point, pulse_radius, (0, 0, 255), -1)
        alpha = float(np.clip(0.22 * flash_strength, 0.0, 0.22))
        cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0.0, frame_bgr)

    cv2.putText(frame_bgr, f"FPS: {fps:5.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if auto_adapt:
        cv2.putText(
            frame_bgr,
            f"Adapt conf: {adapt_conf:.2f}",
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (120, 255, 120) if adapt_conf >= 0.35 else (0, 170, 255),
            2,
        )
    cv2.putText(frame_bgr, f"Detector: {detector_name}", (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if key_hint_override:
        key_hint = key_hint_override
    elif not show_goal_overlay and tracker_name == "none":
        key_hint = "Keys: q=quit"
    elif tracker_name == "none":
        key_hint = "Keys: c=recalibrate  q=quit"
    else:
        key_hint = "Keys: c=recalibrate  b=lock ball  x=unlock  q=quit"
    cv2.putText(frame_bgr, key_hint, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    trust_y = 154
    if playback_status:
        cv2.putText(frame_bgr, playback_status, (20, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 235, 255), 2)
        trust_y = 178
    track_color = (120, 255, 120) if track_consecutive >= track_min_consecutive else (180, 180, 180)
    cv2.putText(
        frame_bgr,
        f"Trust gate: {track_consecutive}/{track_min_consecutive}",
        (20, trust_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        track_color,
        2,
    )
    cv2.putText(
        frame_bgr,
        f"Event gate: {event_source_label} {max(1, event_min_consecutive)}",
        (20, trust_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (200, 200, 200),
        2,
    )
    tracker_color = (120, 255, 120) if tracker_locked else (180, 180, 180)
    cv2.putText(
        frame_bgr,
        f"Tracker: {tracker_name} {'LOCKED' if tracker_locked else 'idle'}",
        (20, trust_y + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        tracker_color,
        2,
    )
    bridge_color = (0, 165, 255) if bridge_active else (180, 180, 180)
    cv2.putText(
        frame_bgr,
        f"Bridge: {'active' if bridge_active else 'idle'}",
        (20, trust_y + 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        bridge_color,
        2,
    )
    info_y = trust_y + 96
    if show_goal_overlay and pose_status:
        pose_color = (120, 255, 120) if pose_status.lower().startswith("ok") else (0, 200, 255)
        pose_text = f"Pose: {pose_status}"
        if pose_reprojection_error_px is not None:
            pose_text += f" ({pose_reprojection_error_px:.1f}px)"
        cv2.putText(frame_bgr, pose_text, (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, pose_color, 2)
        info_y += 24
    if show_goal_overlay and plane_estimate is not None:
        cv2.putText(
            frame_bgr,
            f"Ball->plane center: {plane_estimate.absolute_distance_m:.2f}m",
            (20, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            2,
        )
        info_y += 24
        cv2.putText(
            frame_bgr,
            f"Ball->plane surface: {plane_estimate.surface_distance_m:.2f}m",
            (20, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            2,
        )
        info_y += 24
    if reject_reason:
        cv2.putText(
            frame_bgr,
            reject_reason,
            (20, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
        )

    if not show_goal_overlay:
        return

    # Goal hit map panel.
    h, w = frame_bgr.shape[:2]
    goal_width = float(goal_width_m or 7.32)
    goal_height = float(goal_height_m or 2.44)
    goal_aspect = float(np.clip(goal_width / max(1e-6, goal_height), 0.25, 6.0))
    max_panel_w = min(320, max(220, w // 4))
    max_panel_h = max(140, min(int(h * 0.50), h - 410))
    panel_w = max_panel_w
    panel_h = int(round(panel_w / goal_aspect))
    if panel_h > max_panel_h:
        panel_h = max_panel_h
        panel_w = int(round(panel_h * goal_aspect))
    panel_w = int(np.clip(panel_w, 160, max_panel_w))
    panel_h = int(np.clip(panel_h, 110, max_panel_h))
    panel_x1 = w - panel_w - 20
    panel_y1 = 20
    panel_x2 = panel_x1 + panel_w
    panel_y2 = panel_y1 + panel_h

    cv2.rectangle(frame_bgr, (panel_x1 - 8, panel_y1 - 26), (panel_x2 + 8, panel_y2 + 76), (20, 20, 20), -1)
    cv2.rectangle(frame_bgr, (panel_x1, panel_y1), (panel_x2, panel_y2), (240, 240, 240), 2)

    # 3x3 guide grid.
    for frac in (1.0 / 3.0, 2.0 / 3.0):
        x = int(panel_x1 + frac * panel_w)
        y = int(panel_y1 + frac * panel_h)
        cv2.line(frame_bgr, (x, panel_y1), (x, panel_y2), (110, 110, 110), 1)
        cv2.line(frame_bgr, (panel_x1, y), (panel_x2, y), (110, 110, 110), 1)

    cv2.putText(frame_bgr, "Goal Hit Map", (panel_x1, panel_y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    # Draw recent hits with fading color (old -> dim, latest -> strong).
    if recent_hits:
        count = len(recent_hits)
        for idx, ev in enumerate(recent_hits):
            nx, ny = ev.normalized_point
            px = int(panel_x1 + np.clip(nx, 0.0, 1.0) * panel_w)
            py = int(panel_y1 + np.clip(ny, 0.0, 1.0) * panel_h)
            age = (idx + 1) / max(1, count)
            if ev.event_type == "miss":
                color = (0, int(120 + 120 * age), 255)
            else:
                color = (40, int(90 + 160 * age), int(80 + 170 * age))
            radius = 3 if idx < count - 1 else 6
            cv2.circle(frame_bgr, (px, py), radius, color, -1)
            if idx == count - 1:
                outline = (0, 165, 255) if ev.event_type == "miss" else (0, 0, 255)
                cv2.circle(frame_bgr, (px, py), 10, outline, 2)

        last = recent_hits[-1]
        zone = hit_zone_name(*last.normalized_point)
        age_s = max(0.0, time.time() - last.timestamp)
        cv2.putText(
            frame_bgr,
            f"Last: {last.event_type.upper()} {zone} ({age_s:.1f}s ago)",
            (panel_x1, panel_y2 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame_bgr,
            f"x={last.normalized_point[0]:.3f} y={last.normalized_point[1]:.3f}",
            (panel_x1, panel_y2 + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (235, 235, 235),
            1,
        )
        cv2.putText(
            frame_bgr,
            f"{last.meters_point[0]:.2f}m , {last.meters_point[1]:.2f}m",
            (panel_x1, panel_y2 + 63),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (235, 235, 235),
            1,
        )
    else:
        cv2.putText(
            frame_bgr,
            "No hits yet",
            (panel_x1 + 10, panel_y1 + panel_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            2,
        )

    if plane_estimate is None:
        return

    depth_panel_w = panel_w
    depth_panel_h = 168
    depth_panel_x1 = panel_x1
    depth_panel_y1 = panel_y2 + 96
    if depth_panel_y1 + depth_panel_h + 20 > h:
        depth_panel_y1 = max(20, h - depth_panel_h - 20)
    depth_panel_x2 = depth_panel_x1 + depth_panel_w
    depth_panel_y2 = depth_panel_y1 + depth_panel_h

    cv2.rectangle(
        frame_bgr,
        (depth_panel_x1 - 8, depth_panel_y1 - 26),
        (depth_panel_x2 + 8, depth_panel_y2 + 8),
        (20, 20, 20),
        -1,
    )
    cv2.rectangle(
        frame_bgr,
        (depth_panel_x1, depth_panel_y1),
        (depth_panel_x2, depth_panel_y2),
        (240, 240, 240),
        2,
    )
    cv2.putText(
        frame_bgr,
        "3D Plane Debug",
        (depth_panel_x1, depth_panel_y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )

    axis_margin_x = 28
    axis_y = depth_panel_y1 + 82
    plane_x = depth_panel_x1 + (depth_panel_w // 2)
    axis_left = depth_panel_x1 + axis_margin_x
    axis_right = depth_panel_x2 - axis_margin_x
    axis_half_w = max(20, (axis_right - axis_left) // 2)

    camera_side_sign = 1.0 if plane_estimate.camera_signed_distance_m >= 0.0 else -1.0
    oriented_center_m = plane_estimate.signed_distance_m * camera_side_sign
    oriented_surface_m = plane_estimate.signed_surface_distance_m * camera_side_sign
    display_ball_radius_m = max(0.01, float(ball_radius_m or 0.11))
    depth_range_m = max(
        0.35,
        min(
            3.0,
            max(abs(oriented_center_m), abs(oriented_surface_m), display_ball_radius_m + 0.05) + 0.20,
        ),
    )

    cv2.line(frame_bgr, (axis_left, axis_y), (axis_right, axis_y), (95, 95, 95), 1)
    cv2.line(frame_bgr, (plane_x, depth_panel_y1 + 26), (plane_x, depth_panel_y2 - 24), (0, 255, 0), 2)
    cv2.putText(
        frame_bgr,
        "goal plane",
        (plane_x - 36, depth_panel_y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (120, 255, 120),
        1,
    )
    cv2.putText(
        frame_bgr,
        "camera side",
        (axis_left, depth_panel_y2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (235, 235, 235),
        1,
    )
    cv2.putText(
        frame_bgr,
        "through goal",
        (plane_x + 12, depth_panel_y2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (235, 235, 235),
        1,
    )

    ball_center_x = int(round(plane_x - (oriented_center_m / depth_range_m) * axis_half_w))
    ball_center_x = int(np.clip(ball_center_x, axis_left + 10, axis_right - 10))
    ball_center_y = axis_y
    ball_radius_px_panel = max(8, int(round((display_ball_radius_m / depth_range_m) * axis_half_w)))

    surface_x = int(round(plane_x - (oriented_surface_m / depth_range_m) * axis_half_w))
    surface_x = int(np.clip(surface_x, axis_left, axis_right))
    cv2.line(frame_bgr, (surface_x, axis_y), (plane_x, axis_y), (255, 255, 0), 1)
    cv2.circle(frame_bgr, (ball_center_x, ball_center_y), ball_radius_px_panel, (255, 255, 0), 2)
    cv2.circle(frame_bgr, (ball_center_x, ball_center_y), 2, (0, 255, 255), -1)

    if plane_estimate.surface_distance_m <= 0.01:
        plane_state = "touching plane"
        plane_color = (120, 255, 120)
    elif oriented_surface_m > 0.0:
        plane_state = "camera side of goal"
        plane_color = (0, 200, 255)
    else:
        plane_state = "through the goal plane"
        plane_color = (120, 255, 120)

    cv2.putText(
        frame_bgr,
        f"State: {plane_state}",
        (depth_panel_x1 + 12, depth_panel_y1 + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        plane_color,
        2,
    )
    cv2.putText(
        frame_bgr,
        f"center {plane_estimate.absolute_distance_m:.2f}m",
        (depth_panel_x1 + 12, depth_panel_y1 + depth_panel_h - 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
    )
    cv2.putText(
        frame_bgr,
        f"surface {plane_estimate.surface_distance_m:.2f}m",
        (depth_panel_x1 + 12, depth_panel_y1 + depth_panel_h - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
    )


def append_hit_event(log_path: Path, event: ImpactEvent) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "utc_timestamp",
                    "frame",
                    "event_type",
                    "pixel_x",
                    "pixel_y",
                    "norm_x",
                    "norm_y",
                    "meter_x",
                    "meter_y",
                    "speed_before_px_s",
                    "speed_after_px_s",
                    "angle_change_deg",
                ]
            )
        writer.writerow(
            [
                datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat(),
                event.frame_index,
                event.event_type,
                event.pixel_point[0],
                event.pixel_point[1],
                f"{event.normalized_point[0]:.6f}",
                f"{event.normalized_point[1]:.6f}",
                f"{event.meters_point[0]:.3f}",
                f"{event.meters_point[1]:.3f}",
                f"{event.speed_before:.2f}",
                f"{event.speed_after:.2f}",
                f"{event.angle_change_deg:.2f}",
            ]
        )


def load_or_create_calibration(
    frame_bgr: np.ndarray,
    args: argparse.Namespace,
    calibration_path: Path,
    reference_frame_path: Path,
) -> CalibrationData:
    if calibration_path.exists():
        data = load_calibration(calibration_path)
        frame_size = (frame_bgr.shape[1], frame_bgr.shape[0])
        if data.reference_size != frame_size:
            data.corners_px = scale_corners(data.corners_px, data.reference_size, frame_size)
            data.reference_size = frame_size
        if args.goal_width_m > 0:
            data.goal_width_m = args.goal_width_m
        if args.goal_height_m > 0:
            data.goal_height_m = args.goal_height_m
        return data

    return calibrate_and_persist(
        frame_bgr=frame_bgr,
        calibration_path=calibration_path,
        reference_frame_path=reference_frame_path,
        goal_width_m=args.goal_width_m,
        goal_height_m=args.goal_height_m,
    )


def read_reference_frame(path: Path, fallback: np.ndarray) -> np.ndarray:
    if not path.exists():
        return fallback.copy()
    frame = cv2.imread(str(path))
    if frame is None:
        return fallback.copy()
    if frame.shape[:2] != fallback.shape[:2]:
        frame = cv2.resize(frame, (fallback.shape[1], fallback.shape[0]), interpolation=cv2.INTER_LINEAR)
    return frame


def main() -> None:
    args = parse_args()
    cv2.setUseOptimized(True)
    source = parse_camera_source(args.camera)

    if args.probe_camera:
        run_camera_probe(source, args)
        return

    calibration_path = Path(args.calibration_file)
    reference_frame_path = Path(args.reference_frame)
    log_path = Path(args.log_file)
    camera_intrinsics: CameraIntrinsics | None = None
    goal_marker_layout: GoalMarkerLayout | None = None
    if (not args.ball_only_mode) and args.camera_calibration_file:
        intrinsics_path = Path(args.camera_calibration_file)
        if not intrinsics_path.exists():
            raise RuntimeError(f"Camera calibration file not found: {intrinsics_path}")
        camera_intrinsics = load_camera_intrinsics(intrinsics_path)
    if (not args.ball_only_mode) and args.goal_markers_layout:
        layout_path = Path(args.goal_markers_layout)
        if not layout_path.exists():
            raise RuntimeError(f"Goal marker layout file not found: {layout_path}")
        if camera_intrinsics is None:
            raise RuntimeError("Marker-based goal pose requires --camera-calibration-file.")
        goal_marker_layout = load_goal_marker_layout(layout_path)

    use_undistort_input = bool(args.undistort_input)
    if goal_marker_layout is not None and use_undistort_input:
        use_undistort_input = False

    cap: cv2.VideoCapture | None = None
    ball_detector = create_ball_detector(args)
    if args.detector == "vision":
        if not isinstance(ball_detector, AppleVisionBallDetector):
            raise RuntimeError("Vision detector setup failed.")
        if args.undistort_input and camera_intrinsics is not None:
            print("[Camera] note: --undistort-input is not applied in native vision mode yet.")
        frame = ball_detector.read_frame(timeout_s=5.0)
        if frame is None:
            raise RuntimeError("Apple Vision helper did not deliver an initial frame.")
    else:
        cap = open_capture(source, args.width, args.height, args.fps, backend=args.backend, fourcc=args.fourcc)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera source.")

        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read first frame from camera.")
        if use_undistort_input and camera_intrinsics is not None:
            frame = undistort_frame(frame, camera_intrinsics)
        frame = maybe_resize_input(frame, args.width, args.height, enabled=args.force_resize_input)
    initial_pose: GoalPoseEstimate | None = None
    initial_pose_status: str | None = None
    if goal_marker_layout is not None and camera_intrinsics is not None:
        initial_pose, _marker_corners, _marker_ids, pose_debug = solve_goal_pose(frame, goal_marker_layout, camera_intrinsics)
        initial_pose_status = pose_debug.status

    if args.ball_only_mode:
        frame_h, frame_w = frame.shape[:2]
        calibration = CalibrationData(
            corners_px=np.asarray(
                [
                    [0.0, 0.0],
                    [float(frame_w - 1), 0.0],
                    [float(frame_w - 1), float(frame_h - 1)],
                    [0.0, float(frame_h - 1)],
                ],
                dtype=np.float32,
            ),
            reference_size=(frame_w, frame_h),
            goal_width_m=args.goal_width_m,
            goal_height_m=args.goal_height_m,
            reference_frame_path=str(reference_frame_path),
        )
    elif initial_pose is not None and goal_marker_layout is not None:
        calibration = CalibrationData(
            corners_px=initial_pose.goal_corners_px.astype(np.float32),
            reference_size=(frame.shape[1], frame.shape[0]),
            goal_width_m=goal_marker_layout.opening_width_m,
            goal_height_m=goal_marker_layout.opening_height_m,
            reference_frame_path=str(reference_frame_path),
        )
    else:
        calibration = load_or_create_calibration(frame, args, calibration_path, reference_frame_path)
        if goal_marker_layout is not None:
            calibration.goal_width_m = goal_marker_layout.opening_width_m
            calibration.goal_height_m = goal_marker_layout.opening_height_m
    current_corners = calibration.corners_px.astype(np.float32)
    calibration_ref_path = (
        Path(calibration.reference_frame_path)
        if calibration.reference_frame_path
        else reference_frame_path
    )
    reference_frame = read_reference_frame(calibration_ref_path, frame)
    adapter = CameraAdapter(reference_frame, current_corners, process_scale=args.adapt_scale)
    background_reference: np.ndarray | None = None
    if args.background_gate:
        background_path_raw = str(args.background_reference or "").strip()
        if background_path_raw:
            background_path = Path(background_path_raw)
            if not background_path.exists():
                raise RuntimeError(f"Background reference image not found: {background_path}")
            loaded = cv2.imread(str(background_path))
            if loaded is None:
                raise RuntimeError(f"Could not read background reference image: {background_path}")
            if loaded.shape[:2] != frame.shape[:2]:
                loaded = cv2.resize(
                    loaded,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            background_reference = loaded
        else:
            background_reference = reference_frame.copy()
    latest_goal_pose: GoalPoseEstimate | None = initial_pose
    latest_goal_pose_time = time.time() if initial_pose is not None else 0.0

    event_motion_detector = create_event_motion_fallback(args)
    effective_goal_width_m = goal_marker_layout.opening_width_m if goal_marker_layout is not None else calibration.goal_width_m
    effective_goal_height_m = goal_marker_layout.opening_height_m if goal_marker_layout is not None else calibration.goal_height_m
    impact_detector = create_impact_detector_for_goal(args, effective_goal_width_m, effective_goal_height_m)

    requested_fps = args.fps
    if cap is not None:
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fourcc = fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC))
        backend_name = cap.getBackendName() if hasattr(cap, "getBackendName") else args.backend
    else:
        actual_fps = float(args.fps)
        actual_w = int(frame.shape[1])
        actual_h = int(frame.shape[0])
        actual_fourcc = "N/A"
        backend_name = "BallVisionHelper"
    video_controls_enabled = cap is not None and isinstance(source, str) and Path(str(source)).exists()
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if video_controls_enabled and cap is not None else 0
    async_capture: AsyncLatestCapture | None = None
    async_capture_seq = 0
    if cap is not None and bool(args.async_capture) and not video_controls_enabled:
        async_capture = AsyncLatestCapture(cap)
        async_capture.seed(frame)
        async_capture.start()
    print(
        f"[Camera] requested={args.width}x{args.height}@{requested_fps} "
        f"actual={actual_w}x{actual_h}@{actual_fps:.1f} backend={backend_name} codec={actual_fourcc}"
    )
    if cap is not None:
        if async_capture is not None:
            print("[Camera] async latest-frame reader enabled for live camera input")
        elif bool(args.async_capture) and video_controls_enabled:
            print("[Camera] async latest-frame reader disabled for video playback controls")
        else:
            print("[Camera] async latest-frame reader disabled by flag")
    print(
        f"[Detect] mode={'full-frame' if args.detect_full_frame else f'goal-roi+{args.detect_roi_margin}px'} "
        f"detector={args.detector}"
    )
    if args.ball_only_mode:
        print("[Detect] ball-only mode enabled (goal gate/event logic/goal overlay disabled)")
    fixed_detect_roi = parse_fixed_detect_roi(args.fixed_detect_roi, frame.shape)
    if fixed_detect_roi is not None:
        x1, y1, x2, y2 = fixed_detect_roi
        print(f"[Detect] fixed ROI active: ({x1},{y1})-({x2},{y2})")
    if background_reference is not None:
        source_label = args.background_reference.strip() or str(calibration_ref_path)
        print(f"[Detect] background gate enabled using reference={source_label}")
    if camera_intrinsics is not None:
        intrinsics_size = f"{camera_intrinsics.image_size[0]}x{camera_intrinsics.image_size[1]}"
        print(
            f"[Camera] intrinsics loaded size={intrinsics_size} "
            f"rms={camera_intrinsics.rms_error:.4f}" if camera_intrinsics.rms_error is not None else f"[Camera] intrinsics loaded size={intrinsics_size}"
        )
    if goal_marker_layout is not None:
        print(
            f"[Pose] markers active layout={args.goal_markers_layout} "
            f"outer={goal_marker_layout.goal_width_m:.2f}m x {goal_marker_layout.goal_height_m:.2f}m "
            f"opening={goal_marker_layout.opening_width_m:.2f}m x {goal_marker_layout.opening_height_m:.2f}m "
            f"depth={goal_marker_layout.scoring_plane_depth_m:.2f}m"
        )
        if args.undistort_input:
            print("[Pose] note: --undistort-input is ignored while marker-based goal pose is active.")
        if initial_pose_status is not None:
            print(f"[Pose] startup={initial_pose_status}")
    print_perf_hints(args)
    if args.detector == "vision":
        print(f"[Detect] native helper={args.vision_helper_bin}")
        print(f"[Detect] native model={args.vision_model}")
    if event_motion_detector is not None:
        print("[Detect] motion fallback enabled near the goal when YOLO misses")
    if args.trajectory_bridge:
        print(
            "[Detect] trajectory bridge enabled "
            f"(gap<={max(1, args.trajectory_max_gap_frames)} frames, near-goal margin={args.trajectory_goal_margin:.0f}px)"
        )
    if not args.detect_full_frame:
        print(f"[Detect] global fallback search every {max(0, args.global_search_every)} detector frame(s)")
    if video_controls_enabled:
        print("[Playback] video controls enabled: space=pause/resume, n=next frame while paused")
    print(f"[Tracker] type={args.ball_tracker} auto_lock={args.auto_lock_tracker}")
    print(f"[Event] source={args.event_source} min_consecutive={max(1, args.event_min_consecutive)}")
    if args.miss_detect:
        print(
            "[Miss] enabled "
            f"(near_plane<={max(0.01, float(args.miss_near_plane_m)):.2f}m, "
            f"timeout={max(0.05, float(args.miss_timeout_s)):.2f}s, "
            f"cooldown={max(0.0, float(args.miss_cooldown_s)):.2f}s)"
        )
    if args.force_resize_input and (actual_w != args.width or actual_h != args.height):
        print(f"[Camera] force resize enabled: processing resized frames at {args.width}x{args.height}")
    if args.no_display:
        print("[Display] disabled (--no-display): highest FPS mode; stop with Ctrl+C.")
    elif max(1, int(args.display_every)) > 1:
        print(f"[Display] refreshing preview every {max(1, int(args.display_every))} frames")

    if not args.no_display:
        cv2.namedWindow("Goal Impact Tracker", cv2.WINDOW_NORMAL)
    frame_idx = 0
    last_time = time.time()
    fps = 0.0
    adapt_confidence = 0.0
    latest_event: ImpactEvent | None = None
    last_candidate: BallDetection | None = None
    last_candidate_seen_t = 0.0
    last_ball: BallDetection | None = None
    last_ball_seen_t = 0.0
    last_plane_estimate: BallPlaneEstimate | None = None
    last_plane_seen_t = 0.0
    last_reject_reason = ""
    last_reject_reason_t = 0.0
    track_last_center: tuple[int, int] | None = None
    track_last_radius: float | None = None
    track_consecutive = 0
    track_misses = 0
    ball_tracker = None
    tracker_locked = False
    tracker_failures = 0
    last_event_seen_t = 0.0
    recent_hits: deque[ImpactEvent] = deque(maxlen=18)
    flash_until = 0.0
    impact_armed_at = time.time() + max(0.0, args.impact_arm_seconds)
    stats_last_t = time.time()
    stats_last_frame = 0
    stats_det_runs = 0
    perf_breakdown_enabled = bool(args.perf_breakdown)
    perf_sums_s: dict[str, float] = {}
    stats_pose_solve_calls = 0
    stats_pose_solve_ok = 0
    stats_pose_fail_reasons: dict[str, int] = {}
    pose_fail_backoff = 1
    trajectory_observations: deque[tuple[int, float, BallDetection]] = deque(maxlen=4)
    miss_candidate_active = False
    miss_candidate_started_t = 0.0
    miss_candidate_last_seen_t = 0.0
    miss_candidate_best_surface_m = float("inf")
    miss_candidate_point_xy: tuple[int, int] | None = None
    miss_candidate_frame_idx = 0
    miss_rearm_until = 0.0
    playback_paused = False
    step_video_frame = False

    if args.manual_ball_select_on_start and args.ball_tracker != "none":
        cv2.imshow("Goal Impact Tracker", frame)
        cv2.waitKey(1)
        start_bbox = cv2.selectROI("Goal Impact Tracker", frame, fromCenter=False, showCrosshair=True)
        if start_bbox[2] > 1 and start_bbox[3] > 1:
            tracker = create_tracker(args.ball_tracker)
            if tracker is not None and init_tracker_with_bbox(tracker, frame, start_bbox):
                ball_tracker = tracker
                tracker_locked = True
                tracker_failures = 0
                track_consecutive = max(1, args.track_min_consecutive)
                impact_detector.reset_history()
                trajectory_observations.clear()
                impact_armed_at = time.time() + max(0.0, args.impact_arm_seconds)

    while True:
        loop_perf_t0 = time.perf_counter()
        capture_t0 = loop_perf_t0
        if cap is not None:
            if async_capture is not None:
                ok, next_frame, next_seq = async_capture.read(timeout_s=0.20, after_seq=async_capture_seq)
                if not ok or next_frame is None:
                    break
                if next_seq == async_capture_seq:
                    continue
                async_capture_seq = next_seq
                frame = next_frame
            else:
                ok, frame = cap.read()
                if not ok:
                    break
            if not ok:
                break
            step_video_frame = False
            if use_undistort_input and camera_intrinsics is not None:
                frame = undistort_frame(frame, camera_intrinsics)
            frame = maybe_resize_input(frame, args.width, args.height, enabled=args.force_resize_input)
        else:
            next_frame = ball_detector.read_frame(timeout_s=1.0) if isinstance(ball_detector, AppleVisionBallDetector) else None
            if next_frame is None:
                continue
            frame = next_frame
        if perf_breakdown_enabled:
            add_perf_sample(perf_sums_s, "capture_preproc", time.perf_counter() - capture_t0)

        now = time.time()
        dt = max(1e-6, now - last_time)
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
        last_time = now
        frame_idx += 1

        pose_t0 = time.perf_counter()
        active_goal_pose: GoalPoseEstimate | None = None
        pose_status_label: str | None = None
        pose_error_px: float | None = None
        if goal_marker_layout is not None and camera_intrinsics is not None:
            pose_every = max(1, int(args.goal_pose_every))
            effective_pose_every = max(1, pose_every * max(1, int(pose_fail_backoff)))
            should_solve_pose = (
                frame_idx <= 1
                or (frame_idx % effective_pose_every) == 0
            )
            if should_solve_pose:
                stats_pose_solve_calls += 1
                pose_solve_t0 = time.perf_counter()
                pose_estimate, _marker_corners, _marker_ids, pose_debug = solve_goal_pose(frame, goal_marker_layout, camera_intrinsics)
                if perf_breakdown_enabled:
                    add_perf_sample(perf_sums_s, "pose_solve_only", time.perf_counter() - pose_solve_t0)
                if pose_estimate is not None:
                    stats_pose_solve_ok += 1
                    pose_fail_backoff = 1
                    latest_goal_pose = pose_estimate
                    latest_goal_pose_time = now
                    alpha = float(np.clip(args.goal_pose_alpha, 0.0, 1.0))
                    current_corners = (1.0 - alpha) * current_corners + alpha * pose_estimate.goal_corners_px
                    active_goal_pose = pose_estimate
                    pose_status_label = "OK"
                    pose_error_px = pose_estimate.reprojection_error_px
                elif latest_goal_pose is not None and (now - latest_goal_pose_time) <= max(0.0, args.goal_pose_max_age):
                    active_goal_pose = latest_goal_pose
                    pose_status_label = "OK"
                    pose_error_px = latest_goal_pose.reprojection_error_px
                else:
                    pose_status_label = pose_debug.status
                    reason_key = str(pose_debug.status or "unknown pose solve failure").strip()
                    stats_pose_fail_reasons[reason_key] = stats_pose_fail_reasons.get(reason_key, 0) + 1
                    if reason_key.startswith("Need at least 3 markers"):
                        pose_fail_backoff = min(8, max(1, pose_fail_backoff) * 2)
                    else:
                        pose_fail_backoff = min(4, max(1, pose_fail_backoff + 1))
            elif latest_goal_pose is not None and (now - latest_goal_pose_time) <= max(0.0, args.goal_pose_max_age):
                active_goal_pose = latest_goal_pose
                pose_status_label = "OK"
                pose_error_px = latest_goal_pose.reprojection_error_px
            else:
                pose_status_label = "stale"
        elif not args.no_auto_adapt and frame_idx % max(1, args.adapt_every) == 0:
            adaptation = adapter.adapt(frame)
            adapt_confidence = adaptation.confidence
            if adaptation.corners is not None and adaptation.confidence >= args.adapt_min_confidence:
                alpha = float(np.clip(args.adapt_alpha, 0.0, 1.0))
                current_corners = (1 - alpha) * current_corners + alpha * adaptation.corners

        try:
            goal_homography = build_goal_homography(current_corners)
        except cv2.error:
            goal_homography = np.eye(3, dtype=np.float32)
        if perf_breakdown_enabled:
            add_perf_sample(perf_sums_s, "pose_adapt", time.perf_counter() - pose_t0)

        detect_track_t0 = time.perf_counter()
        roi = fixed_detect_roi
        if roi is None and not args.detect_full_frame:
            roi = goal_roi(current_corners, frame.shape, margin=max(0, int(args.detect_roi_margin)))
        tracker_candidate: BallDetection | None = None
        if tracker_locked and ball_tracker is not None:
            ok_track, bbox = ball_tracker.update(frame)
            if ok_track and bbox[2] > 1 and bbox[3] > 1:
                tracker_candidate = ball_from_bbox(bbox)
                tracker_failures = 0
            else:
                tracker_failures += 1
                if tracker_failures > max(0, args.tracker_fail_max):
                    tracker_locked = False
                    ball_tracker = None
                    tracker_failures = 0
                    impact_detector.reset_history()
                    trajectory_observations.clear()

        run_detector = (tracker_candidate is None) and (frame_idx % max(1, args.process_every) == 0)
        detected_candidate: BallDetection | None = None
        motion_event_candidate: BallDetection | None = None
        if run_detector:
            search_roi = roi
            if not args.detect_full_frame and fixed_detect_roi is None:
                follow_roi = None
                if (
                    last_candidate is not None
                    and (now - last_candidate_seen_t) <= max(0.0, float(args.roi_follow_max_age))
                ):
                    h, w = frame.shape[:2]
                    follow_half = int(
                        max(
                            float(args.roi_follow_min_half_size),
                            float(last_candidate.radius) * max(1.2, float(args.roi_follow_radius_scale)),
                        )
                    )
                    fx1 = max(0, int(last_candidate.center[0]) - follow_half)
                    fy1 = max(0, int(last_candidate.center[1]) - follow_half)
                    fx2 = min(w, int(last_candidate.center[0]) + follow_half)
                    fy2 = min(h, int(last_candidate.center[1]) + follow_half)
                    if fx2 > fx1 and fy2 > fy1:
                        follow_roi = (fx1, fy1, fx2, fy2)
                if follow_roi is not None:
                    if search_roi is None:
                        search_roi = follow_roi
                    else:
                        gx1, gy1, gx2, gy2 = search_roi
                        fx1, fy1, fx2, fy2 = follow_roi
                        search_roi = (
                            min(gx1, fx1),
                            min(gy1, fy1),
                            max(gx2, fx2),
                            max(gy2, fy2),
                        )
                if args.global_search_every > 0 and frame_idx % max(1, args.global_search_every) == 0:
                    search_roi = None
            detected_candidate = ball_detector.detect(frame, roi=search_roi)
            stats_det_runs += 1
            if detected_candidate is None and event_motion_detector is not None:
                motion_roi = roi
                if motion_roi is None:
                    # Even in full-frame detect mode, keep the motion fallback focused near the goal
                    # so we can rescue blurred hand-driven entries without scanning the whole frame.
                    motion_margin = max(int(args.goal_presence_margin_px), int(args.detect_roi_margin))
                    motion_roi = goal_roi(current_corners, frame.shape, margin=motion_margin)
                if motion_roi is not None:
                    motion_event_candidate = event_motion_detector.detect(frame, roi=motion_roi)

        detector_reject_reason = ""
        debug_reason_getter = getattr(ball_detector, "get_debug_reason", None)
        if callable(debug_reason_getter):
            detector_reject_reason = debug_reason_getter()

        candidate_ball = tracker_candidate if tracker_candidate is not None else detected_candidate
        candidate_from_tracker = tracker_candidate is not None
        if (
            (not args.ball_only_mode)
            and candidate_ball is not None
            and not detection_near_goal(candidate_ball, current_corners, args.goal_presence_margin_px)
        ):
            if candidate_from_tracker:
                tracker_failures = max(tracker_failures, args.tracker_fail_max + 1)
            candidate_ball = None
            candidate_from_tracker = False
            detector_reject_reason = "reject: goal margin"
        if candidate_ball is not None and background_reference is not None:
            differs = candidate_differs_from_background(
                frame,
                background_reference,
                candidate_ball,
                diff_threshold=float(args.background_diff_threshold),
                min_active_ratio=float(args.background_active_ratio),
                patch_scale=float(args.background_patch_scale),
            )
            if not differs:
                if candidate_from_tracker:
                    tracker_failures = max(tracker_failures, args.tracker_fail_max + 1)
                candidate_ball = None
                candidate_from_tracker = False
                detector_reject_reason = "reject: background"
        # If YOLO produced a rejectable candidate (e.g. outside goal margin/background),
        # still try motion fallback near the goal in the same frame so entry events are
        # not lost to a single bad class-based pick.
        if (
            run_detector
            and candidate_ball is None
            and motion_event_candidate is None
            and event_motion_detector is not None
            and (not args.ball_only_mode)
            and bool(args.event_motion_after_reject)
        ):
            motion_roi = roi
            if motion_roi is None:
                motion_margin = max(int(args.goal_presence_margin_px), int(args.detect_roi_margin))
                motion_roi = goal_roi(current_corners, frame.shape, margin=motion_margin)
            if motion_roi is not None:
                motion_event_candidate = event_motion_detector.detect(frame, roi=motion_roi)
        if candidate_ball is not None:
            last_candidate = candidate_ball
            last_candidate_seen_t = now
            if not candidate_from_tracker and detector_reject_reason.startswith("reject:"):
                detector_reject_reason = ""

        if detector_reject_reason:
            last_reject_reason = detector_reject_reason
            last_reject_reason_t = now

        trusted_ball: BallDetection | None = None
        if candidate_ball is not None:
            if candidate_from_tracker:
                track_consecutive = max(track_consecutive, max(1, args.track_min_consecutive))
                track_misses = 0
                track_last_center = candidate_ball.center
                track_last_radius = candidate_ball.radius
                trusted_ball = candidate_ball
            else:
                if track_last_center is None or track_last_radius is None:
                    track_consecutive = 1
                else:
                    jump = float(np.linalg.norm(np.asarray(candidate_ball.center, dtype=np.float32) - np.asarray(track_last_center, dtype=np.float32)))
                    ratio = candidate_ball.radius / max(track_last_radius, 1e-4)
                    ratio_ok = (1.0 / max(1.01, args.track_radius_ratio)) <= ratio <= max(1.01, args.track_radius_ratio)
                    jump_ok = jump <= args.track_max_step
                    if jump_ok and ratio_ok:
                        track_consecutive += 1
                    else:
                        track_consecutive = 1
                        impact_detector.reset_history()
                        trajectory_observations.clear()

                track_misses = 0
                track_last_center = candidate_ball.center
                track_last_radius = candidate_ball.radius

                if track_consecutive >= max(1, args.track_min_consecutive):
                    trusted_ball = candidate_ball
        else:
            track_misses += 1
            if track_misses > max(0, args.track_max_misses):
                track_last_center = None
                track_last_radius = None
                track_consecutive = 0
                impact_detector.reset_history()
                trajectory_observations.clear()

        if trusted_ball is not None:
            last_ball = trusted_ball
            last_ball_seen_t = now
            if (
                not tracker_locked
                and args.auto_lock_tracker
                and args.ball_tracker != "none"
                and not candidate_from_tracker
            ):
                tracker = create_tracker(args.ball_tracker)
                if tracker is not None:
                    init_bbox = tracker_bbox_from_ball(trusted_ball, frame.shape, scale=args.tracker_lock_scale)
                    if init_tracker_with_bbox(tracker, frame, init_bbox):
                        ball_tracker = tracker
                        tracker_locked = True
                        tracker_failures = 0
                        impact_detector.reset_history()
                        trajectory_observations.clear()
        if perf_breakdown_enabled:
            add_perf_sample(perf_sums_s, "detect_track", time.perf_counter() - detect_track_t0)

        event_t0 = time.perf_counter()
        event = None
        bridge_ball: BallDetection | None = None
        event_ball: BallDetection | None = trusted_ball
        plane_estimate: BallPlaneEstimate | None = None
        plane_signed_distance_m: float | None = None
        marker_pose_required_for_events = goal_marker_layout is not None and camera_intrinsics is not None
        ball_radius_m_world = max(0.01, float(args.ball_diameter_m) * 0.5)
        if not args.ball_only_mode:
            use_candidate_for_events = args.event_source == "candidate" or (
                args.event_source == "auto" and args.detector in ("yolo", "hybrid")
            )
            if use_candidate_for_events:
                if candidate_ball is not None and (candidate_from_tracker or track_consecutive >= max(1, args.event_min_consecutive)):
                    event_ball = candidate_ball
                elif motion_event_candidate is not None:
                    event_ball = motion_event_candidate
                else:
                    event_ball = None
            if event_ball is not None:
                trajectory_observations.append((frame_idx, now, event_ball))
            elif args.trajectory_bridge:
                bridge_ball = predict_bridge_ball(
                    observed_history=trajectory_observations,
                    now_s=now,
                    frame_index=frame_idx,
                    goal_corners=current_corners,
                    max_gap_frames=args.trajectory_max_gap_frames,
                    min_speed_px_s=args.trajectory_min_speed,
                    goal_margin_px=args.trajectory_goal_margin,
                )
                if bridge_ball is not None:
                    event_ball = bridge_ball
            if (
                event_ball is not None
                and active_goal_pose is not None
                and camera_intrinsics is not None
            ):
                plane_estimate = estimate_ball_plane_distance(
                    center_px=event_ball.center,
                    ball_radius_px=event_ball.radius,
                    ball_radius_m=ball_radius_m_world,
                    frame_size=(frame.shape[1], frame.shape[0]),
                    pose=active_goal_pose,
                    intrinsics=camera_intrinsics,
                )
                if plane_estimate is not None:
                    plane_signed_distance_m = plane_estimate.signed_distance_m
                    last_plane_estimate = plane_estimate
                    last_plane_seen_t = now
            event_geometry_ready = True
            if marker_pose_required_for_events:
                if active_goal_pose is None:
                    event_geometry_ready = False
                    if not detector_reject_reason:
                        detector_reject_reason = "reject: pose"
                elif event_ball is not None and plane_estimate is None:
                    event_geometry_ready = False
                    if not detector_reject_reason:
                        detector_reject_reason = "reject: plane"
            if now >= impact_armed_at and event_geometry_ready:
                event = impact_detector.update(
                    center_px=event_ball.center if event_ball else None,
                    ball_radius_px=event_ball.radius if event_ball else None,
                    plane_signed_distance_m=plane_signed_distance_m,
                    camera_signed_distance_m=plane_estimate.camera_signed_distance_m if plane_estimate is not None else None,
                    ball_radius_m=ball_radius_m_world,
                    plane_contact_tolerance_m=float(args.goal_plane_contact_tolerance_m),
                    now_s=now,
                    frame_index=frame_idx,
                    goal_corners=current_corners,
                    goal_homography=goal_homography,
                    pose=active_goal_pose,
                    intrinsics=camera_intrinsics,
                    frame_size=(frame.shape[1], frame.shape[0]),
                )
            if args.miss_detect:
                near_goal_for_miss = detection_near_goal(
                    event_ball,
                    current_corners,
                    args.goal_presence_margin_px,
                )
                outside_opening_for_miss = detection_outside_goal_opening(
                    event_ball,
                    current_corners,
                    center_inside_margin_px=0.0,
                )
                surface_distance_m = (
                    float(plane_estimate.surface_distance_m)
                    if plane_estimate is not None
                    else None
                )
                near_plane_for_miss = (
                    surface_distance_m is not None
                    and surface_distance_m <= max(0.01, float(args.miss_near_plane_m))
                )
                through_goal_plane_for_miss = (
                    surface_distance_m is not None
                    and surface_distance_m <= 0.0
                )
                timeout_s = max(0.05, float(args.miss_timeout_s))
                stale_window_s = max(0.12, min(0.35, timeout_s * 0.35))

                if event is not None:
                    miss_candidate_active = False
                    miss_candidate_point_xy = None
                    miss_candidate_best_surface_m = float("inf")
                    miss_rearm_until = now + max(0.0, float(args.miss_cooldown_s))
                else:
                    # Guardrail: MISS is only valid for shots that stayed outside
                    # the opening. If the ball is in/through the goal, suppress MISS.
                    if miss_candidate_active and (
                        (not outside_opening_for_miss) or through_goal_plane_for_miss
                    ):
                        miss_candidate_active = False
                        miss_candidate_point_xy = None
                        miss_candidate_best_surface_m = float("inf")
                        miss_rearm_until = now + max(0.0, float(args.miss_cooldown_s))

                    if (
                        near_goal_for_miss
                        and near_plane_for_miss
                        and outside_opening_for_miss
                        and (not through_goal_plane_for_miss)
                        and event_ball is not None
                        and now >= miss_rearm_until
                    ):
                        if not miss_candidate_active:
                            miss_candidate_active = True
                            miss_candidate_started_t = now
                            miss_candidate_last_seen_t = now
                            miss_candidate_best_surface_m = (
                                float(surface_distance_m)
                                if surface_distance_m is not None
                                else 1e9
                            )
                            miss_candidate_point_xy = event_ball.center
                            miss_candidate_frame_idx = frame_idx
                        else:
                            miss_candidate_last_seen_t = now
                            if surface_distance_m is not None and float(surface_distance_m) <= miss_candidate_best_surface_m:
                                miss_candidate_best_surface_m = float(surface_distance_m)
                                miss_candidate_point_xy = event_ball.center
                                miss_candidate_frame_idx = frame_idx

                    if miss_candidate_active and (now - miss_candidate_started_t) >= timeout_s:
                        stale_enough = (now - miss_candidate_last_seen_t) >= stale_window_s
                        if (not (near_goal_for_miss and near_plane_for_miss)) or stale_enough:
                            point_xy = miss_candidate_point_xy
                            if point_xy is not None:
                                # Final safety check: only emit MISS if the saved
                                # candidate point is still outside the opening.
                                point_signed = signed_distance_to_polygon(
                                    (float(point_xy[0]), float(point_xy[1])),
                                    current_corners,
                                )
                                if point_signed >= 0.0:
                                    miss_candidate_active = False
                                    miss_candidate_point_xy = None
                                    miss_candidate_best_surface_m = float("inf")
                                    miss_rearm_until = now + max(0.0, float(args.miss_cooldown_s))
                                    point_xy = None
                            if point_xy is not None:
                                nx, ny = project_point_to_goal_normalized(
                                    point_xy,
                                    goal_homography=goal_homography,
                                    pose=active_goal_pose,
                                    intrinsics=camera_intrinsics,
                                    frame_size=(frame.shape[1], frame.shape[0]),
                                )
                                event = ImpactEvent(
                                    timestamp=now,
                                    frame_index=int(miss_candidate_frame_idx),
                                    event_type="miss",
                                    pixel_point=point_xy,
                                    normalized_point=(nx, ny),
                                    meters_point=(
                                        float(nx * calibration.goal_width_m),
                                        float(ny * calibration.goal_height_m),
                                    ),
                                    speed_before=0.0,
                                    speed_after=0.0,
                                    angle_change_deg=0.0,
                                )
                            miss_candidate_active = False
                            miss_candidate_point_xy = None
                            miss_candidate_best_surface_m = float("inf")
                            miss_rearm_until = now + max(0.0, float(args.miss_cooldown_s))
        if event is not None:
            latest_event = event
            last_event_seen_t = now
            recent_hits.append(event)
            flash_until = now + 0.35
            append_hit_event(log_path, event)
        if perf_breakdown_enabled:
            add_perf_sample(perf_sums_s, "event_logic", time.perf_counter() - event_t0)

        overlay_t0 = time.perf_counter()
        flash_strength = max(0.0, min(1.0, (flash_until - now) / 0.35))
        if video_controls_enabled:
            # In prerecorded-video review mode, show exact state for the current frame instead of
            # wall-clock TTL persistence. Otherwise a valid detection from a previous frame can
            # visually "stick" to later frames and look like a false positive while debugging.
            candidate_for_overlay = candidate_ball
            ball_for_overlay = trusted_ball
            event_for_overlay = event
            reject_reason_for_overlay = detector_reject_reason
        else:
            candidate_for_overlay = (
                last_candidate if (now - last_candidate_seen_t) <= max(0.0, args.candidate_overlay_ttl) else None
            )
            ball_for_overlay = last_ball if (now - last_ball_seen_t) <= max(0.0, args.ball_overlay_ttl) else None
            event_for_overlay = latest_event if (now - last_event_seen_t) <= max(0.0, args.hit_overlay_ttl) else None
            plane_estimate = (
                plane_estimate
                if plane_estimate is not None
                else (
                    last_plane_estimate
                    if (now - last_plane_seen_t) <= max(0.0, args.ball_overlay_ttl)
                    else None
                )
            )
            reject_reason_for_overlay = (
                last_reject_reason if (now - last_reject_reason_t) <= 0.9 else ""
            )
        playback_status = None
        key_hint_override = None
        if video_controls_enabled:
            playback_state = "paused" if playback_paused else "playing"
            playback_status = f"Playback: {playback_state}  frame {frame_idx}"
            if video_total_frames > 0:
                playback_status += f"/{video_total_frames}"
            if actual_fps > 0.0:
                playback_status += f"  t={(max(0, frame_idx - 1) / actual_fps):.2f}s"
            if args.ball_only_mode:
                key_hint_override = "Keys: space=pause/resume  n=next frame  q=quit"
            elif args.ball_tracker == "none":
                key_hint_override = "Keys: space=pause/resume  n=next frame  c=recalibrate  q=quit"
            else:
                key_hint_override = "Keys: space=pause/resume  n=next  c=recalibrate  b=lock  x=unlock  q=quit"

        draw_overlay(
            frame_bgr=frame,
            corners=current_corners,
            candidate_center=candidate_for_overlay.center if candidate_for_overlay else None,
            candidate_radius=int(candidate_for_overlay.radius) if candidate_for_overlay else None,
            ball_center=ball_for_overlay.center if ball_for_overlay else None,
            ball_radius=int(ball_for_overlay.radius) if ball_for_overlay else None,
            ball_radius_scale=float(args.ball_overlay_radius_scale),
            bridge_center=bridge_ball.center if bridge_ball else None,
            bridge_radius=int(bridge_ball.radius) if bridge_ball else None,
            bridge_active=bridge_ball is not None,
            event=event_for_overlay,
            recent_hits=list(recent_hits),
            fps=fps,
            adapt_conf=adapt_confidence,
            auto_adapt=not args.no_auto_adapt,
            detector_name=args.detector,
            flash_strength=flash_strength,
            track_consecutive=track_consecutive,
            track_min_consecutive=max(1, args.track_min_consecutive),
            event_source_label=args.event_source,
            event_min_consecutive=max(1, args.event_min_consecutive),
            tracker_name=args.ball_tracker,
            tracker_locked=tracker_locked,
            pose_status=pose_status_label,
            pose_reprojection_error_px=pose_error_px,
            plane_estimate=plane_estimate,
            ball_radius_m=ball_radius_m_world,
            reject_reason=reject_reason_for_overlay,
            key_hint_override=key_hint_override,
            playback_status=playback_status,
            goal_width_m=effective_goal_width_m,
            goal_height_m=effective_goal_height_m,
            show_goal_overlay=not args.ball_only_mode,
            minimal_overlay=args.minimal_overlay,
        )

        display_every = max(1, int(args.display_every))
        should_refresh_display = (not args.no_display) and ((frame_idx % display_every) == 0)
        if should_refresh_display:
            if roi is not None:
                x1, y1, x2, y2 = roi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
            if args.display_scale < 0.999:
                frame_show = cv2.resize(
                    frame,
                    None,
                    fx=args.display_scale,
                    fy=args.display_scale,
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                frame_show = frame
            cv2.imshow("Goal Impact Tracker", frame_show)
        if perf_breakdown_enabled:
            add_perf_sample(perf_sums_s, "overlay_show", time.perf_counter() - overlay_t0)

        input_t0 = time.perf_counter()
        should_quit = False
        while True:
            if args.no_display:
                key = 0xFF
            elif should_refresh_display or (video_controls_enabled and playback_paused):
                key = read_ui_key(video_controls_enabled=video_controls_enabled, playback_paused=playback_paused)
            else:
                key = 0xFF
            if key in (27, ord("q")):
                should_quit = True
                break
            if video_controls_enabled and key == ord(" "):
                playback_paused = not playback_paused
                if playback_paused:
                    continue
                break
            if key == ord("b") and args.ball_tracker != "none":
                select_bbox = cv2.selectROI("Goal Impact Tracker", frame, fromCenter=False, showCrosshair=True)
                if select_bbox[2] > 1 and select_bbox[3] > 1:
                    tracker = create_tracker(args.ball_tracker)
                    if tracker is not None and init_tracker_with_bbox(tracker, frame, select_bbox):
                        ball_tracker = tracker
                        tracker_locked = True
                        tracker_failures = 0
                        track_consecutive = max(1, args.track_min_consecutive)
                        impact_detector.reset_history()
                        trajectory_observations.clear()
                        impact_armed_at = time.time() + max(0.0, args.impact_arm_seconds)
            if key == ord("x"):
                ball_tracker = None
                tracker_locked = False
                tracker_failures = 0
                impact_detector.reset_history()
                trajectory_observations.clear()
            if key == ord("c"):
                recalibrated = calibrate_and_persist(
                    frame_bgr=frame,
                    calibration_path=calibration_path,
                    reference_frame_path=reference_frame_path,
                    goal_width_m=calibration.goal_width_m,
                    goal_height_m=calibration.goal_height_m,
                )
                calibration = recalibrated
                if goal_marker_layout is not None:
                    calibration.goal_width_m = goal_marker_layout.opening_width_m
                    calibration.goal_height_m = goal_marker_layout.opening_height_m
                current_corners = calibration.corners_px.astype(np.float32)
                adapter.reset_reference(frame, current_corners)
                impact_detector = create_impact_detector_for_goal(
                    args,
                    goal_marker_layout.opening_width_m if goal_marker_layout is not None else calibration.goal_width_m,
                    goal_marker_layout.opening_height_m if goal_marker_layout is not None else calibration.goal_height_m,
                )
                last_ball = None
                last_ball_seen_t = 0.0
                last_candidate = None
                last_candidate_seen_t = 0.0
                track_last_center = None
                track_last_radius = None
                track_consecutive = 0
                track_misses = 0
                ball_tracker = None
                tracker_locked = False
                tracker_failures = 0
                trajectory_observations.clear()
                latest_goal_pose = None
                latest_goal_pose_time = 0.0
                pose_fail_backoff = 1
                latest_event = None
                last_event_seen_t = 0.0
                recent_hits.clear()
                flash_until = 0.0
                impact_armed_at = time.time() + max(0.0, args.impact_arm_seconds)
            if video_controls_enabled and playback_paused:
                if key in (ord("n"), ord(".")):
                    step_video_frame = True
                    break
                continue
            break
        if perf_breakdown_enabled:
            add_perf_sample(perf_sums_s, "input_wait", time.perf_counter() - input_t0)
            add_perf_sample(perf_sums_s, "loop_total", time.perf_counter() - loop_perf_t0)
        if should_quit:
            break

        if args.stats_every > 0 and frame_idx % args.stats_every == 0:
            elapsed = max(1e-6, time.time() - stats_last_t)
            frames_done = frame_idx - stats_last_frame
            measured_fps = frames_done / elapsed
            det_rate = stats_det_runs / elapsed
            print(
                f"[Perf] loop_fps={measured_fps:.1f} detector_calls_s={det_rate:.1f} "
                f"detector={args.detector} process_every={args.process_every} adapt_every={args.adapt_every}"
            )
            if perf_breakdown_enabled:
                print_perf_breakdown(
                    perf_sums_s=perf_sums_s,
                    frames_done=frames_done,
                    top_n=args.perf_breakdown_top,
                    detector_name=args.detector,
                    args=args,
                    ball_detector=ball_detector,
                )
                pose_every = max(1, int(args.goal_pose_every))
                print(
                    f"[PerfDetail] pose_solve_calls={stats_pose_solve_calls} "
                    f"pose_solve_ok={stats_pose_solve_ok} pose_every={pose_every} "
                    f"pose_fail_backoff={pose_fail_backoff}"
                )
                if stats_pose_solve_calls > max(1, int(frames_done / max(1, pose_every)) + 2):
                    print(
                        "[PerfDetailHint] pose solver is running more often than expected for --goal-pose-every. "
                        "Check whether marker pose is repeatedly stale/lost."
                    )
                if stats_pose_solve_calls > 0 and stats_pose_solve_ok == 0 and stats_pose_fail_reasons:
                    top_reason, top_count = max(stats_pose_fail_reasons.items(), key=lambda kv: kv[1])
                    print(f"[PerfDetail] pose_fail_top=\"{top_reason}\" count={top_count}")
                    print(
                        "[PerfDetailHint] pose has zero successful solves in this window. "
                        "Fix marker visibility/layout first or temporarily run --ball-only-mode for detector FPS checks."
                    )
                perf_sums_s.clear()
                stats_pose_solve_calls = 0
                stats_pose_solve_ok = 0
                stats_pose_fail_reasons.clear()
            stats_last_t = time.time()
            stats_last_frame = frame_idx
            stats_det_runs = 0

    if async_capture is not None:
        async_capture.stop()
    if cap is not None:
        cap.release()
    detector_close = getattr(ball_detector, "close", None)
    if callable(detector_close):
        detector_close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
