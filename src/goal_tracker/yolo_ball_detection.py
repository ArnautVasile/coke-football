from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from contextlib import contextmanager
import platform
import re
import time

import cv2
import numpy as np

from .ball_identity import load_identity_verifier
from .ball_detection import BallDetection
from .charuco import create_dictionary


@dataclass
class YoloConfig:
    model: str = "yolo26s.pt"
    conf: float = 0.10
    imgsz: int = 640
    device: str | None = None
    class_id: int = 32  # COCO sports ball class
    use_tracker: bool = True
    tracker_cfg: str = "configs/bytetrack_ball.yaml"
    identity_source: str = ""
    identity_threshold: float = 0.0


def should_preserve_mps_for_onnx(requested: str, model_path: str) -> bool:
    if requested.lower() != "mps":
        return False
    if not model_path.lower().endswith(".onnx"):
        return False
    if platform.system() != "Darwin":
        return False
    try:
        import onnxruntime  # type: ignore

        return "CoreMLExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False


def resolve_yolo_device(requested: str | None, model_path: str = "") -> str | None:
    device = (requested or "").strip()
    if not device:
        return None

    d = device.lower()
    if should_preserve_mps_for_onnx(device, model_path):
        print("[YOLO] Preserving device 'mps' for ONNX Runtime CoreMLExecutionProvider on macOS.")
        return "mps"

    try:
        import torch  # type: ignore
    except Exception:
        if d.startswith("cuda") or d == "mps":
            print(f"[YOLO] Requested device '{device}' is unavailable. Falling back to CPU.")
            return "cpu"
        return device

    if d.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"[YOLO] Requested device '{device}' but CUDA is unavailable. Falling back to CPU.")
            return "cpu"
        return device

    if d == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        has_mps = bool(mps_backend) and bool(mps_backend.is_available())
        if not has_mps:
            print(f"[YOLO] Requested device '{device}' but MPS is unavailable. Falling back to CPU.")
            return "cpu"
        return device

    return device


def _validate_model_reference(model_path: str) -> None:
    path = Path(model_path)
    if model_path.startswith("/absolute/path/to/"):
        raise RuntimeError(
            "The YOLO model path is still a placeholder. Replace '/absolute/path/to/...' with the real exported model path."
        )
    # Allow bare model names like 'yolo26s.pt' so Ultralytics can resolve/download them.
    looks_like_filesystem_path = path.is_absolute() or len(path.parts) > 1
    if looks_like_filesystem_path and not path.exists():
        raise RuntimeError(
            f"YOLO model file not found: {path}. Use the real path printed after training/export, for example "
            f"'runs/custom_ball/ball_yolo26s_identitymix_v2/weights/best.onnx'."
        )


class YoloBallDetector:
    def __init__(self, cfg: YoloConfig) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YOLO detector requested but ultralytics is not installed. "
                "Install with: .venv/bin/python -m pip install ultralytics"
            ) from exc

        self._yolo_ctor = lambda model_path: YOLO(model_path, task="detect")
        self.cfg = cfg
        _validate_model_reference(cfg.model)
        self.model = self._yolo_ctor(cfg.model)
        self.is_onnx = cfg.model.lower().endswith(".onnx")
        self.device = resolve_yolo_device(cfg.device, cfg.model)
        self.fixed_imgsz = self._detect_fixed_onnx_imgsz(cfg.model) if self.is_onnx else None
        if self.fixed_imgsz is not None and int(self.cfg.imgsz) != int(self.fixed_imgsz):
            print(
                f"[YOLO] ONNX model expects fixed input {self.fixed_imgsz}x{self.fixed_imgsz}. "
                f"Overriding requested imgsz={self.cfg.imgsz}."
            )
            self.cfg.imgsz = int(self.fixed_imgsz)
        self.last_track_id: int | None = None
        self._recovered_to_cpu = False
        self._runtime_provider_logged = False
        self._coreml_provider_patch_logged = False
        self._identity_verifier = None
        self._identity_unreliable = False
        self._last_debug_reason = ""
        self._last_debug_time = 0.0
        self._detect_frame_index = 0
        self._last_verified_frame_index: int | None = None
        self._last_verified_detection: BallDetection | None = None
        self._marker_dictionary = create_dictionary("DICT_6X6_50")
        self._marker_detector = cv2.aruco.ArucoDetector(self._marker_dictionary, cv2.aruco.DetectorParameters())
        self._load_identity_verifier()

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
            raise RuntimeError(f"YOLO identity source not found: {source_root}")
        verifier = load_identity_verifier(
            source=source_root,
            threshold=float(self.cfg.identity_threshold),
        )
        if verifier is None:
            raise RuntimeError(
                "YOLO exact-ball verifier could not be built from the provided source. "
                "Use a labeled dataset root, a verifier positives directory, or a saved .npz/.onnx verifier model."
            )
        self._identity_verifier = verifier
        stats = verifier.describe()
        threshold_value = float(stats.get("threshold", 0.0))
        if "probability_threshold" in stats:
            print(
                "[YOLO] learned exact-ball verifier loaded "
                f"(source={source_root}, threshold={threshold_value:.3f}, "
                f"prob={float(stats.get('probability_threshold', 0.0)):.3f}, "
                f"provider={stats.get('provider', 'unknown')})"
            )
        else:
            print(
                "[YOLO] learned exact-ball verifier loaded "
                f"(source={source_root}, threshold={threshold_value:.3f}, "
                f"pos95={float(stats.get('positive_distance_p95', float('nan'))):.3f}, "
                f"impostor05={float(stats.get('impostor_distance_p05', float('nan'))):.3f})"
            )
        self._assess_identity_verifier_reliability(verifier, threshold_value)

    def _assess_identity_verifier_reliability(self, verifier: object, threshold: float) -> None:
        score_crop = getattr(verifier, "score_crop", None)
        if not callable(score_crop):
            return
        probes: list[np.ndarray] = [
            np.zeros((128, 128, 3), dtype=np.uint8),
            np.full((128, 128, 3), 255, dtype=np.uint8),
            np.full((128, 128, 3), (0, 0, 255), dtype=np.uint8),
        ]
        rng = np.random.default_rng(0)
        probes.append(rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8))
        accepted = 0
        for probe in probes:
            try:
                score = float(score_crop(probe))
            except Exception:
                return
            if score <= threshold:
                accepted += 1
        if accepted >= 2:
            self._identity_unreliable = True
            print(
                "[YOLO] warning: learned exact-ball verifier looks over-permissive "
                f"on synthetic probes (accepted {accepted}/{len(probes)}). "
                "Using stricter full-frame candidate gating."
            )

    def _detect_fixed_onnx_imgsz(self, model_path: str) -> int | None:
        try:
            import onnx  # type: ignore

            graph = onnx.load(model_path).graph
            if not graph.input:
                return None
            dims = graph.input[0].type.tensor_type.shape.dim
            if len(dims) < 4:
                return None
            h_dim = dims[2]
            w_dim = dims[3]
            if not (h_dim.HasField("dim_value") and w_dim.HasField("dim_value")):
                return None
            h = int(h_dim.dim_value)
            w = int(w_dim.dim_value)
            if h > 0 and w > 0 and h == w:
                return h
        except Exception:
            return None
        return None

    def _run_model(self, predict_args: dict[str, Any]):
        with self._patched_onnxruntime_coreml():
            if self.cfg.use_tracker:
                predict_args["persist"] = True
                predict_args["tracker"] = self.cfg.tracker_cfg
                return self.model.track(**predict_args)
            return self.model.predict(**predict_args)

    def _coreml_provider_options(self) -> list[Any] | None:
        if not self.is_onnx or self.device != "mps" or platform.system() != "Darwin":
            return None
        if self.fixed_imgsz is None:
            return None
        # MLProgram + static-shape CoreML EP avoids the GatherElements crash we reproduced
        # on real frames and accelerates more of the graph than the default ORT setup.
        return [
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "1",
                    "EnableOnSubgraphs": "0",
                },
            ),
            "CPUExecutionProvider",
        ]

    @contextmanager
    def _patched_onnxruntime_coreml(self):
        provider_options = self._coreml_provider_options()
        if provider_options is None:
            yield
            return

        try:
            import onnxruntime  # type: ignore
        except Exception:
            yield
            return

        original_ctor = onnxruntime.InferenceSession

        def patched_ctor(*args, **kwargs):
            providers = kwargs.get("providers")
            wants_coreml = False
            if isinstance(providers, list):
                wants_coreml = any(
                    (p == "CoreMLExecutionProvider") or (isinstance(p, tuple) and p and p[0] == "CoreMLExecutionProvider")
                    for p in providers
                )
            if wants_coreml:
                kwargs["providers"] = provider_options
                if not self._coreml_provider_patch_logged:
                    print(
                        "[YOLO] Tuning ONNX Runtime CoreMLExecutionProvider with "
                        "MLProgram/static-shape provider options."
                    )
                    self._coreml_provider_patch_logged = True
            return original_ctor(*args, **kwargs)

        onnxruntime.InferenceSession = patched_ctor
        try:
            yield
        finally:
            onnxruntime.InferenceSession = original_ctor

    def _recover_onnx_runtime(self, predict_args: dict[str, Any], message: str):
        is_coreml_path = self.is_onnx and self.device == "mps"
        known_runtime_bug = "GatherElements" in message or "onnxruntime_pybind11_state.RuntimeException" in message
        if self.is_onnx and "Got invalid dimensions for input" in message:
            expected = re.findall(r"Expected:\s*(\d+)", message)
            if len(expected) >= 2 and expected[0] == expected[1]:
                fixed = int(expected[0])
                if fixed > 0 and fixed != int(self.cfg.imgsz):
                    print(
                        f"[YOLO] ONNX runtime reported fixed input {fixed}x{fixed}. "
                        "Retrying with that imgsz."
                    )
                    self.cfg.imgsz = fixed
                    predict_args["imgsz"] = fixed
                    return self._run_model(predict_args)
        if not is_coreml_path or self._recovered_to_cpu or not known_runtime_bug:
            raise RuntimeError(message)

        print(
            "[YOLO] ONNX/CoreML runtime failed on a live frame. "
            "Rebuilding detector on CPU ONNX Runtime and continuing."
        )
        self.device = "cpu"
        self.model = self._yolo_ctor(self.cfg.model)
        self._recovered_to_cpu = True
        predict_args["device"] = "cpu"
        return self._run_model(predict_args)

    def _maybe_log_runtime_provider(self) -> None:
        if self._runtime_provider_logged or not self.is_onnx:
            return
        predictor = getattr(self.model, "predictor", None)
        backend = getattr(predictor, "model", None)
        session = getattr(backend, "session", None)
        if session is None or not hasattr(session, "get_providers"):
            return
        providers = session.get_providers()
        provider = providers[0] if providers else "unknown"
        print(f"[YOLO] actual_runtime_provider={provider}")
        if self.device == "mps" and provider == "CPUExecutionProvider":
            print(
                "[YOLO] warning: requested mps/CoreML path, but ONNX Runtime is actually using CPUExecutionProvider. "
                "Expect lower FPS until a CoreML-compatible runtime path is active."
            )
        self._runtime_provider_logged = True

    def _detection_from_box(self, xyxy: np.ndarray, x1: int, y1: int) -> BallDetection:
        bx1, by1, bx2, by2 = [float(v) for v in xyxy]
        w = max(1.0, bx2 - bx1)
        h = max(1.0, by2 - by1)
        cx = int((bx1 + bx2) * 0.5) + x1
        cy = int((by1 + by2) * 0.5) + y1
        radius = max(w, h) * 0.5
        area = w * h
        circularity = float(min(w, h) / max(w, h))
        return BallDetection(center=(cx, cy), radius=radius, area=area, circularity=circularity)

    def _remember_verified_detection(self, detection: BallDetection, frame_index: int) -> None:
        self._last_verified_frame_index = int(frame_index)
        self._last_verified_detection = detection

    def _matches_recent_verified_ball(self, detection: BallDetection, *, frame_index: int, max_gap_frames: int = 3) -> bool:
        if self._last_verified_frame_index is None or self._last_verified_detection is None:
            return False
        gap_frames = int(frame_index) - int(self._last_verified_frame_index)
        if gap_frames < 0 or gap_frames > max(1, int(max_gap_frames)):
            return False
        prev = self._last_verified_detection
        prev_center = np.asarray(prev.center, dtype=np.float32)
        curr_center = np.asarray(detection.center, dtype=np.float32)
        center_distance = float(np.linalg.norm(curr_center - prev_center))
        max_distance = max(28.0, (3.2 + 0.35 * max(0, gap_frames - 1)) * max(prev.radius, detection.radius))
        if center_distance > max_distance:
            return False
        prev_radius = max(1.0, float(prev.radius))
        curr_radius = max(1.0, float(detection.radius))
        radius_ratio = curr_radius / prev_radius
        if radius_ratio < 0.48 or radius_ratio > 2.15:
            return False
        return True

    def _can_identity_bridge(self, detection: BallDetection, *, frame_index: int, identity_score: float, identity_threshold: float) -> bool:
        if self._identity_verifier is None:
            return False
        soft_threshold = max(0.02, float(identity_threshold) * 1.10)
        if float(identity_score) > soft_threshold:
            return False
        return self._matches_recent_verified_ball(detection, frame_index=frame_index, max_gap_frames=3)

    def _estimate_shape_roundness(self, frame_bgr: np.ndarray, detection: BallDetection) -> float | None:
        crop_half = int(round(max(18.0, float(detection.radius) * 1.12)))
        cx, cy = detection.center
        x1 = max(0, cx - crop_half)
        y1 = max(0, cy - crop_half)
        x2 = min(frame_bgr.shape[1], cx + crop_half)
        y2 = min(frame_bgr.shape[0], cy + crop_half)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < 18:
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
            contour_cx = float(moments["m10"] / moments["m00"])
            contour_cy = float(moments["m01"] / moments["m00"])
            center_distance = float(
                np.linalg.norm(np.asarray([contour_cx, contour_cy], dtype=np.float32) - crop_center)
            )
            if center_distance > (0.35 * max_side):
                continue

            (_, _), enclosing_radius = cv2.minEnclosingCircle(contour)
            if enclosing_radius < 3.0:
                continue
            circularity = float((4.0 * np.pi * area) / max(1e-6, perimeter * perimeter))
            fill_ratio = float(area / max(1e-6, np.pi * enclosing_radius * enclosing_radius))
            points = contour.reshape(-1, 2).astype(np.float32)
            radial = np.linalg.norm(points - np.asarray([contour_cx, contour_cy], dtype=np.float32), axis=1)
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

    def _candidate_order(self, confs: np.ndarray, ids: np.ndarray | None) -> list[int]:
        order = list(np.argsort(-confs).astype(int))
        if ids is None or len(ids) != len(confs) or self.last_track_id is None:
            return order
        for idx, track_id in enumerate(ids):
            if int(track_id) == int(self.last_track_id):
                return [idx] + [i for i in order if int(i) != int(idx)]
        return order

    def _crop_contains_marker(self, frame_bgr: np.ndarray, detection: BallDetection) -> bool:
        crop_half = int(round(max(18.0, float(detection.radius) * 1.15)))
        cx, cy = detection.center
        x1 = max(0, cx - crop_half)
        y1 = max(0, cy - crop_half)
        x2 = min(frame_bgr.shape[1], cx + crop_half)
        y2 = min(frame_bgr.shape[0], cy + crop_half)
        if x2 <= x1 or y2 <= y1:
            return False
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < 18:
            return False
        corners, ids, _ = self._marker_detector.detectMarkers(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        if ids is None or corners is None:
            return False
        crop_center = np.array([crop.shape[1] * 0.5, crop.shape[0] * 0.5], dtype=np.float32)
        for marker_corners in corners:
            pts = np.asarray(marker_corners, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] < 4:
                continue
            marker_center = pts.mean(axis=0)
            marker_half = max(1.0, 0.5 * max(float(pts[:, 0].max() - pts[:, 0].min()), float(pts[:, 1].max() - pts[:, 1].min())))
            if float(np.linalg.norm(marker_center - crop_center)) <= marker_half * 0.9:
                return True
        return False

    def detect(self, frame_bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> BallDetection | None:
        self._detect_frame_index += 1
        frame_index = self._detect_frame_index
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

        crop = frame_bgr[y1:y2, x1:x2]
        predict_args: dict[str, Any] = {
            "source": crop,
            "conf": self.cfg.conf,
            "imgsz": self.cfg.imgsz,
            "classes": [self.cfg.class_id],
            "verbose": False,
        }
        if self.device:
            predict_args["device"] = self.device
        try:
            results = self._run_model(predict_args)
        except ValueError as exc:
            msg = str(exc)
            if "Invalid CUDA" in msg and self.device != "cpu":
                self.device = "cpu"
                predict_args["device"] = "cpu"
                print("[YOLO] CUDA runtime unavailable. Retrying on CPU.")
                results = self._run_model(predict_args)
            else:
                raise
        except Exception as exc:
            results = self._recover_onnx_runtime(predict_args, str(exc))
        self._maybe_log_runtime_provider()
        if not results:
            self._set_debug_reason("reject: yolo empty")
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            self._set_debug_reason("reject: yolo none")
            return None

        confs = boxes.conf.detach().cpu().numpy().astype(float)
        ids = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else None
        ordered_indices = self._candidate_order(confs, ids)
        saw_marker = False
        saw_identity_reject = False
        saw_shape_reject = False
        selected_detection: BallDetection | None = None
        selected_idx: int | None = None
        accepted_candidates: list[tuple[int, BallDetection, float, float, bool, bool, float | None]] = []

        for idx in ordered_indices:
            xyxy = boxes.xyxy[int(idx)].detach().cpu().numpy()
            detection = self._detection_from_box(xyxy, x1, y1)
            recent_match = self._matches_recent_verified_ball(detection, frame_index=frame_index, max_gap_frames=3)
            if self._crop_contains_marker(frame_bgr, detection):
                saw_marker = True
                continue
            confidence = float(confs[int(idx)])
            if self._identity_verifier is not None:
                identity = self._identity_verifier.verify(frame_bgr, detection)
                identity_score = float(identity.score)
                shape_score = self._estimate_shape_roundness(frame_bgr, detection)
                if shape_score is not None:
                    detection.circularity = float(shape_score)
                    very_strong_identity = identity.score <= max(0.02, identity.threshold * 0.68)
                    if not recent_match and shape_score < 0.50 and not very_strong_identity:
                        saw_shape_reject = True
                        continue
                if not identity.accepted:
                    if self._can_identity_bridge(
                        detection,
                        frame_index=frame_index,
                        identity_score=identity.score,
                        identity_threshold=identity.threshold,
                    ):
                        accepted_candidates.append(
                            (int(idx), detection, confidence, identity_score, recent_match, True, shape_score)
                        )
                        continue
                    saw_identity_reject = True
                    continue
                accepted_candidates.append(
                    (int(idx), detection, confidence, identity_score, recent_match, False, shape_score)
                )
                continue
            accepted_candidates.append((int(idx), detection, confidence, 0.0, recent_match, False, None))

        if not accepted_candidates:
            if saw_marker:
                self._set_debug_reason("reject: marker")
            elif saw_shape_reject:
                self._set_debug_reason("reject: shape")
            elif saw_identity_reject:
                self._set_debug_reason("reject: identity")
            else:
                self._set_debug_reason("reject: yolo filter")
            self.last_track_id = None
            return None

        selected_was_bridge = False
        if self._identity_verifier is None:
            selected_idx, selected_detection, _, _, _, selected_was_bridge, _ = accepted_candidates[0]
        else:
            recent_candidates = [entry for entry in accepted_candidates if entry[4]]
            if recent_candidates:
                selected_idx, selected_detection, _, _, _, selected_was_bridge, _ = min(
                    recent_candidates,
                    key=lambda entry: (
                        entry[3],
                        -entry[2],
                        -(entry[6] if entry[6] is not None else -1.0),
                        float(entry[1].radius),
                    ),
                )
            else:
                if self._last_verified_frame_index is not None:
                    gap = int(frame_index) - int(self._last_verified_frame_index)
                    drift_guard_frames = 140 if self._identity_unreliable else 90
                    if 0 <= gap <= drift_guard_frames:
                        self._set_debug_reason("reject: identity drift")
                        self.last_track_id = None
                        return None
                if selected_detection is None:
                    non_bridge = [entry for entry in accepted_candidates if not entry[5]]
                    candidate_pool = non_bridge if non_bridge else accepted_candidates
                    fresh_conf_floor = max(0.14, float(self.cfg.conf) + (0.10 if self._identity_unreliable else 0.06))
                    fresh_confident = [entry for entry in candidate_pool if entry[2] >= fresh_conf_floor]
                    if fresh_confident:
                        candidate_pool = fresh_confident

                    id_scores = np.asarray([entry[3] for entry in candidate_pool], dtype=np.float32)
                    identity_flat = len(candidate_pool) >= 3 and float(np.std(id_scores)) <= max(0.003, float(np.mean(id_scores)) * 0.08)
                    if identity_flat:
                        best_conf = max(entry[2] for entry in candidate_pool)
                        spread_floor = max(
                            fresh_conf_floor,
                            best_conf - (0.55 if self._identity_unreliable else 0.65),
                        )
                        high_conf = [entry for entry in candidate_pool if entry[2] >= spread_floor]
                        realistic_size = [
                            entry
                            for entry in high_conf
                            if 18.0 <= float(entry[1].radius) <= max(220.0, float(frame_bgr.shape[0]) * 0.28)
                        ]
                        candidate_pool = realistic_size if realistic_size else high_conf
                        ranked = sorted(
                            candidate_pool,
                            key=lambda entry: (
                                float(entry[1].radius),
                                -entry[2],
                                entry[3],
                            ),
                        )
                    else:
                        ranked = sorted(
                            candidate_pool,
                            key=lambda entry: (
                                entry[3],
                                -entry[2],
                                -(entry[6] if entry[6] is not None else -1.0),
                                float(entry[1].radius),
                            ),
                        )
                    best = ranked[0]
                    ambiguous = False
                    if len(ranked) > 1:
                        second = ranked[1]
                        if identity_flat:
                            radius_gap = float(second[1].radius) - float(best[1].radius)
                            confidence_gap = float(best[2]) - float(second[2])
                            if self._identity_unreliable:
                                ambiguous = radius_gap < 6.0 and confidence_gap < 0.14
                            else:
                                ambiguous = radius_gap < 4.0 and confidence_gap < 0.10
                        else:
                            identity_close = second[3] <= (best[3] + max(0.010, float(best[3]) * 0.18))
                            confidence_close = second[2] >= (best[2] - 0.07)
                            best_shape = best[6] if best[6] is not None else 0.0
                            second_shape = second[6] if second[6] is not None else 0.0
                            shape_advantage = best_shape - second_shape
                            ambiguous = identity_close and confidence_close and shape_advantage < 0.10
                    if ambiguous:
                        self._set_debug_reason("reject: identity ambiguous")
                        self.last_track_id = None
                        return None
                    selected_idx, selected_detection, _, _, _, selected_was_bridge, _ = best
                    if len(ranked) > 1:
                        self._set_debug_reason("accept: identity tie-break")

            if selected_detection is not None:
                self._remember_verified_detection(selected_detection, frame_index)
                if selected_was_bridge:
                    self._set_debug_reason("accept: identity bridge")

        if ids is not None and len(ids) == len(confs) and selected_idx is not None:
            self.last_track_id = int(ids[selected_idx])
        else:
            self.last_track_id = None
        if not self._last_debug_reason.startswith("accept:"):
            self._clear_debug_reason()
        return selected_detection
