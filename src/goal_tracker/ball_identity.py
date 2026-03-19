from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .ball_detection import BallDetection

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_VERSION = 1


@dataclass
class IdentityMatch:
    accepted: bool
    score: float
    threshold: float


@dataclass
class PositiveSample:
    image: np.ndarray
    bbox: tuple[int, int, int, int] | None = None


def _identity_scores_agree(scores: list[float], threshold: float) -> bool:
    if len(scores) < 2:
        return True
    # Reject one-scale lucky matches: the best score can occasionally be very
    # low on a single crop while the other scales clearly disagree.
    ordered = np.sort(np.asarray(scores, dtype=np.float32))
    second_best = float(ordered[1])
    consistency_cap = max(0.78, float(threshold) + 0.20, float(threshold) * 1.75)
    return second_best <= consistency_cap


def load_identity_verifier(
    source: Path,
    threshold: float = 0.0,
    max_samples: int = 280,
):
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".onnx":
        from .ball_identity_learned import ONNXBallIdentityVerifier

        return ONNXBallIdentityVerifier.load(source, threshold=threshold)
    return BallIdentityVerifier.from_source(source, threshold=threshold, max_samples=max_samples)


class BallIdentityVerifier:
    def __init__(
        self,
        *,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        pca_components: np.ndarray,
        exemplar_embeddings: np.ndarray,
        threshold: float,
        positive_distance_p95: float,
        impostor_distance_p05: float,
    ) -> None:
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = np.maximum(feature_std.astype(np.float32), 1e-6)
        self.pca_components = pca_components.astype(np.float32)
        self.exemplar_embeddings = exemplar_embeddings.astype(np.float32)
        self.threshold = float(threshold)
        self.positive_distance_p95 = float(positive_distance_p95)
        self.impostor_distance_p05 = float(impostor_distance_p05)
        self._hog = _make_hog()

    @classmethod
    def from_source(
        cls,
        source: Path,
        threshold: float = 0.0,
        max_samples: int = 280,
    ) -> BallIdentityVerifier | None:
        source = Path(source)
        if source.is_file():
            if source.suffix.lower() == ".npz":
                return cls.load(source, threshold=threshold)
            return None
        if not source.exists():
            return None
        samples = _collect_positive_samples(source, max_samples=max_samples)
        if len(samples) < 12:
            return None
        return cls.fit(samples, threshold=threshold)

    @classmethod
    def fit(
        cls,
        samples: list[PositiveSample],
        threshold: float = 0.0,
        augment_per_sample: int = 2,
    ) -> BallIdentityVerifier | None:
        raw_features: list[np.ndarray] = []
        origin_index: list[int] = []
        impostor_features: list[np.ndarray] = []

        for sample_idx, sample in enumerate(samples):
            crop = _crop_from_sample(sample)
            if crop is None:
                continue
            base_variants = [crop]
            for _ in range(max(0, augment_per_sample)):
                base_variants.append(_augment_positive(crop))
            for variant in base_variants:
                feat = _extract_feature(variant)
                if feat is None:
                    continue
                raw_features.append(feat)
                origin_index.append(sample_idx)
            for impostor in _sample_impostor_patches(sample, target_count=2):
                feat = _extract_feature(impostor)
                if feat is not None:
                    impostor_features.append(feat)

        if len(raw_features) < 18:
            return None

        feature_matrix = np.stack(raw_features).astype(np.float32)
        mean = feature_matrix.mean(axis=0)
        std = feature_matrix.std(axis=0)
        std[std < 1e-6] = 1.0
        norm = (feature_matrix - mean[None, :]) / std[None, :]

        target_dim = int(min(64, max(12, norm.shape[0] - 1), norm.shape[1]))
        pca_components = _fit_pca(norm, target_dim)
        embeddings = _l2_normalize(norm @ pca_components)

        unique_origins = np.unique(np.asarray(origin_index, dtype=np.int32))
        if unique_origins.size < 8:
            return None

        positive_distances = _leave_one_origin_out_distances(embeddings, origin_index)
        if positive_distances.size == 0:
            return None
        pos_p95 = float(np.percentile(positive_distances, 95))

        impostor_p05 = float("nan")
        if impostor_features:
            impostor_matrix = np.stack(impostor_features).astype(np.float32)
            impostor_norm = (impostor_matrix - mean[None, :]) / std[None, :]
            impostor_embeddings = _l2_normalize(impostor_norm @ pca_components)
            impostor_dists = _nearest_distances(impostor_embeddings, embeddings)
            impostor_p05 = float(np.percentile(impostor_dists, 5))
        else:
            impostor_dists = np.empty((0,), dtype=np.float32)

        if threshold > 0.0:
            final_threshold = float(threshold)
        else:
            if np.isfinite(impostor_p05):
                target = pos_p95 * 1.06
                if impostor_p05 > target * 1.08:
                    target = 0.5 * (target + impostor_p05)
                final_threshold = float(np.clip(target, 0.08, 1.80))
            else:
                final_threshold = float(np.clip(pos_p95 * 1.16, 0.08, 1.80))

        return cls(
            feature_mean=mean,
            feature_std=std,
            pca_components=pca_components,
            exemplar_embeddings=embeddings,
            threshold=final_threshold,
            positive_distance_p95=pos_p95,
            impostor_distance_p05=impostor_p05,
        )

    def verify(self, frame_bgr: np.ndarray, ball: BallDetection) -> IdentityMatch:
        scores: list[float] = []
        for scale in (1.10, 1.28, 1.45):
            crop = _crop_ball_square(frame_bgr, ball.center, ball.radius, scale)
            if crop is None:
                continue
            feat = _extract_feature(crop, hog=self._hog)
            if feat is None:
                continue
            score = self.score_feature(feat)
            scores.append(score)
        if not scores:
            return IdentityMatch(False, 999.0, self.threshold)
        best_score = float(min(scores))
        accepted = best_score <= self.threshold and _identity_scores_agree(scores, self.threshold)
        return IdentityMatch(accepted, best_score, self.threshold)

    def score_feature(self, feature: np.ndarray) -> float:
        norm = (feature.astype(np.float32) - self.feature_mean) / self.feature_std
        embedding = _l2_normalize((norm @ self.pca_components)[None, :])[0]
        dists = np.linalg.norm(self.exemplar_embeddings - embedding[None, :], axis=1)
        return float(np.min(dists))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            model_version=np.asarray([MODEL_VERSION], dtype=np.int32),
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            pca_components=self.pca_components,
            exemplar_embeddings=self.exemplar_embeddings,
            threshold=np.asarray([self.threshold], dtype=np.float32),
            positive_distance_p95=np.asarray([self.positive_distance_p95], dtype=np.float32),
            impostor_distance_p05=np.asarray([self.impostor_distance_p05], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path, threshold: float = 0.0) -> BallIdentityVerifier:
        data = np.load(Path(path), allow_pickle=False)
        file_threshold = float(np.asarray(data["threshold"]).reshape(-1)[0])
        return cls(
            feature_mean=np.asarray(data["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(data["feature_std"], dtype=np.float32),
            pca_components=np.asarray(data["pca_components"], dtype=np.float32),
            exemplar_embeddings=np.asarray(data["exemplar_embeddings"], dtype=np.float32),
            threshold=float(threshold) if threshold > 0.0 else file_threshold,
            positive_distance_p95=float(np.asarray(data["positive_distance_p95"]).reshape(-1)[0]),
            impostor_distance_p05=float(np.asarray(data["impostor_distance_p05"]).reshape(-1)[0]),
        )

    def describe(self) -> dict[str, float]:
        return {
            "threshold": float(self.threshold),
            "positive_distance_p95": float(self.positive_distance_p95),
            "impostor_distance_p05": float(self.impostor_distance_p05),
            "embedding_count": float(self.exemplar_embeddings.shape[0]),
            "embedding_dim": float(self.exemplar_embeddings.shape[1]),
        }


def _list_images(image_dir: Path) -> list[Path]:
    return sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _collect_positive_samples(source: Path, max_samples: int) -> list[PositiveSample]:
    source = Path(source)
    positives: list[PositiveSample] = []

    frames_dir = source / "frames"
    boxes_dir = source / "boxes"
    if frames_dir.exists() and boxes_dir.exists():
        for image_path in _list_images(frames_dir):
            box_path = boxes_dir / f"{image_path.stem}.json"
            if not box_path.exists():
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            bbox = _bbox_from_identity_box_file(box_path, image.shape[1], image.shape[0])
            if bbox is None:
                continue
            positives.append(PositiveSample(image=image, bbox=bbox))
            if len(positives) >= max_samples:
                return positives
        if positives:
            return positives

    positives_dir = source / "positives"
    if positives_dir.exists():
        for image_path in _list_images(positives_dir):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            positives.append(PositiveSample(image=image, bbox=None))
            if len(positives) >= max_samples:
                return positives

    for split in ("train", "val"):
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image_path in _list_images(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            h, w = image.shape[:2]
            for line in label_path.read_text(encoding="utf-8").splitlines():
                bbox = _bbox_from_yolo_line(line, w, h)
                if bbox is None:
                    continue
                positives.append(PositiveSample(image=image, bbox=bbox))
                if len(positives) >= max_samples:
                    return positives

    if source.exists() and not positives:
        for image_path in _list_images(source):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            positives.append(PositiveSample(image=image, bbox=None))
            if len(positives) >= max_samples:
                return positives
    return positives


def _bbox_from_identity_box_file(box_path: Path, image_w: int, image_h: int) -> tuple[int, int, int, int] | None:
    try:
        payload = json.loads(box_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    bbox = payload.get("bbox_xywh")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = [int(round(float(v))) for v in bbox]
    except Exception:
        return None
    x1 = max(0, min(image_w - 1, x))
    y1 = max(0, min(image_h - 1, y))
    x2 = max(x1 + 1, min(image_w, x + max(1, w)))
    y2 = max(y1 + 1, min(image_h, y + max(1, h)))
    return x1, y1, x2, y2


def _bbox_from_yolo_line(line: str, image_w: int, image_h: int) -> tuple[int, int, int, int] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        _, cx, cy, nw, nh = parts
        cx_f = float(cx) * image_w
        cy_f = float(cy) * image_h
        w_f = float(nw) * image_w
        h_f = float(nh) * image_h
    except ValueError:
        return None
    x1 = int(round(cx_f - 0.5 * w_f))
    y1 = int(round(cy_f - 0.5 * h_f))
    x2 = int(round(cx_f + 0.5 * w_f))
    y2 = int(round(cy_f + 0.5 * h_f))
    return x1, y1, x2, y2


def _crop_from_sample(sample: PositiveSample) -> np.ndarray | None:
    if sample.bbox is None:
        crop = sample.image
        if crop.size == 0:
            return None
        return crop.copy()
    x1, y1, x2, y2 = sample.bbox
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    radius = max((x2 - x1), (y2 - y1)) * 0.55
    return _crop_ball_square(sample.image, (int(round(cx)), int(round(cy))), radius, scale=1.35)


def _crop_ball_square(
    image_bgr: np.ndarray,
    center: tuple[int, int],
    radius: float,
    scale: float,
) -> np.ndarray | None:
    h, w = image_bgr.shape[:2]
    side = int(round(max(16.0, 2.0 * float(radius) * scale)))
    cx, cy = center
    x1 = int(round(cx - side * 0.5))
    y1 = int(round(cy - side * 0.5))
    x2 = x1 + side
    y2 = y1 + side
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(32, 32, 32),
        )
    return crop


def _make_hog() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )


def _prepare_square(crop: np.ndarray, size: int = 64) -> np.ndarray:
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 32, dtype=np.uint8)
    y = (side - h) // 2
    x = (side - w) // 2
    canvas[y:y + h, x:x + w] = crop
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR)


def _extract_feature(crop_bgr: np.ndarray, hog: cv2.HOGDescriptor | None = None) -> np.ndarray | None:
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    prepared = _prepare_square(crop_bgr, size=64)
    gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(prepared, cv2.COLOR_BGR2HSV)
    yy, xx = np.ogrid[:64, :64]
    cx = cy = 32
    mask = (((xx - cx) ** 2 + (yy - cy) ** 2) <= int(64 * 0.36) ** 2).astype(np.uint8) * 255
    if hog is None:
        hog = _make_hog()
    hog_vec = hog.compute(gray).reshape(-1).astype(np.float32)
    thumb = cv2.resize(prepared, (12, 12), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    thumb_hsv = cv2.cvtColor((thumb * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    thumb_hsv[:, :, 0] /= 180.0
    thumb_hsv[:, :, 1:] /= 255.0
    thumb_feat = thumb_hsv.reshape(-1).astype(np.float32)
    hs_hist = cv2.calcHist([hsv], [0, 1], mask, [12, 4], [0, 180, 0, 256]).flatten().astype(np.float32)
    v_hist = cv2.calcHist([hsv], [2], mask, [8], [0, 256]).flatten().astype(np.float32)
    mask_bool = mask > 0
    sat = hsv[:, :, 1][mask_bool]
    val = hsv[:, :, 2][mask_bool]
    if sat.size < 20:
        return None
    white_ratio = float(np.mean((sat < 55) & (val > 150)))
    sat_ratio = float(np.mean(sat > 80))
    edge_map = cv2.Canny(gray, 50, 130)
    edge_ratio = float(np.mean(edge_map[mask_bool] > 0))
    color_feat = np.concatenate([hs_hist, v_hist, np.asarray([white_ratio, sat_ratio, edge_ratio], dtype=np.float32)])
    color_sum = float(np.sum(color_feat[:-3]))
    if color_sum > 1e-6:
        color_feat[:-3] /= color_sum
    feature = np.concatenate([hog_vec, thumb_feat, color_feat], axis=0).astype(np.float32)
    return feature


def _fit_pca(features: np.ndarray, out_dim: int) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:out_dim].T.astype(np.float32)


def _nearest_distances(samples: np.ndarray, exemplars: np.ndarray) -> np.ndarray:
    if samples.size == 0 or exemplars.size == 0:
        return np.empty((0,), dtype=np.float32)
    diffs = samples[:, None, :] - exemplars[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    return np.min(dists, axis=1).astype(np.float32)


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return (array / norms).astype(np.float32)


def _leave_one_origin_out_distances(embeddings: np.ndarray, origin_index: list[int]) -> np.ndarray:
    origin_arr = np.asarray(origin_index, dtype=np.int32)
    scores: list[float] = []
    for idx in range(embeddings.shape[0]):
        mask = origin_arr != origin_arr[idx]
        if not np.any(mask):
            continue
        dists = np.linalg.norm(embeddings[mask] - embeddings[idx][None, :], axis=1)
        scores.append(float(np.min(dists)))
    return np.asarray(scores, dtype=np.float32)


def _augment_positive(crop: np.ndarray) -> np.ndarray:
    augmented = crop.copy()
    if augmented.size == 0:
        return augmented
    h, w = augmented.shape[:2]
    angle = float(np.random.uniform(-8.0, 8.0))
    matrix = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, np.random.uniform(0.96, 1.04))
    augmented = cv2.warpAffine(
        augmented,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    if np.random.rand() < 0.65:
        k = int(np.random.choice([3, 5]))
        augmented = cv2.GaussianBlur(augmented, (k, k), 0)
    hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= np.random.uniform(0.90, 1.12)
    hsv[:, :, 2] *= np.random.uniform(0.88, 1.14)
    hsv[:, :, 1:] = np.clip(hsv[:, :, 1:], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _sample_impostor_patches(sample: PositiveSample, target_count: int) -> list[np.ndarray]:
    if sample.bbox is None:
        return []
    image = sample.image
    h, w = image.shape[:2]
    x1, y1, x2, y2 = sample.bbox
    box_w = max(16, x2 - x1)
    box_h = max(16, y2 - y1)
    side = int(round(max(box_w, box_h) * 1.35))
    positives = []
    tries = 0
    while len(positives) < target_count and tries < target_count * 10:
        tries += 1
        rx = int(np.random.randint(0, max(1, w - side + 1)))
        ry = int(np.random.randint(0, max(1, h - side + 1)))
        patch_box = (rx, ry, rx + side, ry + side)
        if _intersection_over_union(patch_box, (x1, y1, x2, y2)) > 0.05:
            continue
        patch = image[ry:ry + side, rx:rx + side]
        if patch.size == 0:
            continue
        positives.append(patch.copy())
    return positives


def _intersection_over_union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
    area_b = float(max(1, bx2 - bx1) * max(1, by2 - by1))
    return inter / max(1e-6, area_a + area_b - inter)
