from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .ball_detection import BallDetection
from .ball_identity import (
    IdentityMatch,
    PositiveSample,
    _collect_positive_samples,
    _crop_ball_square,
    _crop_from_sample,
    _identity_scores_agree,
    _sample_impostor_patches,
)

MODEL_FORMAT_VERSION = 1
DEFAULT_INPUT_SIZE = 96


@dataclass
class LearnedVerifierStats:
    threshold: float
    probability_threshold: float
    validation_accuracy: float
    validation_precision: float
    validation_recall: float
    provider: str
    input_size: int
    positive_count: int
    negative_count: int


class ONNXBallIdentityVerifier:
    def __init__(
        self,
        *,
        session,
        input_name: str,
        threshold: float,
        probability_threshold: float,
        input_size: int,
        provider: str,
        validation_accuracy: float,
        validation_precision: float,
        validation_recall: float,
        positive_count: int,
        negative_count: int,
    ) -> None:
        self.session = session
        self.input_name = str(input_name)
        self.threshold = float(threshold)
        self.probability_threshold = float(probability_threshold)
        self.input_size = int(input_size)
        self.provider = str(provider)
        self.validation_accuracy = float(validation_accuracy)
        self.validation_precision = float(validation_precision)
        self.validation_recall = float(validation_recall)
        self.positive_count = int(positive_count)
        self.negative_count = int(negative_count)

    @classmethod
    def load(cls, model_path: Path, threshold: float = 0.0) -> ONNXBallIdentityVerifier:
        model_path = Path(model_path)
        meta = _load_metadata(model_path)
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime is required for the learned exact-ball verifier. Install with: pip install onnxruntime"
            ) from exc

        providers = _preferred_ort_providers(ort)
        session = _create_ort_session(ort, model_path, providers)
        inputs = session.get_inputs()
        if not inputs:
            raise RuntimeError(f"ONNX verifier has no inputs: {model_path}")
        runtime_provider = session.get_providers()[0] if session.get_providers() else "unknown"
        score_threshold = float(threshold) if threshold > 0.0 else float(meta["threshold"])
        return cls(
            session=session,
            input_name=str(meta.get("input_name") or inputs[0].name),
            threshold=score_threshold,
            probability_threshold=float(meta.get("probability_threshold", 1.0 - score_threshold)),
            input_size=int(meta.get("input_size", DEFAULT_INPUT_SIZE)),
            provider=runtime_provider,
            validation_accuracy=float(meta.get("validation_accuracy", float("nan"))),
            validation_precision=float(meta.get("validation_precision", float("nan"))),
            validation_recall=float(meta.get("validation_recall", float("nan"))),
            positive_count=int(meta.get("positive_count", 0)),
            negative_count=int(meta.get("negative_count", 0)),
        )

    def describe(self) -> dict[str, float | str]:
        return {
            "threshold": float(self.threshold),
            "probability_threshold": float(self.probability_threshold),
            "validation_accuracy": float(self.validation_accuracy),
            "validation_precision": float(self.validation_precision),
            "validation_recall": float(self.validation_recall),
            "provider": self.provider,
            "input_size": float(self.input_size),
            "positive_count": float(self.positive_count),
            "negative_count": float(self.negative_count),
        }

    def verify(self, frame_bgr: np.ndarray, ball: BallDetection) -> IdentityMatch:
        scores: list[float] = []
        for scale in (1.10, 1.28, 1.45):
            crop = _crop_ball_square(frame_bgr, ball.center, ball.radius, scale)
            if crop is None:
                continue
            scores.append(self.score_crop(crop))
        if not scores:
            return IdentityMatch(False, 999.0, self.threshold)
        best_score = float(min(scores))
        accepted = best_score <= self.threshold and _identity_scores_agree(scores, self.threshold)
        return IdentityMatch(accepted, best_score, self.threshold)

    def score_crop(self, crop_bgr: np.ndarray) -> float:
        input_tensor = _prepare_network_input(crop_bgr, self.input_size)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        if not outputs:
            return 999.0
        logits = float(np.asarray(outputs[0], dtype=np.float32).reshape(-1)[0])
        probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logits))))
        return float(1.0 - probability)


@dataclass
class ONNXTrainingResult:
    output_path: Path
    metadata_path: Path
    device: str
    threshold: float
    probability_threshold: float
    validation_accuracy: float
    validation_precision: float
    validation_recall: float
    positive_count: int
    negative_count: int
    train_examples: int
    val_examples: int


def train_onnx_ball_identity(
    *,
    source: Path,
    val_source: Path | None,
    output_path: Path,
    device: str,
    threshold: float,
    probability_threshold: float,
    max_samples: int,
    input_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    val_fraction: float,
    min_positive_samples: int = 24,
    min_negative_samples: int = 24,
) -> ONNXTrainingResult:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to train the learned exact-ball verifier. Install torch and torchvision first."
        ) from exc

    rng = np.random.default_rng(int(seed))
    resolved_device = resolve_identity_train_device(device)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    positives, negatives = _build_training_crops(Path(source), max_samples=max_samples, rng=rng)
    if len(positives) < max(2, int(min_positive_samples)):
        raise RuntimeError(
            f"Need at least {max(2, int(min_positive_samples))} positive ball crops to train the learned verifier. Capture more ball crops first."
        )
    if len(negatives) < max(2, int(min_negative_samples)):
        raise RuntimeError(
            "Need more background/impostor crops to train the learned verifier. Use the updated identity capture tool or a labeled detector dataset root."
        )

    if val_source is not None:
        pos_train = list(positives)
        neg_train = list(negatives)
        pos_val, neg_val = _build_training_crops(Path(val_source), max_samples=max_samples, rng=rng)
        if len(pos_val) < 2 or len(neg_val) < 2:
            raise RuntimeError(
                "The separate validation source does not contain enough positive and negative crops. "
                "Capture at least a few ball crops and background/impostor patches in that validation set."
            )
    else:
        pos_train, pos_val = _split_examples(positives, val_fraction=val_fraction, rng=rng)
        neg_train, neg_val = _split_examples(negatives, val_fraction=val_fraction, rng=rng)
        if not pos_val:
            pos_val = pos_train[-max(1, len(pos_train) // 6):]
        if not neg_val:
            neg_val = neg_train[-max(1, len(neg_train) // 6):]

    train_examples = [(crop, 1) for crop in pos_train] + [(crop, 0) for crop in neg_train]
    val_examples = [(crop, 1) for crop in pos_val] + [(crop, 0) for crop in neg_val]

    class IdentityCropDataset(Dataset):
        def __init__(self, examples: list[tuple[np.ndarray, int]], augment: bool) -> None:
            self.examples = examples
            self.augment = augment

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int):
            crop, label = self.examples[index]
            image = crop.copy()
            if self.augment:
                image = _augment_training_crop(image, positive=bool(label), rng=rng)
            tensor = _prepare_network_input(image, input_size)[0]
            return torch.from_numpy(tensor), torch.tensor(float(label), dtype=torch.float32)

    train_loader = DataLoader(
        IdentityCropDataset(train_examples, augment=True),
        batch_size=max(4, int(batch_size)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        IdentityCropDataset(val_examples, augment=False),
        batch_size=max(4, int(batch_size)),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    class TinyBallIdentityNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(24),
                nn.SiLU(inplace=True),
                nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(48),
                nn.SiLU(inplace=True),
                nn.Conv2d(48, 80, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(80),
                nn.SiLU(inplace=True),
                nn.Conv2d(80, 112, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(112),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(112, 64),
                nn.SiLU(inplace=True),
                nn.Dropout(0.10),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = TinyBallIdentityNet().to(resolved_device)
    pos_weight = max(1.0, float(len(neg_train)) / max(1, len(pos_train)))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=resolved_device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)

    def evaluate(loader) -> tuple[np.ndarray, np.ndarray]:
        probs: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for images, target in loader:
                images = images.to(resolved_device)
                logits = model(images).squeeze(1)
                probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
                probs.append(probabilities)
                labels.append(target.numpy().astype(np.float32))
        if not probs:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return np.concatenate(probs), np.concatenate(labels)

    best_state = None
    best_metric = -1.0
    for epoch_idx in range(max(1, int(epochs))):
        model.train()
        for images, target in train_loader:
            images = images.to(resolved_device)
            target = target.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

        val_probs, val_labels = evaluate(val_loader)
        if val_probs.size == 0:
            metric = 0.0
        else:
            metric = float(_balanced_accuracy_from_probs(val_probs, val_labels, prob_threshold=0.5))
        if metric >= best_metric:
            best_metric = metric
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"[IdentityTrain] epoch={epoch_idx + 1}/{max(1, int(epochs))} val_bal_acc={metric:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_probs, val_labels = evaluate(val_loader)
    pos_probs = val_probs[val_labels > 0.5]
    neg_probs = val_probs[val_labels < 0.5]
    prob_threshold_final = float(probability_threshold) if probability_threshold > 0.0 else _choose_probability_threshold(pos_probs, neg_probs)
    score_threshold_final = float(threshold) if threshold > 0.0 else float(1.0 - prob_threshold_final)

    val_pred = (val_probs >= prob_threshold_final).astype(np.uint8)
    val_true = (val_labels >= 0.5).astype(np.uint8)
    tp = int(np.sum((val_pred == 1) & (val_true == 1)))
    tn = int(np.sum((val_pred == 0) & (val_true == 0)))
    fp = int(np.sum((val_pred == 1) & (val_true == 0)))
    fn = int(np.sum((val_pred == 0) & (val_true == 1)))
    val_accuracy = float((tp + tn) / max(1, val_true.size))
    val_precision = float(tp / max(1, tp + fp))
    val_recall = float(tp / max(1, tp + fn))

    model_cpu = model.to("cpu").eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,
    )

    metadata = {
        "format_version": MODEL_FORMAT_VERSION,
        "type": "ball_identity_onnx",
        "input_name": "input",
        "input_size": int(input_size),
        "threshold": float(score_threshold_final),
        "probability_threshold": float(prob_threshold_final),
        "training_device": resolved_device,
        "validation_accuracy": float(val_accuracy),
        "validation_precision": float(val_precision),
        "validation_recall": float(val_recall),
        "positive_count": int(len(positives)),
        "negative_count": int(len(negatives)),
        "train_examples": int(len(train_examples)),
        "val_examples": int(len(val_examples)),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return ONNXTrainingResult(
        output_path=output_path,
        metadata_path=metadata_path,
        device=resolved_device,
        threshold=score_threshold_final,
        probability_threshold=prob_threshold_final,
        validation_accuracy=val_accuracy,
        validation_precision=val_precision,
        validation_recall=val_recall,
        positive_count=len(positives),
        negative_count=len(negatives),
        train_examples=len(train_examples),
        val_examples=len(val_examples),
    )


def resolve_identity_train_device(requested: str | None) -> str:
    desired = (requested or "auto").strip().lower()
    try:
        import torch
    except ImportError:
        return "cpu"

    if desired in {"", "auto"}:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
    if desired == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        print("[IdentityTrain] requested device 'mps' is unavailable. Falling back to CPU.")
        return "cpu"
    if desired.startswith("cuda"):
        if torch.cuda.is_available():
            return desired
        print(f"[IdentityTrain] requested device '{desired}' is unavailable. Falling back to CPU.")
        return "cpu"
    return desired


def _load_metadata(model_path: Path) -> dict:
    meta_path = Path(str(model_path) + ".json")
    if not meta_path.exists():
        raise RuntimeError(f"Missing ONNX verifier metadata file: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _preferred_ort_providers(ort) -> list[str]:
    available = set(ort.get_available_providers())
    ordered: list[str] = []
    for provider in (
        "CoreMLExecutionProvider",
        "DmlExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ):
        if provider in available:
            ordered.append(provider)
    if not ordered:
        ordered = list(ort.get_available_providers())
    return ordered


def _create_ort_session(ort, model_path: Path, providers: list[str]):
    attempts: list[list[str]] = []
    if providers:
        attempts.append(providers)
        for provider in providers:
            attempts.append([provider])
    if ["CPUExecutionProvider"] not in attempts:
        attempts.append(["CPUExecutionProvider"])
    last_error: Exception | None = None
    for candidate in attempts:
        try:
            return ort.InferenceSession(str(model_path), providers=candidate)
        except Exception as exc:
            last_error = exc
            if "CPUExecutionProvider" in candidate and len(candidate) == 1:
                break
    raise RuntimeError(f"Could not load ONNX verifier session for {model_path}: {last_error}") from last_error


def _prepare_network_input(crop_bgr: np.ndarray, input_size: int) -> np.ndarray:
    square = _prepare_square_rgb(crop_bgr, input_size)
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))[None, :, :, :]
    return np.ascontiguousarray(chw, dtype=np.float32)


def _prepare_square_rgb(crop: np.ndarray, size: int) -> np.ndarray:
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 24, dtype=np.uint8)
    y = (side - h) // 2
    x = (side - w) // 2
    canvas[y:y + h, x:x + w] = crop
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR)


def _build_training_crops(source: Path, max_samples: int, rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    samples = _collect_positive_samples(Path(source), max_samples=max_samples)
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    for sample in samples:
        crop = _crop_from_sample(sample)
        if crop is None:
            continue
        positives.append(crop)
        negatives.extend(_sample_impostor_patches(sample, target_count=2))
        negatives.extend(_sample_hard_negative_patches(sample, target_count=1, rng=rng))
    rng.shuffle(positives)
    rng.shuffle(negatives)
    neg_limit = max(len(positives), min(len(negatives), len(positives) * 3))
    return positives, negatives[:neg_limit]


def _sample_hard_negative_patches(sample: PositiveSample, target_count: int, rng: np.random.Generator) -> list[np.ndarray]:
    if sample.bbox is None:
        return []
    image = sample.image
    h, w = image.shape[:2]
    x1, y1, x2, y2 = sample.bbox
    side = int(round(max(x2 - x1, y2 - y1) * 1.35))
    if side < 24:
        side = 24
    collected: list[np.ndarray] = []
    attempts = 0
    while len(collected) < target_count and attempts < target_count * 24:
        attempts += 1
        jitter_x = int(rng.integers(-side, side + 1))
        jitter_y = int(rng.integers(-side, side + 1))
        rx = int(np.clip(x1 + jitter_x, 0, max(0, w - side)))
        ry = int(np.clip(y1 + jitter_y, 0, max(0, h - side)))
        candidate_box = (rx, ry, rx + side, ry + side)
        if _iou(candidate_box, (x1, y1, x2, y2)) > 0.03:
            continue
        patch = image[ry:ry + side, rx:rx + side]
        if patch.size == 0:
            continue
        collected.append(patch.copy())
    return collected


def _split_examples(examples: list[np.ndarray], val_fraction: float, rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if not examples:
        return [], []
    order = np.arange(len(examples))
    rng.shuffle(order)
    val_count = int(round(len(order) * max(0.1, min(0.4, float(val_fraction)))))
    val_count = max(1, min(len(order) - 1, val_count)) if len(order) > 1 else 1
    val_idx = set(int(x) for x in order[:val_count])
    train: list[np.ndarray] = []
    val: list[np.ndarray] = []
    for idx, example in enumerate(examples):
        if idx in val_idx:
            val.append(example)
        else:
            train.append(example)
    if not train:
        train, val = val, train
    return train, val


def _augment_training_crop(image: np.ndarray, *, positive: bool, rng: np.random.Generator) -> np.ndarray:
    augmented = image.copy()
    if augmented.size == 0:
        return augmented
    h, w = augmented.shape[:2]
    angle = float(rng.uniform(-10.0, 10.0 if positive else 6.0))
    scale = float(rng.uniform(0.96, 1.04))
    matrix = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
    augmented = cv2.warpAffine(
        augmented,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    if rng.random() < (0.65 if positive else 0.35):
        k = int(rng.choice(np.asarray([3, 5], dtype=np.int32)))
        augmented = cv2.GaussianBlur(augmented, (k, k), 0)
    hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= float(rng.uniform(0.88, 1.15))
    hsv[:, :, 2] *= float(rng.uniform(0.84, 1.18))
    hsv[:, :, 1:] = np.clip(hsv[:, :, 1:], 0, 255)
    augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if positive and rng.random() < 0.22:
        occ_side = max(8, int(round(min(h, w) * rng.uniform(0.12, 0.24))))
        ox = int(rng.integers(0, max(1, w - occ_side + 1)))
        oy = int(rng.integers(0, max(1, h - occ_side + 1)))
        fill = int(rng.integers(16, 96))
        augmented[oy:oy + occ_side, ox:ox + occ_side] = fill
    return augmented


def _choose_probability_threshold(pos_probs: np.ndarray, neg_probs: np.ndarray) -> float:
    if pos_probs.size == 0 or neg_probs.size == 0:
        return 0.68
    candidates = np.linspace(0.45, 0.97, 53, dtype=np.float32)
    candidate_rows: list[tuple[float, float, float, float, float]] = []
    for threshold in candidates:
        tp = float(np.mean(pos_probs >= threshold))
        fp = float(np.mean(neg_probs >= threshold))
        precision = tp / max(1e-6, tp + fp)
        recall = tp
        specificity = float(np.mean(neg_probs < threshold))
        beta_sq = 0.5 ** 2
        fbeta = (1.0 + beta_sq) * precision * recall / max(1e-6, beta_sq * precision + recall)
        candidate_rows.append((float(threshold), precision, recall, specificity, fbeta))

    recall_floor = 0.60
    usable = [row for row in candidate_rows if row[2] >= recall_floor]
    if not usable:
        usable = candidate_rows

    best_threshold = 0.68
    best_score = -1.0
    for threshold, precision, recall, specificity, fbeta in usable:
        score = (1.15 * fbeta) + (0.20 * specificity) + (0.10 * precision)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    pos_anchor = float(np.percentile(pos_probs, 25))
    neg_anchor = float(np.percentile(neg_probs, 85))
    if neg_anchor < pos_anchor:
        best_threshold = 0.5 * (pos_anchor + neg_anchor)
    best_threshold = float(np.clip(best_threshold, 0.55, 0.88))
    return best_threshold


def _balanced_accuracy_from_probs(probs: np.ndarray, labels: np.ndarray, prob_threshold: float) -> float:
    if probs.size == 0 or labels.size == 0:
        return 0.0
    preds = (probs >= prob_threshold).astype(np.uint8)
    positives = labels >= 0.5
    negatives = ~positives
    tpr = float(np.mean(preds[positives] == 1)) if np.any(positives) else 0.0
    tnr = float(np.mean(preds[negatives] == 0)) if np.any(negatives) else 0.0
    return 0.5 * (tpr + tnr)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
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
