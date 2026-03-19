from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goal_tracker.ball_identity import BallIdentityVerifier
from goal_tracker.ball_identity_learned import resolve_identity_train_device, train_onnx_ball_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an exact-ball verifier. Supports classic .npz fitting and learned ONNX export."
    )
    parser.add_argument(
        "--source",
        default="datasets/ball_identity",
        help="Source directory. Supports frames/boxes/positives identity sets or images/labels train/val dataset roots.",
    )
    parser.add_argument(
        "--val-source",
        default="",
        help="Optional separate validation identity dataset root. When set, validation comes from this source instead of a random holdout from --source.",
    )
    parser.add_argument("--output", default="data/identity/ball_identity.onnx", help="Output verifier artifact (.onnx or .npz)")
    parser.add_argument(
        "--format",
        choices=["auto", "onnx", "classic"],
        default="auto",
        help="Verifier format. Auto chooses ONNX for .onnx outputs and classic for .npz outputs.",
    )
    parser.add_argument("--threshold", type=float, default=0.0, help="Manual score-threshold override for the classic verifier or ONNX score mode")
    parser.add_argument("--prob-threshold", type=float, default=0.0, help="Optional manual probability threshold for the learned ONNX verifier")
    parser.add_argument("--max-samples", type=int, default=280, help="Max positive samples to use while fitting")
    parser.add_argument("--device", default="auto", help="Training device for ONNX mode: auto, mps, cpu, cuda:0")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for ONNX mode")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for ONNX mode")
    parser.add_argument("--input-size", type=int, default=128, help="Square input size for the learned ONNX verifier")
    parser.add_argument("--learning-rate", type=float, default=1.5e-3, help="Learning rate for ONNX mode")
    parser.add_argument("--val-fraction", type=float, default=0.18, help="Holdout fraction for ONNX mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for ONNX mode")
    parser.add_argument("--smoke-test", action="store_true", help="Allow tiny datasets for a quick verifier proof-of-concept")
    return parser.parse_args()


def resolve_format(fmt: str, output: Path) -> str:
    if fmt != "auto":
        return fmt
    if output.suffix.lower() == ".npz":
        return "classic"
    return "onnx"


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    val_source = Path(args.val_source) if str(args.val_source).strip() else None
    output = Path(args.output)
    model_format = resolve_format(args.format, output)

    if model_format == "classic":
        verifier = BallIdentityVerifier.from_source(source, threshold=args.threshold, max_samples=args.max_samples)
        if verifier is None:
            raise SystemExit(
                "Could not build a classic exact-ball verifier from the provided source. "
                "Use either a positives directory or a labeled dataset root."
            )
        verifier.save(output)
        stats = verifier.describe()
        print(f"[IdentityTrain] format=classic")
        print(f"[IdentityTrain] source={source.resolve()}")
        print(f"[IdentityTrain] model={output.resolve()}")
        print(f"[IdentityTrain] threshold={stats['threshold']:.4f}")
        print(f"[IdentityTrain] pos95={stats['positive_distance_p95']:.4f}")
        if stats["impostor_distance_p05"] == stats["impostor_distance_p05"]:
            print(f"[IdentityTrain] impostor05={stats['impostor_distance_p05']:.4f}")
        else:
            print("[IdentityTrain] impostor05=nan (no background calibration patches available)")
        print(f"[IdentityTrain] embeddings={int(stats['embedding_count'])} dim={int(stats['embedding_dim'])}")
        print("[Next] Run the tracker with:")
        print("python run.py --detector vision " f'--vision-identity-source "{output.resolve()}"')
        return

    if output.suffix.lower() != ".onnx":
        raise SystemExit("ONNX mode expects an output path ending in .onnx")

    resolved_device = resolve_identity_train_device(args.device)
    print(f"[IdentityTrain] format=onnx")
    print(f"[IdentityTrain] source={source.resolve()}")
    print(f"[IdentityTrain] output={output.resolve()}")
    if val_source is not None:
        print(f"[IdentityTrain] val_source={val_source.resolve()}")
    print(f"[IdentityTrain] device={resolved_device} epochs={args.epochs} batch={args.batch_size} input={args.input_size}")
    min_positive_samples = 5 if args.smoke_test else 24
    min_negative_samples = 10 if args.smoke_test else 24
    result = train_onnx_ball_identity(
        source=source,
        val_source=val_source,
        output_path=output,
        device=args.device,
        threshold=args.threshold,
        probability_threshold=args.prob_threshold,
        max_samples=args.max_samples,
        input_size=args.input_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        val_fraction=args.val_fraction,
        min_positive_samples=min_positive_samples,
        min_negative_samples=min_negative_samples,
    )
    print(f"[IdentityTrain] model={result.output_path.resolve()}")
    print(f"[IdentityTrain] metadata={result.metadata_path.resolve()}")
    print(f"[IdentityTrain] threshold={result.threshold:.4f} prob_threshold={result.probability_threshold:.4f}")
    print(
        f"[IdentityTrain] val_acc={result.validation_accuracy:.4f} "
        f"precision={result.validation_precision:.4f} recall={result.validation_recall:.4f}"
    )
    print(
        f"[IdentityTrain] positives={result.positive_count} negatives={result.negative_count} "
        f"train_examples={result.train_examples} val_examples={result.val_examples}"
    )
    print("[Next] Run the tracker with:")
    print("python run.py --detector vision " f'--vision-identity-source "{result.output_path.resolve()}"')


if __name__ == "__main__":
    main()
