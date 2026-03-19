from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goal_tracker.yolo_ball_detection import resolve_yolo_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and benchmark YOLO models to ONNX/CoreML")
    parser.add_argument("--model", default="yolo26s.pt", help="Path to YOLO .pt model")
    parser.add_argument("--formats", default="onnx,coreml", help="Comma-separated export formats")
    parser.add_argument("--imgsz", type=int, default=640, help="Benchmark/export image size")
    parser.add_argument("--device", default="", help="Preferred device for benchmark/export (cpu, mps, cuda:0)")
    parser.add_argument("--data", default="", help="Optional dataset yaml for benchmark")
    parser.add_argument("--batch", type=int, default=1, help="Export batch size")
    parser.add_argument("--half", action="store_true", help="Request FP16 where supported")
    parser.add_argument("--int8", action="store_true", help="Request INT8 where supported")
    parser.add_argument("--nms", action="store_true", help="Include NMS in exported model where supported")
    parser.add_argument("--opset", type=int, default=0, help="Optional ONNX opset override for export")
    parser.add_argument("--no-simplify", dest="simplify", action="store_false", help="Disable ONNX simplify pass")
    parser.set_defaults(simplify=True)
    parser.add_argument("--skip-export", action="store_true", help="Skip export step")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark step")
    parser.add_argument("--check", action="store_true", help="Only print environment/dependency status")
    parser.add_argument("--name", default="", help="Optional run name")
    parser.add_argument("--output-dir", default="data/benchmarks", help="Directory for benchmark results")
    parser.add_argument("--verbose", action="store_true", help="Enable Ultralytics benchmark verbosity")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def detect_environment(requested_device: str | None) -> dict[str, Any]:
    py_tuple = sys.version_info[:3]
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "python_tuple": list(py_tuple),
        "requested_device": (requested_device or "").strip() or "default",
        "resolved_device": resolve_yolo_device(requested_device),
        "packages": {
            "ultralytics": package_version("ultralytics"),
            "torch": package_version("torch"),
            "onnx": package_version("onnx"),
            "onnxruntime": package_version("onnxruntime"),
            "onnxruntime_silicon_legacy": package_version("onnxruntime-silicon"),
            "coremltools": package_version("coremltools"),
        },
    }
    try:
        import torch  # type: ignore

        mps_backend = getattr(torch.backends, "mps", None)
        env["torch"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_built": bool(mps_backend) and bool(mps_backend.is_built()),
            "mps_available": bool(mps_backend) and bool(mps_backend.is_available()),
        }
    except Exception as exc:
        env["torch_error"] = repr(exc)
    return env


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{str(k): normalize_scalar(v) for k, v in row.items()} for row in rows]


def dataframe_rows(df: Any) -> list[dict[str, Any]]:
    if df is None:
        return []
    if hasattr(df, "to_dicts"):
        return normalize_rows(df.to_dicts())
    if isinstance(df, list):
        return normalize_rows(df)
    return [{"value": normalize_scalar(df)}]


def default_run_name(model_path: str, formats: list[str]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{Path(model_path).stem}_{'-'.join(formats)}_{stamp}"


def print_environment_summary(env: dict[str, Any]) -> None:
    print(f"[Env] platform={env['platform']}")
    print(f"[Env] python={env['python']} machine={env['machine']}")
    print(
        f"[Env] requested_device={env['requested_device']} resolved_device={env['resolved_device'] or 'default'}"
    )
    torch_info = env.get("torch")
    if isinstance(torch_info, dict):
        print(
            f"[Env] torch cuda={torch_info['cuda_available']} "
            f"mps_built={torch_info['mps_built']} mps_available={torch_info['mps_available']}"
        )
    packages = env.get("packages", {})
    for name in ("ultralytics", "torch", "onnx", "onnxruntime", "coremltools"):
        print(f"[Env] {name}={packages.get(name) or 'not-installed'}")
    legacy_ort = packages.get("onnxruntime_silicon_legacy")
    if legacy_ort:
        print(f"[Env] onnxruntime-silicon(legacy)={legacy_ort}")

    if env["machine"] == "arm64" and "macOS" in env["platform"]:
        resolved = env["resolved_device"]
        mps_available = isinstance(torch_info, dict) and bool(torch_info.get("mps_available"))
        if not mps_available:
            print("[Hint] Apple Silicon detected but PyTorch MPS is unavailable. CoreML is likely the best acceleration path.")
        if resolved in (None, "cpu"):
            print("[Hint] Benchmark with ONNX and CoreML exports rather than expecting PyTorch GPU speedups.")
        if legacy_ort:
            print("[Hint] onnxruntime-silicon is legacy. Prefer the current official onnxruntime package.")
        py_tuple = tuple(env.get("python_tuple", []))
        if py_tuple >= (3, 14):
            print("[Hint] coremltools does not currently publish macOS wheels for Python 3.14. Use Python 3.13 for CoreML export.")


def should_skip_coreml(env: dict[str, Any], formats: list[str]) -> bool:
    if "coreml" not in formats:
        return False
    py_tuple = tuple(env.get("python_tuple", []))
    return "macOS" in env.get("platform", "") and py_tuple >= (3, 14)


def export_one(model: Any, fmt: str, args: argparse.Namespace, resolved_device: str | None) -> dict[str, Any]:
    export_kwargs: dict[str, Any] = {
        "format": fmt,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "half": args.half,
        "int8": args.int8,
        "nms": args.nms,
    }
    if resolved_device:
        export_kwargs["device"] = resolved_device
    if fmt == "onnx":
        export_kwargs["simplify"] = args.simplify
        if args.opset > 0:
            export_kwargs["opset"] = args.opset

    clean_kwargs = {k: v for k, v in export_kwargs.items() if v not in ("", None, False)}
    print(f"[Export] format={fmt} kwargs={clean_kwargs}")
    try:
        path = model.export(**clean_kwargs)
        return {"format": fmt, "ok": True, "path": str(path), "kwargs": clean_kwargs}
    except Exception as exc:
        return {"format": fmt, "ok": False, "error": repr(exc), "kwargs": clean_kwargs}


def benchmark_one(model: Any, fmt: str, args: argparse.Namespace, resolved_device: str | None) -> dict[str, Any]:
    bench_kwargs: dict[str, Any] = {
        "format": fmt,
        "imgsz": args.imgsz,
        "verbose": args.verbose,
    }
    if args.data:
        bench_kwargs["data"] = args.data
    if resolved_device:
        bench_kwargs["device"] = resolved_device
    if args.half:
        bench_kwargs["half"] = True
    if args.int8:
        bench_kwargs["int8"] = True

    print(f"[Benchmark] format={fmt} kwargs={bench_kwargs}")
    try:
        table = model.benchmark(**bench_kwargs)
        return {"format": fmt, "ok": True, "rows": dataframe_rows(table), "kwargs": bench_kwargs}
    except Exception as exc:
        return {"format": fmt, "ok": False, "error": repr(exc), "kwargs": bench_kwargs}


def write_summary_csv(path: Path, payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    exports = {item["format"]: item for item in payload.get("exports", [])}
    for bench in payload.get("benchmarks", []):
        fmt = bench["format"]
        export_info = exports.get(fmt, {})
        if bench.get("rows"):
            for row in bench["rows"]:
                flat = {
                    "format": fmt,
                    "export_ok": export_info.get("ok"),
                    "export_path": export_info.get("path", ""),
                    "benchmark_ok": bench.get("ok"),
                }
                flat.update({k: normalize_scalar(v) for k, v in row.items()})
                rows.append(flat)
        else:
            rows.append(
                {
                    "format": fmt,
                    "export_ok": export_info.get("ok"),
                    "export_path": export_info.get("path", ""),
                    "benchmark_ok": bench.get("ok"),
                    "error": bench.get("error", ""),
                }
            )

    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    if not formats:
        raise SystemExit("No formats selected. Example: --formats onnx,coreml")

    env = detect_environment(args.device)
    print_environment_summary(env)
    if args.check:
        return

    if should_skip_coreml(env, formats):
        print("[Warn] Skipping CoreML in this environment because Python 3.14 is unsupported by coremltools wheels.")
        formats = [fmt for fmt in formats if fmt != "coreml"]
        if not formats:
            raise SystemExit("No runnable formats remain in this environment. Use Python 3.13 for CoreML export.")

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Install with: pip install -r requirements-yolo.txt") from exc

    model = YOLO(args.model)
    resolved_device = env["resolved_device"]
    run_name = args.name.strip() or default_run_name(args.model, formats)

    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "model": args.model,
        "formats": formats,
        "imgsz": args.imgsz,
        "environment": env,
        "exports": [],
        "benchmarks": [],
    }

    for fmt in formats:
        if not args.skip_export:
            payload["exports"].append(export_one(model, fmt, args, resolved_device))
        if not args.skip_benchmark:
            payload["benchmarks"].append(benchmark_one(model, fmt, args, resolved_device))

    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{run_name}.json"
    csv_path = out_dir / f"{run_name}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary_csv(csv_path, payload)

    print(f"[Saved] json={json_path}")
    if csv_path.exists():
        print(f"[Saved] csv={csv_path}")


if __name__ == "__main__":
    main()
