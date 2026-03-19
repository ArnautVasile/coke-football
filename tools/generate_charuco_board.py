from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goal_tracker.charuco import CharucoSpec, generate_board_image, save_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a printable ChArUco board image + spec JSON")
    parser.add_argument("--output-image", default="data/calibration/charuco_board.png")
    parser.add_argument("--output-spec", default="", help="Optional JSON spec path (defaults next to image)")
    parser.add_argument("--dictionary", default="DICT_6X6_50")
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-length-mm", type=float, default=30.0)
    parser.add_argument("--marker-length-mm", type=float, default=22.0)
    parser.add_argument("--pixels-per-mm", type=float, default=12.0, help="~12 px/mm is close to 300 DPI")
    parser.add_argument("--margin-mm", type=float, default=10.0)
    parser.add_argument("--legacy-pattern", action="store_true", help="Generate legacy ChArUco pattern for older OpenCV compatibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = CharucoSpec(
        dictionary_name=args.dictionary,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_mm=args.square_length_mm,
        marker_length_mm=args.marker_length_mm,
        legacy_pattern=args.legacy_pattern,
    )
    output_image = Path(args.output_image)
    output_spec = Path(args.output_spec) if args.output_spec else output_image.with_suffix(".json")

    image = generate_board_image(spec, pixels_per_mm=args.pixels_per_mm, margin_mm=args.margin_mm)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_image), image):
        raise SystemExit(f"Failed to write board image to {output_image}")
    save_spec(output_spec, spec)

    board_w_mm, board_h_mm = spec.board_size_mm
    print(f"[ChArUco] image={output_image.resolve()}")
    print(f"[ChArUco] spec={output_spec.resolve()}")
    print(f"[ChArUco] dictionary={spec.dictionary_name}")
    print(
        "[ChArUco] board_size_mm="
        f"{board_w_mm:.1f}x{board_h_mm:.1f} square={spec.square_length_mm:.1f} marker={spec.marker_length_mm:.1f}"
    )
    print("[Next] Print the PNG at 100% scale with no fit-to-page scaling.")


if __name__ == "__main__":
    main()
