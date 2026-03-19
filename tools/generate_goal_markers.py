from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goal_tracker.charuco import create_dictionary
from goal_tracker.goal_markers import GoalMarkerLayout, default_goal_marker_layout, save_goal_marker_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate printable goal-frame ArUco markers and a placement diagram")
    parser.add_argument("--output-sheet", default="data/calibration/goal_markers_sheet.png")
    parser.add_argument("--output-sheet-2up", default="data/calibration/goal_markers_sheet_2up.png")
    parser.add_argument("--output-layout", default="data/calibration/goal_markers_layout.json")
    parser.add_argument("--output-diagram", default="data/calibration/goal_markers_layout.png")
    parser.add_argument("--output-dir", default="data/calibration/goal_markers", help="Directory for individual marker PNGs")
    parser.add_argument("--page-width-px", type=int, default=2480, help="A4 @ 300dpi width")
    parser.add_argument("--page-height-px", type=int, default=3508, help="A4 @ 300dpi height")
    parser.add_argument("--margin-px", type=int, default=120)
    parser.add_argument("--marker-gap-px", type=int, default=120)
    parser.add_argument("--quiet-zone-ratio", type=float, default=0.22, help="White border around each marker as a fraction of marker side")
    return parser.parse_args()


def draw_text(img: np.ndarray, text: str, xy: tuple[int, int], scale: float = 1.0, color: tuple[int, int, int] = (0, 0, 0), thickness: int = 2) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_dashed_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
    dash_px: int = 22,
    gap_px: int = 12,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    segments = []
    segments.append(((x1, y1), (x2, y1)))
    segments.append(((x2, y1), (x2, y2)))
    segments.append(((x2, y2), (x1, y2)))
    segments.append(((x1, y2), (x1, y1)))
    for (ax, ay), (bx, by) in segments:
        length = int(round(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5))
        if length <= 0:
            continue
        dx = (bx - ax) / float(length)
        dy = (by - ay) / float(length)
        pos = 0
        while pos < length:
            start = pos
            end = min(length, pos + dash_px)
            p1 = (int(round(ax + dx * start)), int(round(ay + dy * start)))
            p2 = (int(round(ax + dx * end)), int(round(ay + dy * end)))
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
            pos += dash_px + gap_px


def marker_image(dictionary, marker_id: int, side_px: int, quiet_zone_ratio: float = 0.22) -> np.ndarray:
    quiet = max(24, int(round(side_px * float(max(0.05, quiet_zone_ratio)))))
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    canvas = np.full((side_px + 2 * quiet, side_px + 2 * quiet), 255, dtype=np.uint8)
    canvas[quiet : quiet + side_px, quiet : quiet + side_px] = marker
    return canvas


def save_marker_pngs(layout: GoalMarkerLayout, output_dir: Path, side_px: int, quiet_zone_ratio: float) -> None:
    dictionary = create_dictionary(layout.dictionary_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    for marker in layout.markers:
        img = marker_image(dictionary, marker.marker_id, side_px, quiet_zone_ratio=quiet_zone_ratio)
        label_strip = np.full((120, img.shape[1], 3), 255, dtype=np.uint8)
        draw_text(label_strip, f"ID {marker.marker_id}", (30, 46), scale=1.0, thickness=2)
        draw_text(label_strip, marker.name, (30, 90), scale=0.85, thickness=2)
        color_marker = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out = np.vstack([color_marker, label_strip])
        cv2.imwrite(str(output_dir / f"marker_{marker.marker_id}_{marker.name}.png"), out)


def build_sheet(layout: GoalMarkerLayout, page_w: int, page_h: int, margin_px: int, gap_px: int, quiet_zone_ratio: float) -> np.ndarray:
    canvas = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
    usable_w = page_w - 2 * margin_px
    usable_h = page_h - 2 * margin_px
    label_h = 120
    q = float(max(0.05, quiet_zone_ratio))
    tile_scale = 1.0 + 2.0 * q
    max_side_w = int((usable_w - gap_px) / max(2.0 * tile_scale, 1.0))
    max_side_h = int((usable_h - 2 * gap_px - 3 * label_h - 130) / max(3.0 * tile_scale, 1.0))
    marker_side = max(320, min(max_side_w, max_side_h))
    marker_img = marker_image(create_dictionary(layout.dictionary_name), layout.markers[0].marker_id, marker_side, quiet_zone_ratio=quiet_zone_ratio)
    tile_w = marker_img.shape[1]
    tile_h = marker_img.shape[0]

    dictionary = create_dictionary(layout.dictionary_name)
    positions: list[tuple[int, int]] = []
    start_x = margin_px + (usable_w - (2 * tile_w + gap_px)) // 2
    start_y = margin_px + 130
    for row in range(3):
        for col in range(2):
            x = start_x + col * (tile_w + gap_px)
            y = start_y + row * (tile_h + 120 + gap_px)
            positions.append((x, y))

    draw_text(canvas, "Goal Frame Markers", (margin_px, 64), scale=1.2, thickness=3)
    draw_text(
        canvas,
        f"Dictionary {layout.dictionary_name} | Marker size {layout.marker_length_mm:.0f} mm + white quiet zone | Print at 100% / no fill page",
        (margin_px, 104),
        scale=0.72,
        thickness=2,
    )

    for marker, (x, y) in zip(layout.markers, positions):
        img = marker_image(dictionary, marker.marker_id, marker_side, quiet_zone_ratio=quiet_zone_ratio)
        canvas[y : y + img.shape[0], x : x + img.shape[1]] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        draw_text(canvas, f"ID {marker.marker_id}", (x + 10, y + img.shape[0] + 36), scale=0.95, thickness=2)
        draw_text(canvas, marker.name, (x + 10, y + img.shape[0] + 74), scale=0.82, thickness=2)
    return canvas


def build_two_up_sheet(layout: GoalMarkerLayout, page_w: int, page_h: int, margin_px: int, gap_px: int, quiet_zone_ratio: float) -> tuple[np.ndarray, float, float]:
    canvas = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
    label_h = 90
    usable_w = page_w - 2 * margin_px
    usable_h = page_h - 2 * margin_px
    px_per_mm = page_w / 210.0
    marker_side = int(round(layout.marker_length_mm * px_per_mm))
    sample = marker_image(
        create_dictionary(layout.dictionary_name),
        layout.markers[0].marker_id,
        marker_side,
        quiet_zone_ratio=quiet_zone_ratio,
    )
    tile_w = sample.shape[1]
    tile_h = sample.shape[0]
    total_h = 2 * tile_h + 2 * label_h + gap_px + 120
    if tile_w > usable_w or total_h > usable_h:
        raise SystemExit(
            "The configured marker size + quiet zone do not fit two-per-page on A4. "
            "Reduce --quiet-zone-ratio or marker size."
        )

    start_x = margin_px + (usable_w - tile_w) // 2
    start_y = margin_px + 80
    draw_text(canvas, "Goal Frame Markers (2 per page)", (margin_px, 58), scale=1.1, thickness=3)
    draw_text(canvas, "Print each page at 100% / no fill page. Cut on dashed lines.", (margin_px, 95), scale=0.72, thickness=2)

    dictionary = create_dictionary(layout.dictionary_name)
    for index, marker in enumerate(layout.markers[:2]):
        x = start_x
        y = start_y + index * (tile_h + label_h + gap_px)
        img = marker_image(dictionary, marker.marker_id, marker_side, quiet_zone_ratio=quiet_zone_ratio)
        canvas[y : y + img.shape[0], x : x + img.shape[1]] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        draw_dashed_rect(canvas, (x - 10, y - 10), (x + img.shape[1] + 10, y + img.shape[0] + 10), thickness=2)
        draw_text(canvas, f"ID {marker.marker_id}", (x + 10, y + img.shape[0] + 34), scale=0.95, thickness=2)
        draw_text(canvas, marker.name, (x + 10, y + img.shape[0] + 72), scale=0.82, thickness=2)

    printed_total_mm = tile_w / px_per_mm
    printed_black_mm = marker_side / px_per_mm
    return canvas, float(printed_total_mm), float(printed_black_mm)


def save_two_up_pages(layout: GoalMarkerLayout, output_dir: Path, page_w: int, page_h: int, margin_px: int, gap_px: int, quiet_zone_ratio: float) -> tuple[float, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    printed_total_mm = 0.0
    printed_black_mm = 0.0
    for page_index in range(0, len(layout.markers), 2):
        page_markers = GoalMarkerLayout(
            dictionary_name=layout.dictionary_name,
            marker_length_mm=layout.marker_length_mm,
            goal_width_m=layout.goal_width_m,
            goal_height_m=layout.goal_height_m,
            markers=tuple(layout.markers[page_index : page_index + 2]),
        )
        page, printed_total_mm, printed_black_mm = build_two_up_sheet(page_markers, page_w, page_h, margin_px, gap_px, quiet_zone_ratio)
        out_path = output_dir / f"goal_markers_page_{page_index // 2 + 1}.png"
        cv2.imwrite(str(out_path), page)
    return printed_total_mm, printed_black_mm


def build_layout_diagram(layout: GoalMarkerLayout, width: int = 1800, height: int = 900) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    pad_x = 180
    pad_y = 120
    goal_w = width - 2 * pad_x
    goal_h = height - 2 * pad_y
    x0, y0 = pad_x, pad_y
    x1, y1 = x0 + goal_w, y0 + goal_h
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 170, 0), 6)
    draw_text(canvas, "Goal Marker Placement", (80, 60), scale=1.4, thickness=3)
    draw_text(canvas, "Mount markers on flat tabs in the same front plane as the goal opening.", (80, 100), scale=0.8, thickness=2)

    for marker in layout.markers:
        mx = x0 + int(round((marker.center_m[0] / layout.goal_width_m) * goal_w))
        my = y0 + int(round((marker.center_m[1] / layout.goal_height_m) * goal_h))
        cv2.rectangle(canvas, (mx - 42, my - 42), (mx + 42, my + 42), (0, 0, 0), 3)
        cv2.circle(canvas, (mx, my), 5, (0, 0, 255), -1)
        draw_text(canvas, f"ID {marker.marker_id}", (mx + 54, my - 8), scale=0.8, thickness=2)
        draw_text(canvas, marker.name, (mx + 54, my + 24), scale=0.72, thickness=2)

    draw_text(canvas, "0,0", (x0 - 54, y0 - 10), scale=0.74, thickness=2)
    draw_text(canvas, f"{layout.goal_width_m:.2f}m", (x1 - 90, y0 - 10), scale=0.74, thickness=2)
    draw_text(canvas, f"{layout.goal_height_m:.2f}m", (x1 + 20, y1), scale=0.74, thickness=2)
    return canvas


def main() -> None:
    args = parse_args()
    layout = default_goal_marker_layout()

    sheet_path = Path(args.output_sheet)
    sheet_2up_path = Path(args.output_sheet_2up)
    layout_path = Path(args.output_layout)
    diagram_path = Path(args.output_diagram)
    marker_dir = Path(args.output_dir)

    sheet = build_sheet(layout, args.page_width_px, args.page_height_px, args.margin_px, args.marker_gap_px, args.quiet_zone_ratio)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(sheet_path), sheet):
        raise SystemExit(f"Failed to write marker sheet to {sheet_path}")

    individual_side = max(700, int(round(layout.marker_length_mm * 12.0)))
    save_marker_pngs(layout, marker_dir, side_px=individual_side, quiet_zone_ratio=args.quiet_zone_ratio)
    save_goal_marker_layout(layout_path, layout)
    printed_total_mm, printed_black_mm = save_two_up_pages(
        layout,
        output_dir=sheet_2up_path.parent / "goal_markers_pages",
        page_w=args.page_width_px,
        page_h=args.page_height_px,
        margin_px=args.margin_px,
        gap_px=args.marker_gap_px,
        quiet_zone_ratio=args.quiet_zone_ratio,
    )

    diagram = build_layout_diagram(layout)
    if not cv2.imwrite(str(diagram_path), diagram):
        raise SystemExit(f"Failed to write layout diagram to {diagram_path}")

    print(f"[Markers] sheet={sheet_path.resolve()}")
    print(f"[Markers] two_up_dir={(sheet_2up_path.parent / 'goal_markers_pages').resolve()}")
    print(f"[Markers] layout={layout_path.resolve()}")
    print(f"[Markers] diagram={diagram_path.resolve()}")
    print(f"[Markers] individual_dir={marker_dir.resolve()}")
    print(f"[Markers] printed_black_marker_size={printed_black_mm / 10.0:.2f}cm")
    print(f"[Markers] printed_total_size_per_marker={printed_total_mm / 10.0:.2f}cm including white quiet zone")
    print(
        "[Markers] ids="
        + ", ".join(f"{marker.marker_id}:{marker.name}" for marker in layout.markers)
    )
    print("[Next] Print the sheet at 100% scale. Do not use fill page.")


if __name__ == "__main__":
    main()
