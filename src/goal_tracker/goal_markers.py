from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class GoalMarker:
    marker_id: int
    name: str
    center_m: tuple[float, float, float]


@dataclass
class GoalMarkerLayout:
    dictionary_name: str = "DICT_6X6_50"
    marker_length_mm: float = 80.0
    goal_width_m: float = 7.32
    goal_height_m: float = 2.44
    scoring_plane_depth_m: float = 0.0
    opening_inset_left_m: float = 0.0
    opening_inset_right_m: float = 0.0
    opening_inset_top_m: float = 0.0
    opening_inset_bottom_m: float = 0.0
    markers: tuple[GoalMarker, ...] = ()

    @property
    def marker_length_m(self) -> float:
        return float(self.marker_length_mm) / 1000.0

    @property
    def opening_width_m(self) -> float:
        return max(
            0.01,
            float(self.goal_width_m)
            - max(0.0, float(self.opening_inset_left_m))
            - max(0.0, float(self.opening_inset_right_m)),
        )

    @property
    def opening_height_m(self) -> float:
        return max(
            0.01,
            float(self.goal_height_m)
            - max(0.0, float(self.opening_inset_top_m))
            - max(0.0, float(self.opening_inset_bottom_m)),
        )


def default_goal_marker_layout() -> GoalMarkerLayout:
    return GoalMarkerLayout(
        dictionary_name="DICT_6X6_50",
        marker_length_mm=80.0,
        goal_width_m=7.32,
        goal_height_m=2.44,
        scoring_plane_depth_m=0.0,
        opening_inset_left_m=0.0,
        opening_inset_right_m=0.0,
        opening_inset_top_m=0.0,
        opening_inset_bottom_m=0.0,
        markers=(
            GoalMarker(marker_id=10, name="top-left", center_m=(0.18, 0.16, 0.0)),
            GoalMarker(marker_id=11, name="top-right", center_m=(7.14, 0.16, 0.0)),
            GoalMarker(marker_id=12, name="mid-left", center_m=(0.18, 1.22, 0.0)),
            GoalMarker(marker_id=13, name="mid-right", center_m=(7.14, 1.22, 0.0)),
            GoalMarker(marker_id=14, name="bottom-left", center_m=(0.18, 2.28, 0.0)),
            GoalMarker(marker_id=15, name="bottom-right", center_m=(7.14, 2.28, 0.0)),
        ),
    )


def save_goal_marker_layout(path: Path, layout: GoalMarkerLayout) -> None:
    payload = {
        "dictionary_name": layout.dictionary_name,
        "marker_length_mm": layout.marker_length_mm,
        "goal_width_m": layout.goal_width_m,
        "goal_height_m": layout.goal_height_m,
        "scoring_plane_depth_m": layout.scoring_plane_depth_m,
        "opening_inset_left_m": layout.opening_inset_left_m,
        "opening_inset_right_m": layout.opening_inset_right_m,
        "opening_inset_top_m": layout.opening_inset_top_m,
        "opening_inset_bottom_m": layout.opening_inset_bottom_m,
        "markers": [asdict(marker) for marker in layout.markers],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_goal_marker_layout(path: Path) -> GoalMarkerLayout:
    payload = json.loads(path.read_text(encoding="utf-8"))
    markers = tuple(
        GoalMarker(
            marker_id=int(item["marker_id"]),
            name=str(item["name"]),
            center_m=(
                float(item["center_m"][0]),
                float(item["center_m"][1]),
                float(item["center_m"][2]),
            ),
        )
        for item in payload["markers"]
    )
    return GoalMarkerLayout(
        dictionary_name=str(payload.get("dictionary_name", "DICT_6X6_50")),
        marker_length_mm=float(payload.get("marker_length_mm", 80.0)),
        goal_width_m=float(payload.get("goal_width_m", 7.32)),
        goal_height_m=float(payload.get("goal_height_m", 2.44)),
        scoring_plane_depth_m=float(payload.get("scoring_plane_depth_m", 0.0)),
        opening_inset_left_m=float(payload.get("opening_inset_left_m", 0.0)),
        opening_inset_right_m=float(payload.get("opening_inset_right_m", 0.0)),
        opening_inset_top_m=float(payload.get("opening_inset_top_m", 0.0)),
        opening_inset_bottom_m=float(payload.get("opening_inset_bottom_m", 0.0)),
        markers=markers,
    )
