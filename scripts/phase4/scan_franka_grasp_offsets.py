#!/usr/bin/env python3
"""Offline Franka grasp offset calibration for a captured TORC pick.

The scan keeps TORC's selected candidates unchanged and only evaluates small
Franka TCP offsets in the adapter frame: local X for closing-axis centering and
local Z for approach/contact retreat.  It uses the current Panda pad dimensions
as the scoring geometry.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXP_DIR = (
    PROJECT_ROOT
    / "phase4_artifacts/torc_franka_pipeline_1783345171/"
    "exp_2026-07-06_17-39-44__difficult_116_obj_000070_0_dg_only"
)


def _load_exp_dir() -> Path:
    override = os.environ.get("TORC_SELECTED_GRASP_EXP_DIR")
    return Path(override).expanduser().resolve() if override else DEFAULT_EXP_DIR


def _pad_geometry() -> dict[str, float]:
    return {
        "opening_half_m": float(os.environ.get("TORC_FRANKA_PAD_OPENING_HALF_M", "0.040")),
        "center_extra_x_m": float(os.environ.get("TORC_FRANKA_PAD_CENTER_EXTRA_X_M", "0.0035")),
        "half_x_m": float(os.environ.get("TORC_FRANKA_PAD_HALF_X_M", "0.0040")),
        "half_y_m": float(os.environ.get("TORC_FRANKA_PAD_HALF_Y_M", "0.0080")),
        "half_z_m": float(os.environ.get("TORC_FRANKA_PAD_HALF_Z_M", "0.0080")),
        "front_z_m": float(os.environ.get("TORC_FRANKA_PAD_FRONT_Z_M", "0.0044")),
        "penetration_margin_m": float(os.environ.get("TORC_FRANKA_PAD_PENETRATION_MARGIN_M", "0.0010")),
    }


def _local_points(points: np.ndarray, pose: np.ndarray, dx: float, dz: float) -> np.ndarray:
    adjusted = np.array(pose, dtype=np.float64, copy=True)
    adjusted[:3, 3] += dx * adjusted[:3, 0] + dz * adjusted[:3, 2]
    return (points - adjusted[:3, 3]) @ adjusted[:3, :3]


def _near_grasp_points(local: np.ndarray) -> np.ndarray:
    # Use a robust window around the open fingers.  This avoids distant points
    # from the same segmented object dominating the centering metric.
    mask = (
        (np.abs(local[:, 0]) <= 0.09)
        & (np.abs(local[:, 1]) <= 0.075)
        & (local[:, 2] >= -0.035)
        & (local[:, 2] <= 0.16)
    )
    near = local[mask]
    return near if len(near) >= 20 else local


def _pad_penetration_count(local: np.ndarray, geom: dict[str, float]) -> int:
    half_extents = np.array(
        [
            geom["half_x_m"] + geom["penetration_margin_m"],
            geom["half_y_m"] + geom["penetration_margin_m"],
            geom["half_z_m"] + geom["penetration_margin_m"],
        ],
        dtype=np.float64,
    )
    z_center = geom["front_z_m"] - geom["half_z_m"]
    x_center = geom["opening_half_m"] + geom["center_extra_x_m"]
    centers = np.array(
        [[-x_center, 0.0, z_center], [x_center, 0.0, z_center]],
        dtype=np.float64,
    )
    inside = np.zeros(len(local), dtype=bool)
    for center in centers:
        inside |= np.all(np.abs(local - center) <= half_extents, axis=1)
    return int(np.count_nonzero(inside))


def _score_pose(points: np.ndarray, pose: np.ndarray, dx: float, dz: float, geom: dict[str, float]) -> dict:
    local = _local_points(points, pose, dx, dz)
    near = _near_grasp_points(local)
    x05, x50, x95 = np.percentile(near[:, 0], [5, 50, 95])
    y05, y95 = np.percentile(near[:, 1], [5, 95])
    z05, z50 = np.percentile(near[:, 2], [5, 50])
    x_center = 0.5 * (x05 + x95)
    x_span = x95 - x05
    y_span = y95 - y05
    penetration = _pad_penetration_count(near, geom)

    desired_z05 = geom["front_z_m"] + float(os.environ.get("TORC_FRANKA_SCAN_TARGET_CLEARANCE_M", "0.002"))
    z_error = abs(z05 - desired_z05)
    x_error = abs(x_center)
    opening_half = geom["opening_half_m"]
    span_error = max(0.0, x_span - 2.0 * (opening_half - geom["half_x_m"]))

    score = (
        1000.0 * penetration
        + 80.0 * x_error
        + 60.0 * z_error
        + 30.0 * span_error
        + 10.0 * max(0.0, y_span - 0.06)
    )
    return {
        "score": float(score),
        "dx_m": float(dx),
        "dz_m": float(dz),
        "penetration_points": penetration,
        "near_points": int(len(near)),
        "x05_m": float(x05),
        "x50_m": float(x50),
        "x95_m": float(x95),
        "x_center_m": float(x_center),
        "x_span_m": float(x_span),
        "y_span_m": float(y_span),
        "z05_m": float(z05),
        "z50_m": float(z50),
        "z_error_m": float(z_error),
    }


def main() -> int:
    exp_dir = _load_exp_dir()
    npz_path = exp_dir / "selected_grasp_debug/pick_01_selected_grasp_debug.npz"
    json_path = exp_dir / "selected_grasp_debug/pick_01_selected_grasp_debug.json"
    points_path = exp_dir / "output_target_points.txt"
    data = np.load(npz_path, allow_pickle=True)
    meta = json.loads(json_path.read_text())
    points = np.loadtxt(points_path, dtype=np.float64).reshape(-1, 3)

    poses = data["validated_candidate_matrices_sorted"]
    scores = data["validated_scores_sorted"]
    obj_ids = data["validated_object_ids_sorted"]
    source_indices = data["validated_source_indices_sorted"]
    geom = _pad_geometry()

    dx_values = np.arange(-0.018, 0.0181, 0.002)
    dz_values = np.arange(-0.018, 0.0181, 0.002)
    results = []
    for candidate_index, pose in enumerate(poses):
        for dx in dx_values:
            for dz in dz_values:
                item = _score_pose(points, pose, float(dx), float(dz), geom)
                item.update(
                    {
                        "candidate_index": int(candidate_index),
                        "candidate_score": float(scores[candidate_index]),
                        "candidate_object_id": int(obj_ids[candidate_index]),
                        "candidate_source_index": int(source_indices[candidate_index]),
                    }
                )
                results.append(item)

    results.sort(key=lambda item: item["score"])
    selected_candidate = int(meta.get("selected_index_in_sorted_validated_candidates", 0))
    selected_results = [r for r in results if r["candidate_index"] == selected_candidate]
    selected_results.sort(key=lambda item: item["score"])

    output = {
        "exp_dir": str(exp_dir),
        "source_npz": str(npz_path),
        "source_points": str(points_path),
        "scene_name": meta.get("scene_name"),
        "target_object": meta.get("target_object"),
        "dg_chosen_segment_id": meta.get("dg_chosen_segment_id"),
        "selected_object_segment_id": meta.get("selected_object_segment_id"),
        "selected_candidate_index": selected_candidate,
        "pad_geometry": geom,
        "scan_grid": {
            "dx_values_m": [float(x) for x in dx_values],
            "dz_values_m": [float(z) for z in dz_values],
        },
        "best_overall": results[:20],
        "best_for_selected_candidate": selected_results[:20],
    }
    out_path = PROJECT_ROOT / "franka_grasp_offset_scan.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(out_path)
    print(json.dumps({"best_overall": results[0], "best_selected": selected_results[0]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
