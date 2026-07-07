#!/usr/bin/env python3
"""Render per-pick overview images for all saved Franka grasp candidates.

This is a visualization-only tool. It consumes the debug NPZ files dumped by the
TORC grasp planner and never changes grasp selection, filtering, or execution.
For pick N it renders the candidates on the scene state before that pick:
initial scene for pick 1, state_(N-1).pkl for later picks when available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import render_selected_grasp_pose_from_dump as selected_pose


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(os.environ["TORC_SELECTED_GRASP_EXP_DIR"]).resolve()
DEBUG_DIR = EXP_DIR / "selected_grasp_debug"
OUT_DIR = PROJECT_ROOT / "phase4_artifacts" / "all_grasp_pose_overviews"
OUT_COMBINED = PROJECT_ROOT / "franka_all_grasp_pose_overviews.png"
OUT_MANIFEST = PROJECT_ROOT / "phase4_artifacts" / "all_grasp_pose_overviews_manifest.json"

RENDER_WIDTH = selected_pose.RENDER_WIDTH
RENDER_HEIGHT = selected_pose.RENDER_HEIGHT
HIDE_ROBOT = os.environ.get("TORC_HIDE_ROBOT_IN_GRASP_OVERVIEW", "1") != "0"


def hide_robot_geoms(model: mujoco.MjModel) -> None:
    robot_prefixes = (
        "robot0_",
        "gripper0_",
        "panda_",
        "link",
        "finger",
        "hand",
    )
    for geom_id in range(model.ngeom):
        geom_name = model.geom(geom_id).name or ""
        body_name = model.body(int(model.geom(geom_id).bodyid[0])).name or ""
        if geom_name.startswith(robot_prefixes) or body_name.startswith(robot_prefixes):
            model.geom_rgba[geom_id, 3] = 0.0


def add_candidate_frame(
    scene: mujoco.MjvScene,
    T: np.ndarray,
    rank: int,
    selected: bool = False,
) -> None:
    origin = np.asarray(T[:3, 3], dtype=np.float64)
    length = 0.085 if selected else 0.055
    radius = 0.006 if selected else 0.003
    alpha = 1.0 if selected else 0.50
    colors = [
        [1.0, 0.04, 0.04, alpha],
        [0.04, 1.0, 0.04, alpha],
        [0.04, 0.20, 1.0, alpha],
    ]
    for axis in range(3):
        if scene.ngeom >= len(scene.geoms):
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(colors[axis], dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            radius,
            origin,
            origin + length * np.asarray(T[:3, axis], dtype=np.float64),
        )
        scene.ngeom += 1

    sphere_rgba = [0.0, 1.0, 1.0, 1.0] if selected else [1.0, 1.0, 1.0, 0.35]
    selected_pose.add_sphere(scene, origin, 0.014 if selected else 0.007, sphere_rgba)


def pick_pose_arrays(arrays: np.lib.npyio.NpzFile) -> tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if "validated_candidate_matrices_sorted" in arrays:
        poses = np.asarray(arrays["validated_candidate_matrices_sorted"], dtype=np.float64)
        scores = np.asarray(arrays["validated_scores_sorted"], dtype=np.float64)
        obj_ids = np.asarray(arrays["validated_object_ids_sorted"], dtype=np.int64) if "validated_object_ids_sorted" in arrays else None
        return "validated_candidate_matrices_sorted", poses, scores, obj_ids
    if "validated_candidate_matrices" in arrays:
        poses = np.asarray(arrays["validated_candidate_matrices"], dtype=np.float64)
        scores = np.asarray(arrays["validated_candidate_scores"], dtype=np.float64) if "validated_candidate_scores" in arrays else None
        obj_ids = np.asarray(arrays["validated_candidate_object_ids"], dtype=np.int64) if "validated_candidate_object_ids" in arrays else None
        return "validated_candidate_matrices", poses, scores, obj_ids
    if "ik_candidate_matrices" in arrays:
        poses = np.asarray(arrays["ik_candidate_matrices"], dtype=np.float64)
        scores = np.asarray(arrays["ik_candidate_scores"], dtype=np.float64) if "ik_candidate_scores" in arrays else None
        obj_ids = np.asarray(arrays["ik_candidate_object_ids"], dtype=np.int64) if "ik_candidate_object_ids" in arrays else None
        return "ik_candidate_matrices", poses, scores, obj_ids
    return "none", np.empty((0, 4, 4), dtype=np.float64), None, None


def render_overview(debug_json: Path, out_png: Path) -> dict:
    debug_npz = debug_json.with_suffix(".npz")
    metadata = json.loads(debug_json.read_text(encoding="utf-8"))
    arrays = np.load(debug_npz)
    array_name, poses, scores, obj_ids = pick_pose_arrays(arrays)
    selected_index = int(metadata.get("selected_index_in_sorted_validated_candidates", 0))

    scene_xml, raw_state, state_source = selected_pose.state_for_pick(debug_json)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    if HIDE_ROBOT:
        hide_robot_geoms(model)
    data = mujoco.MjData(model)
    if raw_state is not None:
        mujoco.mj_setState(model, data, raw_state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
    renderer.update_scene(data, camera=selected_pose.camera())
    scene = renderer.scene
    for rank, T in enumerate(poses):
        add_candidate_frame(scene, T, rank, selected=(rank == selected_index))
    image = renderer.render()
    renderer.close()

    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle((0, 0, RENDER_WIDTH, 132), fill=(0, 0, 0, 155))
    draw.text((16, 12), f"All saved grasp poses: {debug_json.stem}", fill=(255, 255, 255, 255))
    draw.text(
        (16, 38),
        f"object={metadata.get('intended_object_name')} segment={metadata.get('selected_object_segment_id')} "
        f"poses={len(poses)} array={array_name} selected_rank={selected_index}",
        fill=(230, 245, 255, 255),
    )
    score_text = ""
    if scores is not None and len(scores):
        score_text = f" score_range=[{float(np.nanmin(scores)):.3f},{float(np.nanmax(scores)):.3f}] selected={float(scores[min(selected_index, len(scores)-1)]):.3f}"
    draw.text((16, 64), f"scene state: {state_source}{score_text}", fill=(255, 245, 190, 255))
    draw.text((16, 90), "small RGB frames: grasp candidates; cyan sphere/larger frame: selected candidate; robot hidden", fill=(255, 255, 255, 255))
    if obj_ids is not None and len(obj_ids):
        unique = sorted(set(int(v) for v in obj_ids.tolist()))
        draw.text((16, 114), f"candidate object ids: {unique[:18]}{' ...' if len(unique) > 18 else ''}", fill=(220, 220, 220, 255))
    img.save(out_png)

    return {
        "output_png": str(out_png),
        "source_debug_json": str(debug_json),
        "source_debug_npz": str(debug_npz),
        "scene_state_source": state_source,
        "array_name": array_name,
        "pose_count": int(len(poses)),
        "selected_rank": selected_index,
        "intended_object_name": metadata.get("intended_object_name"),
        "selected_object_segment_id": metadata.get("selected_object_segment_id"),
        "score_min": float(np.nanmin(scores)) if scores is not None and len(scores) else None,
        "score_max": float(np.nanmax(scores)) if scores is not None and len(scores) else None,
    }


def combine(paths: list[Path], output: Path) -> None:
    images = [Image.open(p).convert("RGB") for p in paths]
    width = max(im.width for im in images)
    height = sum(im.height for im in images)
    canvas = Image.new("RGB", (width, height), (10, 10, 10))
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(output)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rendered = []
    paths = []
    for debug_json in sorted(DEBUG_DIR.glob("pick_*_selected_grasp_debug.json")):
        out_png = OUT_DIR / debug_json.name.replace("_selected_grasp_debug.json", "_all_validated_poses.png")
        rendered.append(render_overview(debug_json, out_png))
        paths.append(out_png)
    if not paths:
        raise RuntimeError(f"no debug files found in {DEBUG_DIR}")
    combine(paths, OUT_COMBINED)
    OUT_MANIFEST.write_text(
        json.dumps(
            {
                "exp_dir": str(EXP_DIR),
                "output_combined": str(OUT_COMBINED),
                "output_dir": str(OUT_DIR),
                "robot_hidden": HIDE_ROBOT,
                "visualized_pose_stage": "validated grasps after IK / scene-pregrasp filtering, sorted when available",
                "rendered": rendered,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUT_COMBINED)


if __name__ == "__main__":
    main()
