#!/usr/bin/env python3
"""Render raw CGN grasp poses as small gripper glyphs without the robot arm.

This visualizes the pre-IK, pre-scene-filter grasp set saved by TORC's grasp
planner (`raw_cgn_grasp_matrices`). It is intentionally separate from the
selected Franka pose renderer, which still shows the full arm for one pose.
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
from output_paths import artifact_path, result_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(os.environ["TORC_SELECTED_GRASP_EXP_DIR"]).resolve()
DEBUG_DIR = EXP_DIR / "selected_grasp_debug"
OUT_DIR = artifact_path("raw_cgn_gripper_overviews")
OUT_COMBINED = result_path("franka_raw_cgn_gripper_overviews.png")
OUT_MANIFEST = artifact_path("raw_cgn_gripper_overviews_manifest.json")

RENDER_WIDTH = selected_pose.RENDER_WIDTH
RENDER_HEIGHT = selected_pose.RENDER_HEIGHT
GRIPPER_OPENING_M = float(os.environ.get("TORC_RAW_GRIPPER_GLYPH_OPENING_M", "0.075"))
FINGER_LENGTH_M = float(os.environ.get("TORC_RAW_GRIPPER_GLYPH_FINGER_LENGTH_M", "0.055"))
FINGER_RADIUS_M = float(os.environ.get("TORC_RAW_GRIPPER_GLYPH_RADIUS_M", "0.0016"))
MAX_GRASPS = int(os.environ.get("TORC_RAW_GRIPPER_GLYPH_MAX_GRASPS", "0"))


PALETTE = np.asarray(
    [
        [0.00, 0.85, 1.00, 0.42],
        [1.00, 0.35, 0.15, 0.42],
        [0.10, 1.00, 0.25, 0.42],
        [1.00, 0.15, 0.65, 0.42],
        [0.95, 0.90, 0.10, 0.42],
        [0.50, 0.35, 1.00, 0.42],
        [0.10, 0.75, 0.65, 0.42],
        [1.00, 0.55, 0.90, 0.42],
        [0.85, 0.85, 0.85, 0.42],
    ],
    dtype=np.float32,
)


def hide_robot_geoms(model: mujoco.MjModel) -> None:
    prefixes = ("robot0_", "gripper0_", "panda_", "link", "finger", "hand")
    for geom_id in range(model.ngeom):
        geom_name = model.geom(geom_id).name or ""
        body_name = model.body(int(model.geom(geom_id).bodyid[0])).name or ""
        if geom_name.startswith(prefixes) or body_name.startswith(prefixes):
            model.geom_rgba[geom_id, 3] = 0.0


def add_cylinder(scene: mujoco.MjvScene, p0: np.ndarray, p1: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= len(scene.geoms):
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        radius,
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    scene.ngeom += 1


def add_gripper_glyph(scene: mujoco.MjvScene, T: np.ndarray, rgba: np.ndarray) -> None:
    center = np.asarray(T[:3, 3], dtype=np.float64)
    closing_axis = np.asarray(T[:3, 0], dtype=np.float64)
    approach_axis = np.asarray(T[:3, 2], dtype=np.float64)
    closing_axis /= max(np.linalg.norm(closing_axis), 1e-9)
    approach_axis /= max(np.linalg.norm(approach_axis), 1e-9)

    half_open = 0.5 * GRIPPER_OPENING_M
    palm_center = center - approach_axis * FINGER_LENGTH_M
    left_tip = center + closing_axis * half_open
    right_tip = center - closing_axis * half_open
    left_base = palm_center + closing_axis * half_open
    right_base = palm_center - closing_axis * half_open

    add_cylinder(scene, left_base, left_tip, FINGER_RADIUS_M, rgba)
    add_cylinder(scene, right_base, right_tip, FINGER_RADIUS_M, rgba)
    add_cylinder(scene, left_base, right_base, FINGER_RADIUS_M, rgba)


def load_raw_grasps(arrays: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses = np.asarray(arrays["raw_cgn_grasp_matrices"], dtype=np.float64)
    scores = np.asarray(arrays["raw_cgn_scores"], dtype=np.float64)
    object_ids = np.asarray(arrays["raw_cgn_object_ids"], dtype=np.int64)
    finite = np.isfinite(poses).all(axis=(1, 2))
    poses = poses[finite]
    scores = scores[finite]
    object_ids = object_ids[finite]
    if MAX_GRASPS > 0 and len(poses) > MAX_GRASPS:
        order = np.argsort(-scores)[:MAX_GRASPS]
        poses = poses[order]
        scores = scores[order]
        object_ids = object_ids[order]
    return poses, scores, object_ids


def render_pick(debug_json: Path, out_png: Path) -> dict:
    debug_npz = debug_json.with_suffix(".npz")
    metadata = json.loads(debug_json.read_text(encoding="utf-8"))
    arrays = np.load(debug_npz)
    poses, scores, object_ids = load_raw_grasps(arrays)

    scene_xml, raw_state, state_source = selected_pose.state_for_pick(debug_json)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    hide_robot_geoms(model)
    data = mujoco.MjData(model)
    if raw_state is not None:
        mujoco.mj_setState(model, data, raw_state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)

    max_geom = max(20000, int(len(poses) * 3 + model.ngeom + 2000))
    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH, max_geom=max_geom)
    renderer.update_scene(data, camera=selected_pose.camera())
    scene = renderer.scene
    for T, object_id in zip(poses, object_ids):
        rgba = PALETTE[int(object_id) % len(PALETTE)]
        add_gripper_glyph(scene, T, rgba)
    image = renderer.render()
    renderer.close()

    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle((0, 0, RENDER_WIDTH, 132), fill=(0, 0, 0, 155))
    draw.text((16, 12), f"Raw CGN grasp poses as grippers: {debug_json.stem}", fill=(255, 255, 255, 255))
    draw.text(
        (16, 38),
        f"pre-IK/pre-scene-filter raw_cgn_grasp_matrices count={len(poses)} "
        f"intended_object={metadata.get('intended_object_name')} segment={metadata.get('selected_object_segment_id')}",
        fill=(230, 245, 255, 255),
    )
    if len(scores):
        draw.text(
            (16, 64),
            f"score_range=[{float(np.nanmin(scores)):.3f},{float(np.nanmax(scores)):.3f}] "
            f"object_ids={sorted(set(int(v) for v in object_ids.tolist()))[:24]}",
            fill=(255, 245, 190, 255),
        )
    draw.text((16, 90), f"scene state: {state_source}", fill=(255, 255, 255, 255))
    draw.text((16, 114), "robot hidden; each colored glyph is one raw generated grasp pose", fill=(225, 225, 225, 255))
    img.save(out_png)

    return {
        "output_png": str(out_png),
        "source_debug_json": str(debug_json),
        "source_debug_npz": str(debug_npz),
        "scene_state_source": state_source,
        "pose_stage": "raw_cgn_grasp_matrices",
        "pose_count": int(len(poses)),
        "max_grasps_limit": MAX_GRASPS,
        "intended_object_name": metadata.get("intended_object_name"),
        "selected_object_segment_id": metadata.get("selected_object_segment_id"),
        "object_ids": sorted(set(int(v) for v in object_ids.tolist())),
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
        out_png = OUT_DIR / debug_json.name.replace("_selected_grasp_debug.json", "_raw_cgn_grippers.png")
        rendered.append(render_pick(debug_json, out_png))
        paths.append(out_png)
    if not paths:
        raise RuntimeError(f"no selected grasp debug JSON files found in {DEBUG_DIR}")
    combine(paths, OUT_COMBINED)
    OUT_MANIFEST.write_text(
        json.dumps(
            {
                "exp_dir": str(EXP_DIR),
                "output_combined": str(OUT_COMBINED),
                "output_dir": str(OUT_DIR),
                "robot_hidden": True,
                "pose_stage": "raw_cgn_grasp_matrices before IK / scene-pregrasp filtering",
                "gripper_glyph": {
                    "opening_m": GRIPPER_OPENING_M,
                    "finger_length_m": FINGER_LENGTH_M,
                    "radius_m": FINGER_RADIUS_M,
                },
                "rendered": rendered,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUT_COMBINED)


if __name__ == "__main__":
    main()
