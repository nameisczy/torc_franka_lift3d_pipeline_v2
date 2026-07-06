#!/usr/bin/env python3
"""Render Franka at the selected grasp pose dumped by the TORC pipeline.

This does not select, filter, or modify a grasp. It only visualizes the exact
Franka joint target and grasp matrix written by curobo_open_loop.py during the
same TORC pipeline path used for Phase 4.3.
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

import render_franka_asset_alignment as phase41


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = (
    PROJECT_ROOT
    / "phase4_artifacts/torc_franka_pipeline_1783268501"
    / "exp_2026-07-05_20-21-53__difficult_116_obj_000070_0_dg_only"
)
EXP_DIR = Path(os.environ.get("TORC_SELECTED_GRASP_EXP_DIR", str(DEFAULT_RUN)))
DEBUG_DIR = EXP_DIR / "selected_grasp_debug"
DEBUG_JSON = DEBUG_DIR / "pick_01_selected_grasp_debug.json"
DEBUG_NPZ = DEBUG_DIR / "pick_01_selected_grasp_debug.npz"
OUT_PNG = PROJECT_ROOT / "franka_selected_grasp_pose.png"
OUT_MANIFEST = PROJECT_ROOT / "phase4_artifacts/selected_grasp_pose_render_manifest.json"
POSE_XML = PROJECT_ROOT / "phase4_artifacts/selected_grasp_pose_scene.xml"

ARM_JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = {
    "gripper0_right_finger_joint1": 0.04,
    "gripper0_right_finger_joint2": -0.04,
}
TCP_SITE = "gripper0_right_grip_site"


def set_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RuntimeError(f"missing joint {name}")
    data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)


def add_sphere(scene: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba: list[float]) -> None:
    if scene.ngeom >= len(scene.geoms):
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_frame(scene: mujoco.MjvScene, T: np.ndarray, length: float = 0.11) -> None:
    colors = [
        [1.0, 0.05, 0.05, 0.95],
        [0.05, 1.0, 0.05, 0.95],
        [0.05, 0.25, 1.0, 0.95],
    ]
    origin = T[:3, 3]
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
            0.005,
            origin,
            origin + length * T[:3, axis],
        )
        scene.ngeom += 1


def camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array([0.58, 0.02, 1.10], dtype=np.float64)
    cam.distance = 1.18
    cam.azimuth = 138.0
    cam.elevation = -18.0
    return cam


def main() -> None:
    metadata = json.loads(DEBUG_JSON.read_text(encoding="utf-8"))
    arrays = np.load(DEBUG_NPZ)
    selected_q = np.asarray(arrays["selected_grasp_joint_values"], dtype=np.float64)
    selected_T = np.asarray(arrays["selected_transformed_grasp_matrix"], dtype=np.float64)

    phase41.TORC_SCENE_XML = PROJECT_ROOT / "original_torc/lab_vbnpm/tests/scenes/final/difficult_116.xml"
    phase41.patch_torc_scene()
    POSE_XML.write_text(phase41.PATCHED_XML.read_text(encoding="utf-8"), encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(POSE_XML))
    data = mujoco.MjData(model)
    for name, value in zip(ARM_JOINTS, selected_q):
        set_qpos(model, data, name, float(value))
    for name, value in FINGER_JOINTS.items():
        set_qpos(model, data, name, value)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    achieved_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64)

    renderer = mujoco.Renderer(model, height=950, width=1400)
    renderer.update_scene(data, camera=camera())
    scene = renderer.scene
    add_frame(scene, selected_T)
    add_sphere(scene, selected_T[:3, 3], 0.018, [0.0, 1.0, 1.0, 1.0])
    add_sphere(scene, achieved_pos, 0.014, [1.0, 1.0, 0.0, 1.0])
    image = renderer.render()
    renderer.close()

    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle((0, 0, 1400, 118), fill=(0, 0, 0, 145))
    draw.text((18, 14), "Franka at selected TORC pipeline grasp pose", fill=(255, 255, 255, 255))
    draw.text((18, 40), f"source: {DEBUG_NPZ}", fill=(220, 240, 255, 255))
    draw.text(
        (18, 66),
        f"object={metadata.get('intended_object_name')} segment={metadata.get('selected_object_segment_id')} "
        f"candidate={metadata.get('selected_index_in_sorted_validated_candidates')} score={metadata.get('selected_score'):.3f}",
        fill=(255, 255, 255, 255),
    )
    draw.text((18, 92), "cyan: selected grasp TCP target; yellow: MuJoCo TCP from selected_grasp_joint_values", fill=(255, 245, 180, 255))
    img.save(OUT_PNG)

    OUT_MANIFEST.write_text(
        json.dumps(
            {
                "output_png": str(OUT_PNG),
                "source_debug_json": str(DEBUG_JSON),
                "source_debug_npz": str(DEBUG_NPZ),
                "source_pipeline": "scripts/phase4/render_franka_planning_full.py with TORC_CAPTURE_SELECTED_GRASP=1",
                "selected_metadata": metadata,
                "selected_grasp_joint_values": selected_q.tolist(),
                "selected_transformed_grasp_matrix": selected_T.tolist(),
                "achieved_tcp_world": achieved_pos.tolist(),
                "tcp_position_error_m": float(np.linalg.norm(achieved_pos - selected_T[:3, 3])),
                "planning_executed_for_pose_selection": bool(metadata.get("planning", {}).get("plan2_grasp_approach_success")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUT_PNG)


if __name__ == "__main__":
    main()
