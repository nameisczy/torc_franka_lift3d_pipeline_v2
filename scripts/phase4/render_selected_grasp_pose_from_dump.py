#!/usr/bin/env python3
"""Render Franka at the selected grasp pose dumped by the TORC pipeline.

This does not select, filter, or modify a grasp. It only visualizes the exact
Franka joint target and grasp matrix written by curobo_open_loop.py during the
same TORC pipeline path used for Phase 4.3.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from output_paths import artifact_path, result_path
import render_franka_asset_alignment as phase41


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = (
    PROJECT_ROOT
    / "phase4_artifacts/torc_franka_pipeline_1783268501"
    / "exp_2026-07-05_20-21-53__difficult_116_obj_000070_0_dg_only"
)
EXP_DIR = Path(os.environ.get("TORC_SELECTED_GRASP_EXP_DIR", str(DEFAULT_RUN)))
DEBUG_DIR = EXP_DIR / "selected_grasp_debug"
OUT_PNG = result_path("franka_selected_grasp_pose.png")
OUT_MANIFEST = artifact_path("selected_grasp_pose_render_manifest.json")
POSE_XML = artifact_path("selected_grasp_pose_scene.xml")

ARM_JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = {
    "gripper0_right_finger_joint1": 0.04,
    "gripper0_right_finger_joint2": -0.04,
}
TCP_SITE = "gripper0_right_grip_site"
PICK_TAG_RE = re.compile(r"pick_(\d+)")
RENDER_WIDTH = 1280
RENDER_HEIGHT = 720


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


def initial_scene_xml() -> Path:
    phase41.TORC_SCENE_XML = PROJECT_ROOT / "original_torc/lab_vbnpm/tests/scenes/final/difficult_116.xml"
    phase41.patch_torc_scene()
    POSE_XML.parent.mkdir(parents=True, exist_ok=True)
    POSE_XML.write_text(phase41.PATCHED_XML.read_text(encoding="utf-8"), encoding="utf-8")
    return POSE_XML


def state_for_pick(debug_json: Path) -> tuple[Path, np.ndarray | None, str]:
    match = PICK_TAG_RE.search(debug_json.name)
    pick_index = int(match.group(1)) if match else 1
    if pick_index <= 1:
        return initial_scene_xml(), None, "initial_scene"

    state_file = EXP_DIR / f"state_{pick_index - 1}.pkl"
    if not state_file.exists():
        return initial_scene_xml(), None, f"missing_previous_state:{state_file}"

    with state_file.open("rb") as file:
        state_data = pickle.load(file)
    scene_xml = Path(state_data.get("scene_xml", ""))
    if not scene_xml.exists():
        return initial_scene_xml(), None, f"missing_state_scene_xml:{scene_xml}"
    raw_state = np.asarray(state_data["mujoco_state"], dtype=np.float64)
    return scene_xml, raw_state, str(state_file)


def render_pick(debug_json: Path, debug_npz: Path, out_png: Path) -> dict:
    metadata = json.loads(debug_json.read_text(encoding="utf-8"))
    arrays = np.load(debug_npz)
    selected_q = np.asarray(arrays["selected_grasp_joint_values"], dtype=np.float64)
    selected_T = np.asarray(arrays["selected_transformed_grasp_matrix"], dtype=np.float64)

    scene_xml, raw_state, state_source = state_for_pick(debug_json)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    if raw_state is not None:
        mujoco.mj_setState(model, data, raw_state, mujoco.mjtState.mjSTATE_INTEGRATION)
    for name, value in zip(ARM_JOINTS, selected_q):
        set_qpos(model, data, name, float(value))
    for name, value in FINGER_JOINTS.items():
        set_qpos(model, data, name, value)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    achieved_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64)

    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
    renderer.update_scene(data, camera=camera())
    scene = renderer.scene
    add_frame(scene, selected_T)
    add_sphere(scene, selected_T[:3, 3], 0.018, [0.0, 1.0, 1.0, 1.0])
    add_sphere(scene, achieved_pos, 0.014, [1.0, 1.0, 0.0, 1.0])
    image = renderer.render()
    renderer.close()

    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle((0, 0, RENDER_WIDTH, 118), fill=(0, 0, 0, 145))
    draw.text((18, 14), "Franka at selected TORC pipeline grasp pose", fill=(255, 255, 255, 255))
    draw.text((18, 40), f"source: {debug_npz}", fill=(220, 240, 255, 255))
    draw.text(
        (18, 66),
        f"object={metadata.get('intended_object_name')} segment={metadata.get('selected_object_segment_id')} "
        f"candidate={metadata.get('selected_index_in_sorted_validated_candidates')} score={metadata.get('selected_score'):.3f}",
        fill=(255, 255, 255, 255),
    )
    draw.text((18, 92), f"scene state: {state_source}; cyan target, yellow achieved TCP", fill=(255, 245, 180, 255))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return {
        "output_png": str(out_png),
        "source_debug_json": str(debug_json),
        "source_debug_npz": str(debug_npz),
        "scene_xml": str(scene_xml),
        "scene_state_source": state_source,
        "selected_metadata": metadata,
        "selected_grasp_joint_values": selected_q.tolist(),
        "selected_transformed_grasp_matrix": selected_T.tolist(),
        "achieved_tcp_world": achieved_pos.tolist(),
        "tcp_position_error_m": float(np.linalg.norm(achieved_pos - selected_T[:3, 3])),
        "planning_executed_for_pose_selection": bool(metadata.get("planning", {}).get("plan2_grasp_approach_success")),
    }


def combine_images(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    if len(images) == 1:
        output.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(output)
        return
    width = max(img.width for img in images)
    height = sum(img.height for img in images)
    canvas = Image.new("RGB", (width, height), (12, 12, 12))
    y = 0
    for img in images:
        canvas.paste(img, (0, y))
        y += img.height
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    pick_jsons = sorted(DEBUG_DIR.glob("pick_*_selected_grasp_debug.json"))
    if not pick_jsons:
        raise RuntimeError(f"no selected grasp debug JSON files found in {DEBUG_DIR}")

    rendered = []
    output_paths = []
    for debug_json in pick_jsons:
        debug_npz = debug_json.with_suffix(".npz")
        if not debug_npz.exists():
            continue
        pick_tag = debug_json.name.replace("_selected_grasp_debug.json", "")
        out_png = result_path(f"franka_selected_grasp_pose_{pick_tag}.png")
        rendered.append(render_pick(debug_json, debug_npz, out_png))
        output_paths.append(out_png)

    if not output_paths:
        raise RuntimeError(f"no selected grasp debug NPZ files found in {DEBUG_DIR}")
    combine_images(output_paths, OUT_PNG)

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(
        json.dumps(
            {
                "output_png": str(OUT_PNG),
                "individual_outputs": [str(path) for path in output_paths],
                "pick_count": len(output_paths),
                "source_pipeline": "scripts/phase4/render_franka_planning_full.py with TORC_CAPTURE_SELECTED_GRASP=1",
                "rendered_picks": rendered,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUT_PNG)


if __name__ == "__main__":
    main()
