#!/usr/bin/env python3
"""Render the selected grasp with the real Franka/MuJoCo and CuRobo geometry.

This is a diagnostic visualizer only. It does not select, filter, or modify a
grasp. It overlays the selected TORC pipeline grasp, target TSDF points,
MuJoCo finger pad boxes, and the CuRobo finger collision spheres in one scene.
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
from output_paths import artifact_path, result_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXP_DIR = (
    PROJECT_ROOT
    / "phase4_artifacts/torc_franka_pipeline_1783338153"
    / "exp_2026-07-06_15-42-46__difficult_116_obj_000070_0_dg_only"
)
EXP_DIR = Path(os.environ.get("TORC_SELECTED_GRASP_EXP_DIR", str(DEFAULT_EXP_DIR)))
POINTS_TXT = EXP_DIR / "output_target_points.txt"
DEBUG_DIR = EXP_DIR / "selected_grasp_debug"
DEBUG_JSON = DEBUG_DIR / "pick_01_selected_grasp_debug.json"
DEBUG_NPZ = DEBUG_DIR / "pick_01_selected_grasp_debug.npz"
SCENE_XML = PROJECT_ROOT / "original_torc/lab_vbnpm/tests/scenes/final/difficult_116.xml"
OUT_XML = artifact_path("franka_grasp_geometry_contract_scene.xml")
OUT_PNG = result_path("franka_grasp_geometry_contract.png")
OUT_MANIFEST = artifact_path("franka_grasp_geometry_contract_manifest.json")

ARM_JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]
FINGER_JOINTS_OPEN = {
    "gripper0_right_finger_joint1": 0.04,
    "gripper0_right_finger_joint2": -0.04,
}
TCP_SITE = "gripper0_right_grip_site"

PAD_GEOMS = [
    "gripper0_right_finger1_pad_collision",
    "gripper0_right_finger2_pad_collision",
]
CUROBO_FINGER_SPHERES = {
    "gripper0_right_leftfinger": [
        ([-0.004, 0.0035, 0.035], 0.0065),
        ([0.004, 0.0035, 0.035], 0.0065),
        ([-0.004, 0.0035, 0.047], 0.0065),
        ([0.004, 0.0035, 0.047], 0.0065),
        ([0.0, 0.012, 0.021], 0.010),
        ([0.0, 0.018, 0.012], 0.010),
    ],
    "gripper0_right_rightfinger": [
        ([-0.004, -0.0035, 0.035], 0.0065),
        ([0.004, -0.0035, 0.035], 0.0065),
        ([-0.004, -0.0035, 0.047], 0.0065),
        ([0.004, -0.0035, 0.047], 0.0065),
        ([0.0, -0.012, 0.021], 0.010),
        ([0.0, -0.018, 0.012], 0.010),
    ],
}


def set_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RuntimeError(f"missing joint {name}")
    data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)


def add_sphere(scene: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba: list[float]) -> bool:
    if scene.ngeom >= len(scene.geoms):
        return False
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
    return True


def add_box(
    scene: mujoco.MjvScene,
    pos: np.ndarray,
    mat: np.ndarray,
    half_size: np.ndarray,
    rgba: list[float],
) -> bool:
    if scene.ngeom >= len(scene.geoms):
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        np.asarray(half_size, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.asarray(mat, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
    return True


def add_frame(scene: mujoco.MjvScene, transform: np.ndarray, length: float = 0.10) -> None:
    colors = [
        [1.0, 0.05, 0.05, 0.95],
        [0.05, 1.0, 0.05, 0.95],
        [0.05, 0.25, 1.0, 0.95],
    ]
    origin = transform[:3, 3]
    for axis, color in enumerate(colors):
        if scene.ngeom >= len(scene.geoms):
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(color, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            0.004,
            origin,
            origin + length * transform[:3, axis],
        )
        scene.ngeom += 1


def camera(lookat: np.ndarray, mode: str) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    if mode == "top":
        cam.distance = 0.34
        cam.azimuth = 90.0
        cam.elevation = -89.0
    else:
        cam.distance = 0.46
        cam.azimuth = 138.0
        cam.elevation = -16.0
    return cam


def fade_robot_geoms(model: mujoco.MjModel) -> None:
    """Make the arm visually unobtrusive while preserving object rendering."""
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[geom_id]),
        ) or ""
        if geom_name.startswith("robot0_") or geom_name.startswith("gripper0_") or body_name.startswith("robot0_") or body_name.startswith("gripper0_"):
            model.geom_rgba[geom_id, 3] = 0.08


def curobo_spheres_world(model: mujoco.MjModel, data: mujoco.MjData) -> list[tuple[np.ndarray, float]]:
    spheres: list[tuple[np.ndarray, float]] = []
    for body_name, local_spheres in CUROBO_FINGER_SPHERES.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise RuntimeError(f"missing body {body_name}")
        pos = np.asarray(data.xpos[body_id], dtype=np.float64)
        rot = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        for local_center, radius in local_spheres:
            spheres.append((pos + rot @ np.asarray(local_center, dtype=np.float64), float(radius)))
    return spheres


def point_box_distances(points: np.ndarray, pos: np.ndarray, mat: np.ndarray, half_size: np.ndarray) -> np.ndarray:
    rot = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    local = (points - pos) @ rot
    delta = np.abs(local) - half_size
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
    inside = np.minimum(np.maximum.reduce(delta, axis=1), 0.0)
    return outside + inside


def render_view(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    points: np.ndarray,
    selected_t: np.ndarray,
    achieved_tcp: np.ndarray,
    spheres: list[tuple[np.ndarray, float]],
    mode: str,
) -> Image.Image:
    lookat = selected_t[:3, 3].copy()
    renderer = mujoco.Renderer(model, height=860, width=1180)
    renderer.update_scene(data, camera=camera(lookat, mode))
    scene = renderer.scene

    rng = np.random.default_rng(11)
    max_points = min(len(points), 5000)
    sample = points[rng.choice(len(points), max_points, replace=False)] if len(points) > max_points else points
    for point in sample:
        add_sphere(scene, point, 0.0020, [0.0, 0.95, 1.0, 0.62])

    for geom_name in PAD_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        add_box(
            scene,
            np.asarray(data.geom_xpos[gid], dtype=np.float64),
            np.asarray(data.geom_xmat[gid], dtype=np.float64),
            np.asarray(model.geom_size[gid], dtype=np.float64),
            [0.05, 1.0, 0.15, 0.45],
        )

    for center, radius in spheres:
        add_sphere(scene, center, radius, [0.85, 0.15, 1.0, 0.42])

    add_frame(scene, selected_t)
    add_sphere(scene, selected_t[:3, 3], 0.015, [1.0, 0.05, 0.05, 1.0])
    add_sphere(scene, achieved_tcp, 0.012, [1.0, 1.0, 0.0, 1.0])

    image = Image.fromarray(renderer.render()).convert("RGB")
    renderer.close()
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, 1180, 110), fill=(0, 0, 0, 150))
    draw.text((16, 12), f"{mode} view: selected grasp geometry contract", fill=(255, 255, 255, 255))
    draw.text((16, 38), "cyan: TORC target TSDF points; green boxes: MuJoCo finger pads", fill=(220, 255, 235, 255))
    draw.text((16, 64), "purple: CuRobo finger spheres; red: selected TCP target; yellow: achieved MuJoCo TCP", fill=(255, 235, 255, 255))
    draw.text((16, 88), f"source: {DEBUG_NPZ}", fill=(220, 240, 255, 255))
    return image


def main() -> None:
    metadata = json.loads(DEBUG_JSON.read_text(encoding="utf-8"))
    arrays = np.load(DEBUG_NPZ)
    points = np.loadtxt(POINTS_TXT, dtype=np.float64).reshape(-1, 3)
    selected_q = np.asarray(arrays["selected_grasp_joint_values"], dtype=np.float64)
    selected_t = np.asarray(arrays["selected_transformed_grasp_matrix"], dtype=np.float64)

    phase41.TORC_SCENE_XML = SCENE_XML
    phase41.patch_torc_scene()
    OUT_XML.parent.mkdir(parents=True, exist_ok=True)
    OUT_XML.write_text(phase41.PATCHED_XML.read_text(encoding="utf-8"), encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(OUT_XML))
    fade_robot_geoms(model)
    data = mujoco.MjData(model)
    for name, value in zip(ARM_JOINTS, selected_q):
        set_qpos(model, data, name, value)
    for name, value in FINGER_JOINTS_OPEN.items():
        set_qpos(model, data, name, value)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    achieved_tcp = np.asarray(data.site_xpos[site_id], dtype=np.float64)
    spheres = curobo_spheres_world(model, data)

    pad_stats = {}
    for geom_name in PAD_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        distances = point_box_distances(
            points,
            np.asarray(data.geom_xpos[gid], dtype=np.float64),
            np.asarray(data.geom_xmat[gid], dtype=np.float64),
            np.asarray(model.geom_size[gid], dtype=np.float64),
        )
        pad_stats[geom_name] = {
            "min_signed_distance_m": float(np.min(distances)),
            "num_tsdf_points_inside_pad_box": int(np.sum(distances < 0.0)),
            "num_tsdf_points_within_2mm": int(np.sum(distances < 0.002)),
            "pad_world_center": np.asarray(data.geom_xpos[gid], dtype=np.float64).tolist(),
            "pad_half_size_m": np.asarray(model.geom_size[gid], dtype=np.float64).tolist(),
        }

    sphere_distances = []
    for center, radius in spheres:
        sphere_distances.append(np.linalg.norm(points - center, axis=1) - radius)
    sphere_distances_arr = np.concatenate(sphere_distances) if sphere_distances else np.array([np.inf])

    side = render_view(model, data, points, selected_t, achieved_tcp, spheres, "side")
    top = render_view(model, data, points, selected_t, achieved_tcp, spheres, "top")
    canvas = Image.new("RGB", (2360, 860), (255, 255, 255))
    canvas.paste(side, (0, 0))
    canvas.paste(top, (1180, 0))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PNG)

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(
        json.dumps(
            {
                "output_png": str(OUT_PNG),
                "source_debug_npz": str(DEBUG_NPZ),
                "source_debug_json": str(DEBUG_JSON),
                "points_txt": str(POINTS_TXT),
                "selected_metadata": metadata,
                "target_point_count": int(len(points)),
                "target_point_min_world": points.min(axis=0).tolist(),
                "target_point_max_world": points.max(axis=0).tolist(),
                "target_point_centroid_world": points.mean(axis=0).tolist(),
                "selected_tcp_target_world": selected_t[:3, 3].tolist(),
                "achieved_tcp_world": achieved_tcp.tolist(),
                "tcp_position_error_m": float(np.linalg.norm(achieved_tcp - selected_t[:3, 3])),
                "mujoco_pad_box_vs_target_tsdf": pad_stats,
                "curobo_sphere_vs_target_tsdf": {
                    "min_signed_distance_m": float(np.min(sphere_distances_arr)),
                    "num_tsdf_points_inside_any_sphere": int(np.sum(sphere_distances_arr < 0.0)),
                    "num_tsdf_points_within_2mm_any_sphere": int(np.sum(sphere_distances_arr < 0.002)),
                    "sphere_count": len(spheres),
                },
                "notes": [
                    "This script only visualizes existing TORC pipeline outputs.",
                    "Green pad boxes are MuJoCo collision geometry at open gripper qpos.",
                    "Purple spheres are current CuRobo finger collision spheres transformed by MuJoCo FK.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUT_PNG)


if __name__ == "__main__":
    main()
