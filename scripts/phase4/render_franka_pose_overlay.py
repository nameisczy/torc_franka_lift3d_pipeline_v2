#!/usr/bin/env python3
"""Phase 4.2 pose mapping validation inside the original TORC scene.

No planning and no collision feasibility are evaluated here. The script:
1. Builds the Phase 4.1 TORC scene with Franka assets.
2. Loads current-project CGN lowlevel contact/axis candidates.
3. Converts lowlevel -> CanonicalGrasp -> Franka RobotAdapter TCP target.
4. Runs MuJoCo Jacobian IK without collision constraints for visualization.
5. Renders one overlay image.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import render_franka_asset_alignment as phase41


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TORC_SCRIPTS = PROJECT_ROOT / "original_torc/lab_vbnpm/scripts"
if str(TORC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TORC_SCRIPTS))
TORC_SCENE_XML = Path(
    os.environ.get(
        "TORC_SCENE_XML",
        str(PROJECT_ROOT / "original_torc/lab_vbnpm/tests/scenes/tabletop_unstructured/scene20.xml"),
    )
)
phase41.TORC_SCENE_XML = TORC_SCENE_XML
WORK_DIR = PROJECT_ROOT / "phase4_artifacts"
POSE_XML = WORK_DIR / "phase4_2_franka_pose_overlay.xml"
OUT_PNG = PROJECT_ROOT / "franka_pose_overlay.png"
TARGET_OBJECT = os.environ.get("TORC_TARGET_OBJECT", "obj_000044_0")
TARGET_OBJECT_KEY = "_".join(TARGET_OBJECT.split("_")[:2])
TARGET_GRASP_INDEX = 0

ARM_JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["gripper0_right_finger_joint1", "gripper0_right_finger_joint2"]
TCP_SITE = "gripper0_right_grip_site"
START_QPOS = np.array([0.0, -1.25, 0.0, -2.35, 0.0, 1.10, 0.0], dtype=np.float64)

from grasp_representation.canonical_filters import filter_canonical_grasps
from grasp_representation.lowlevel_grasp import lowlevel_grasps_from_arrays, lowlevel_to_canonical_grasp
from perception.scene_capture import capture_scene_from_mujoco_xml
from robot_interface.franka_adapter import RobotAdapter


def load_gripper_open_contract() -> dict[str, float]:
    text = phase41.FRANKA_CUROBO_YAML.read_text(encoding="utf-8")
    block_match = re.search(r"open:\n((?:\s{8}gripper0_right_finger_joint[12]:\s*[-0-9.]+\n)+)", text)
    if not block_match:
        raise RuntimeError(f"missing robosuite gripper open contract in {phase41.FRANKA_CUROBO_YAML}")
    values: dict[str, float] = {}
    for name, value in re.findall(r"(gripper0_right_finger_joint[12]):\s*([-0-9.]+)", block_match.group(1)):
        values[name] = float(value)
    if set(values) != set(FINGER_JOINTS):
        raise RuntimeError(f"incomplete gripper open contract: {values}")
    return values


FINGER_OPEN = load_gripper_open_contract()


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    m = np.asarray(rot, dtype=np.float64)
    tr = float(np.trace(m))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    idx = int(np.argmax(np.diag(m)))
    if idx == 0:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif idx == 1:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def transform_from_xml_body(body: ET.Element) -> np.ndarray:
    pos = np.fromstring(body.attrib.get("pos", "0 0 0"), sep=" ", dtype=np.float64)
    quat = np.fromstring(body.attrib.get("quat", "1 0 0 0"), sep=" ", dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_matrix(quat)
    T[:3, 3] = pos
    return T


def make_franka_frame(approach_direction: np.ndarray, closing_direction: np.ndarray) -> np.ndarray:
    z_axis = np.asarray(approach_direction, dtype=np.float64)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_seed = np.asarray(closing_direction, dtype=np.float64)
    y_seed = y_seed - z_axis * float(np.dot(y_seed, z_axis))
    if np.linalg.norm(y_seed) < 1e-6:
        y_seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y_axis = y_seed / np.linalg.norm(y_seed)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def lowlevel_cache_path(body_name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in body_name)
    return WORK_DIR / "cgn_lowlevel" / TORC_SCENE_XML.stem / f"{safe}.raw_output.npz"


def load_lowlevel_candidates_for_body(body_name: str, object_id: int):
    path = lowlevel_cache_path(body_name)
    if not path.exists():
        raise RuntimeError(
            f"missing CGN lowlevel cache for {body_name}: {path}. "
            "Run phase4.3 lowlevel CGN generation first; build_6d_grasp poses are forbidden for Franka."
        )
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    grasps = lowlevel_grasps_from_arrays(arrays, object_id=int(object_id))
    grasps.sort(key=lambda grasp: (-float(grasp.score), int(grasp.selection_index), int(grasp.source_index)))
    return grasps


def select_overlay_object(scene):
    preferred = next((obj for obj in scene.objects if obj.name == TARGET_OBJECT), None)
    ordered = ([preferred] if preferred is not None else []) + [
        obj for obj in scene.objects if preferred is None or obj.name != preferred.name
    ]
    errors = []
    for obj in ordered:
        try:
            candidates = load_lowlevel_candidates_for_body(str(obj.name), int(obj.object_id))
        except Exception as exc:
            errors.append({"object": str(obj.name), "error": repr(exc)})
            continue
        if candidates:
            return obj, candidates, errors
    raise RuntimeError(f"no usable CGN lowlevel candidates in current-project cache; errors={errors[:8]}")


def load_target_pose() -> tuple[np.ndarray, None, np.ndarray, dict[str, object]]:
    root = ET.parse(TORC_SCENE_XML).getroot()
    scene = capture_scene_from_mujoco_xml(TORC_SCENE_XML)
    selected_obj, lowlevel_candidates, selection_errors = select_overlay_object(scene)
    obj = root.find(f".//body[@name='{selected_obj.name}']")
    if obj is None:
        raise RuntimeError(f"missing target body {selected_obj.name}")
    T_world_obj = transform_from_xml_body(obj)

    adapter = RobotAdapter()
    chosen = None
    for lowlevel in lowlevel_candidates:
        canonical = lowlevel_to_canonical_grasp(lowlevel)
        filtered = list(filter_canonical_grasps([canonical], min_score=0.0))
        if filtered:
            chosen = (lowlevel, filtered[0], adapter.adapt(filtered[0]))
            break
    if chosen is None:
        raise RuntimeError(f"no canonical-valid lowlevel candidates for {selected_obj.name}")
    lowlevel, canonical, command = chosen
    metadata = {
        "selected_object": str(selected_obj.name),
        "selected_object_id": int(selected_obj.object_id),
        "requested_target_object": TARGET_OBJECT,
        "fallback_used": str(selected_obj.name) != TARGET_OBJECT,
        "candidate_source": f"phase4_artifacts/cgn_lowlevel/{TORC_SCENE_XML.stem}/*.raw_output.npz",
        "forbidden_source": "gc6d_grasp_poses.npy/build_6d_grasp pose",
        "lowlevel_source_index": int(lowlevel.source_index),
        "lowlevel_selection_index": int(lowlevel.selection_index),
        "lowlevel_score": float(lowlevel.score),
        "selection_errors": selection_errors[:8],
        "canonical_opening_width_m": float(canonical.opening_width),
    }
    return command.tcp_contact_pose_world, None, T_world_obj, metadata


def set_initial_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for name, value in zip(ARM_JOINTS, START_QPOS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[jid])] = float(value)
    for name, value in FINGER_OPEN.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[jid])] = float(value)
    mujoco.mj_forward(model, data)


def orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.cross(current[:, 0], target[:, 0])
        + np.cross(current[:, 1], target[:, 1])
        + np.cross(current[:, 2], target[:, 2])
    )


def solve_visual_ik(model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray) -> dict[str, float]:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
    dof_ids = np.array([int(model.jnt_dofadr[jid]) for jid in joint_ids], dtype=np.int32)
    qpos_ids = np.array([int(model.jnt_qposadr[jid]) for jid in joint_ids], dtype=np.int32)
    lo = np.array([model.jnt_range[jid, 0] for jid in joint_ids], dtype=np.float64)
    hi = np.array([model.jnt_range[jid, 1] for jid in joint_ids], dtype=np.float64)

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    best = {"pos_err": float("inf"), "rot_err": float("inf")}

    for _ in range(650):
        mujoco.mj_forward(model, data)
        pos_cur = np.asarray(data.site_xpos[site_id], dtype=np.float64)
        rot_cur = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        e_pos = target[:3, 3] - pos_cur
        e_rot = orientation_error(rot_cur, target[:3, :3])
        err = np.concatenate([1.0 * e_pos, 0.18 * e_rot])
        pos_norm = float(np.linalg.norm(e_pos))
        rot_norm = float(np.linalg.norm(e_rot))
        if pos_norm < best["pos_err"]:
            best = {"pos_err": pos_norm, "rot_err": rot_norm}
        if pos_norm < 0.01 and rot_norm < 0.25:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, dof_ids], 0.18 * jacr[:, dof_ids]])
        damping = 0.035
        dq = J.T @ np.linalg.solve(J @ J.T + damping * damping * np.eye(6), err)
        dq = np.clip(dq, -0.045, 0.045)
        data.qpos[qpos_ids] = np.clip(data.qpos[qpos_ids] + dq, lo, hi)

    mujoco.mj_forward(model, data)
    pos_cur = np.asarray(data.site_xpos[site_id], dtype=np.float64)
    rot_cur = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    return {
        "final_pos_err_m": float(np.linalg.norm(target[:3, 3] - pos_cur)),
        "final_rot_err_norm": float(np.linalg.norm(orientation_error(rot_cur, target[:3, :3]))),
        "best_pos_err_m": float(best["pos_err"]),
        "best_rot_err_norm": float(best["rot_err"]),
    }


def add_frame(scene: mujoco.MjvScene, T: np.ndarray, length: float, width: float, alpha: float) -> None:
    colors = [
        [1.0, 0.05, 0.05, alpha],
        [0.05, 1.0, 0.05, alpha],
        [0.05, 0.25, 1.0, alpha],
    ]
    origin = T[:3, 3]
    for axis in range(3):
        if scene.ngeom >= len(scene.geoms):
            return
        geom = scene.geoms[scene.ngeom]
        end = origin + length * T[:3, axis]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(colors[axis], dtype=np.float32),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CYLINDER, width, origin, end)
        scene.ngeom += 1


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


def free_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.54, -0.04, 1.08], dtype=np.float64)
    camera.distance = 1.35
    camera.azimuth = 138.0
    camera.elevation = -16.0
    return camera


def render_overlay() -> None:
    phase41.patch_torc_scene()
    POSE_XML.write_text(phase41.PATCHED_XML.read_text(encoding="utf-8"), encoding="utf-8")

    canonical_target, old_torc_target, T_world_obj, target_metadata = load_target_pose()
    model = mujoco.MjModel.from_xml_path(str(POSE_XML))
    data = mujoco.MjData(model)
    set_initial_qpos(model, data)
    ik_metrics = solve_visual_ik(model, data, canonical_target)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    achieved = np.eye(4, dtype=np.float64)
    achieved[:3, :3] = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    achieved[:3, 3] = np.asarray(data.site_xpos[site_id], dtype=np.float64)

    renderer = mujoco.Renderer(model, height=950, width=1400)
    renderer.update_scene(data, camera=free_camera())
    scene = renderer.scene
    add_frame(scene, canonical_target, 0.11, 0.006, 1.0)
    add_sphere(scene, canonical_target[:3, 3], 0.018, [0.0, 1.0, 1.0, 1.0])
    add_frame(scene, achieved, 0.09, 0.004, 0.70)
    add_sphere(scene, achieved[:3, 3], 0.012, [1.0, 1.0, 0.0, 1.0])
    image = renderer.render()
    renderer.close()

    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle((0, 0, 1400, 132), fill=(0, 0, 0, 145))
    draw.text((18, 14), "PHASE 4.2 - Franka pose mapping validation in TORC scene", fill=(255, 255, 255, 255))
    draw.text((18, 40), "cyan axes/sphere: CGN lowlevel -> CanonicalGrasp -> FrankaAdapter TCP target", fill=(210, 255, 255, 255))
    draw.text((18, 66), "yellow sphere: Franka TCP after no-collision MuJoCo IK; build_6d_grasp pose input is forbidden", fill=(255, 245, 180, 255))
    draw.text((18, 92), f"IK visual error: position {ik_metrics['final_pos_err_m']:.3f} m, rotation norm {ik_metrics['final_rot_err_norm']:.3f}", fill=(255, 255, 255, 255))
    img.save(OUT_PNG)

    manifest = {
        "phase": "4.2",
        "scene": str(TORC_SCENE_XML),
        "target_object": target_metadata["selected_object"],
        "requested_target_object": TARGET_OBJECT,
        "target_metadata": target_metadata,
        "target_grasp_index": TARGET_GRASP_INDEX,
        "franka_base_world_m": phase41.FRANKA_BASE_XYZ.tolist(),
        "tcp_site": TCP_SITE,
        "canonical_target_matrix": canonical_target.tolist(),
        "old_torc_shifted_target_matrix": None,
        "achieved_tcp_matrix": achieved.tolist(),
        "ik_metrics": ik_metrics,
        "notes": [
            "No CuRobo planning was run.",
            "Collision is ignored; penetration is allowed for this pose-space check.",
            "TORC-specific EE link semantics are replaced by a canonical Franka TCP adapter at gripper0_right_grip_site.",
            "The old TORC/build_6d shifted pose is not loaded or rendered; lowlevel CGN contact/axis fields are the only grasp source.",
        ],
    }
    (WORK_DIR / "phase4_2_pose_overlay_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(OUT_PNG)


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    render_overlay()


if __name__ == "__main__":
    main()
