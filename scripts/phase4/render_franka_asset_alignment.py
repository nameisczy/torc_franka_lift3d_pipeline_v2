#!/usr/bin/env python3
"""Render Phase 4.1 Franka asset alignment inside the original TORC scene.

This is intentionally render-only:
- no IK
- no planner
- no execution interface
- no grasp pose mapping
"""

from __future__ import annotations

import os
from pathlib import Path
import copy
import re
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path("/mnt/ssd/ziyaochen/torc_franka_lift3d_pipeline_v2")
TORC_SCENE_XML = Path(
    os.environ.get(
        "TORC_SCENE_XML",
        str(PROJECT_ROOT / "original_torc/lab_vbnpm/tests/scenes/tabletop_unstructured/scene20.xml"),
    )
)
TORC_LAB_ROOT = PROJECT_ROOT / "original_torc/lab_vbnpm"
FRANKA_COMPILED_XML = PROJECT_ROOT / "assets/franka/mjcf/robosuite/panda_lift_compiled.xml"
FRANKA_MESH_ROOT = PROJECT_ROOT / "assets/franka/meshes"
FRANKA_CUROBO_YAML = PROJECT_ROOT / "assets/franka/config/curobo/franka_panda.yml"
WORK_DIR = PROJECT_ROOT / "phase4_artifacts"
PATCHED_XML = WORK_DIR / "phase4_1_franka_asset_alignment.xml"
OUT_PNG = PROJECT_ROOT / "franka_asset_alignment.png"

TORC_BASE_XYZ = np.array([0.0, 0.0, 0.0], dtype=np.float64)


def load_franka_base_xyz() -> np.ndarray:
    text = FRANKA_CUROBO_YAML.read_text(encoding="utf-8")
    match = re.search(r"base_pose_world_m:\s*\[([^\]]+)\]", text)
    if not match:
        raise RuntimeError(f"missing base_pose_world_m in {FRANKA_CUROBO_YAML}")
    values = [float(item.strip()) for item in match.group(1).split(",")]
    if len(values) != 3:
        raise RuntimeError(f"invalid base_pose_world_m in {FRANKA_CUROBO_YAML}: {values}")
    return np.array(values, dtype=np.float64)


FRANKA_BASE_XYZ = load_franka_base_xyz()
START_QPOS = {
    "robot0_joint1": 0.0,
    "robot0_joint2": -1.25,
    "robot0_joint3": 0.0,
    "robot0_joint4": -2.35,
    "robot0_joint5": 0.0,
    "robot0_joint6": 1.10,
    "robot0_joint7": 0.0,
    "gripper0_right_finger_joint1": 0.04,
    "gripper0_right_finger_joint2": -0.04,
}


def fmt(values: np.ndarray) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def should_copy_franka_asset(node: ET.Element) -> bool:
    name = node.attrib.get("name", "")
    file_name = node.attrib.get("file", "")
    return (
        name.startswith("robot0_")
        or name.startswith("gripper0_right_")
        or name.startswith("fixed_mount0_")
        or "panda_gripper" in file_name
        or "/robots/panda/" in file_name
        or "/bases/meshes/rethink_mount/" in file_name
    )


def copy_franka_assets_and_actuators(root: ET.Element) -> None:
    franka_root = ET.parse(FRANKA_COMPILED_XML).getroot()
    target_asset = root.find("asset")
    source_asset = franka_root.find("asset")
    if target_asset is None or source_asset is None:
        raise RuntimeError("missing asset node while composing TORC + Franka scene")

    existing_names = {node.attrib.get("name") for node in target_asset if node.attrib.get("name")}
    for node in list(source_asset):
        if should_copy_franka_asset(node) and node.attrib.get("name") not in existing_names:
            copied = copy.deepcopy(node)
            file_name = copied.attrib.get("file")
            if file_name:
                if "/robots/panda/meshes/" in file_name:
                    copied.set("file", str(FRANKA_MESH_ROOT / "robosuite_panda_assets/meshes" / Path(file_name).name))
                elif "/robots/panda/obj_meshes/" in file_name:
                    rel = file_name.split("/robots/panda/obj_meshes/", 1)[1]
                    copied.set("file", str(FRANKA_MESH_ROOT / "robosuite_panda_assets/obj_meshes" / rel))
                elif "panda_gripper/hand_vis.stl" in file_name:
                    copied.set("file", str(FRANKA_MESH_ROOT / "panda_gripper/hand_vis.stl"))
                elif "panda_gripper/finger_vis.stl" in file_name:
                    copied.set("file", str(FRANKA_MESH_ROOT / "panda_gripper/finger_vis.stl"))
                elif "panda_gripper/finger_longer.stl" in file_name:
                    copied.set("file", str(FRANKA_MESH_ROOT / "panda_gripper/finger_longer.stl"))
                elif "panda_gripper/hand.stl" in file_name:
                    copied.set("file", str(FRANKA_MESH_ROOT / "panda_gripper/hand.stl"))
                elif "panda_gripper/finger.stl" in file_name:
                    copied.set("file", str(FRANKA_MESH_ROOT / "panda_gripper/finger.stl"))
                elif "/bases/meshes/rethink_mount/pedestal.stl" in file_name:
                    copied.set("file", str(FRANKA_MESH_ROOT / "bases/rethink_mount/pedestal.stl"))
            target_asset.append(copied)
            existing_names.add(node.attrib.get("name"))

    for tag in ["actuator", "sensor"]:
        old = root.find(tag)
        if old is not None:
            root.remove(old)
        source = franka_root.find(tag)
        if source is not None:
            root.append(copy.deepcopy(source))
    replace_arm_torque_actuators_with_position(root)


def replace_arm_torque_actuators_with_position(root: ET.Element) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for child in list(actuator):
        name = child.attrib.get("name", "")
        if name.startswith("robot0_torq_j"):
            actuator.remove(child)
    joint_ranges = {
        "robot0_joint1": "-2.8973 2.8973",
        "robot0_joint2": "-1.7628 1.7628",
        "robot0_joint3": "-2.8973 2.8973",
        "robot0_joint4": "-3.0718 -0.0698",
        "robot0_joint5": "-2.8973 2.8973",
        "robot0_joint6": "-0.0175 3.7525",
        "robot0_joint7": "-2.8973 2.8973",
    }
    force_ranges = {
        "robot0_joint1": "-80 80",
        "robot0_joint2": "-80 80",
        "robot0_joint3": "-80 80",
        "robot0_joint4": "-80 80",
        "robot0_joint5": "-12 12",
        "robot0_joint6": "-12 12",
        "robot0_joint7": "-12 12",
    }
    existing = {child.attrib.get("joint") for child in actuator}
    for idx in range(1, 8):
        joint = f"robot0_joint{idx}"
        if joint in existing:
            continue
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"robot0_pos_j{idx}",
                "joint": joint,
                "kp": "650",
                "ctrllimited": "true",
                "ctrlrange": joint_ranges[joint],
                "forcelimited": "true",
                "forcerange": force_ranges[joint],
            },
        )


def remove_torc_robot(worldbody: ET.Element) -> None:
    for body in list(worldbody):
        if body.attrib.get("name") == "base":
            worldbody.remove(body)
            return
    raise RuntimeError("TORC scene did not contain root robot body named 'base'")


def add_franka_robot_body(worldbody: ET.Element) -> None:
    franka_root = ET.parse(FRANKA_COMPILED_XML).getroot()
    source_worldbody = franka_root.find("worldbody")
    if source_worldbody is None:
        raise RuntimeError("compiled Franka XML has no worldbody")

    robot_base = source_worldbody.find(".//body[@name='robot0_base']")
    if robot_base is None:
        raise RuntimeError("compiled Franka XML has no robot0_base body")
    robot_base = copy.deepcopy(robot_base)
    robot_base.set("pos", fmt(FRANKA_BASE_XYZ))
    worldbody.insert(2, robot_base)


def patch_torc_scene() -> None:
    tree = ET.parse(TORC_SCENE_XML)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(TORC_LAB_ROOT.resolve()))
    compiler.set("texturedir", str(TORC_LAB_ROOT.resolve()))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("TORC scene XML has no worldbody")

    remove_torc_robot(worldbody)
    for tag in ["contact", "equality", "tendon", "keyframe"]:
        node = root.find(tag)
        if node is not None:
            root.remove(node)
    copy_franka_assets_and_actuators(root)
    add_franka_robot_body(worldbody)

    ET.SubElement(
        worldbody,
        "body",
        {"name": "phase4_1_torc_base_origin_marker", "pos": fmt(TORC_BASE_XYZ)},
    )
    marker = worldbody.find(".//body[@name='phase4_1_torc_base_origin_marker']")
    assert marker is not None
    ET.SubElement(
        marker,
        "geom",
        {
            "name": "torc_base_origin_disc",
            "type": "cylinder",
            "pos": "0 0 0.01",
            "size": "0.12 0.01",
            "rgba": "1 0.18 0.05 1",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )
    ET.SubElement(
        marker,
        "geom",
        {
            "name": "phase4_1_franka_z_support",
            "type": "cylinder",
            "pos": fmt(np.array([0.0, 0.0, FRANKA_BASE_XYZ[2] / 2.0])),
            "size": "0.055 0.43",
            "rgba": "0.1 0.1 0.12 0.85",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )
    ET.SubElement(marker, "site", {"name": "torc_base_origin_site", "pos": "0 0 0", "size": "0.025", "rgba": "1 0 0 1"})

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_vis = visual.find("global")
    if global_vis is None:
        global_vis = ET.SubElement(visual, "global")
    global_vis.set("offwidth", "1400")
    global_vis.set("offheight", "950")

    tree.write(PATCHED_XML, encoding="utf-8", xml_declaration=True)


def set_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for joint, value in START_QPOS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid < 0:
            raise RuntimeError(f"missing joint: {joint}")
        adr = int(model.jnt_qposadr[jid])
        data.qpos[adr] = float(value)
    mujoco.mj_forward(model, data)


def free_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.42, 0.0, 1.08], dtype=np.float64)
    camera.distance = 2.15
    camera.azimuth = 142.0
    camera.elevation = -18.0
    return camera


def render() -> None:
    model = mujoco.MjModel.from_xml_path(str(PATCHED_XML))
    data = mujoco.MjData(model)
    set_qpos(model, data)

    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.endswith("_collision") or "collision" in name:
            model.geom_rgba[gid, 3] = max(float(model.geom_rgba[gid, 3]), 0.35)

    renderer = mujoco.Renderer(model, height=950, width=1400)
    renderer.update_scene(data, camera=free_camera())
    image = renderer.render()
    renderer.close()

    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle((0, 0, 1400, 102), fill=(0, 0, 0, 128))
    draw.text((18, 14), "PHASE 4.1 - Franka asset replacement inside TORC scene", fill=(255, 255, 255, 255))
    scene_rel = TORC_SCENE_XML.relative_to(TORC_LAB_ROOT)
    draw.text((18, 40), f"Source scene: {scene_rel}; old TORC robot body removed", fill=(255, 255, 255, 255))
    draw.text((18, 66), f"Franka robot0_base: original TORC base xy [0, 0] with contract z {FRANKA_BASE_XYZ[2]:.3f}; no IK/planner/execution changes", fill=(255, 255, 255, 255))
    img.save(OUT_PNG)


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    patch_torc_scene()
    render()
    print(OUT_PNG)


if __name__ == "__main__":
    main()
