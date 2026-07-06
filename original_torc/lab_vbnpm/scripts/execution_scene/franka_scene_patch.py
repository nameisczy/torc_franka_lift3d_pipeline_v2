from __future__ import annotations

import copy
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TORC_LAB_ROOT = PROJECT_ROOT / "original_torc/lab_vbnpm"
FRANKA_COMPILED_XML = PROJECT_ROOT / "assets/franka/mjcf/robosuite/panda_lift_compiled.xml"
FRANKA_MESH_ROOT = PROJECT_ROOT / "assets/franka/meshes"
FRANKA_CUROBO_YAML = PROJECT_ROOT / "assets/franka/config/curobo/franka_panda.yml"
PATCH_DIR = PROJECT_ROOT / "phase4_artifacts/franka_runtime_scenes"


def _fmt(values: np.ndarray) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def _load_franka_base_xyz() -> np.ndarray:
    text = FRANKA_CUROBO_YAML.read_text(encoding="utf-8")
    match = re.search(r"base_pose_world_m:\s*\[([^\]]+)\]", text)
    if not match:
        raise RuntimeError(f"missing base_pose_world_m in {FRANKA_CUROBO_YAML}")
    values = [float(item.strip()) for item in match.group(1).split(",")]
    if len(values) != 3:
        raise RuntimeError(f"invalid base_pose_world_m: {values}")
    return np.asarray(values, dtype=np.float64)


def _should_copy_franka_asset(node: ET.Element) -> bool:
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


def _rewrite_franka_mesh_path(node: ET.Element) -> None:
    file_name = node.attrib.get("file")
    if not file_name:
        return
    if "/robots/panda/meshes/" in file_name:
        node.set("file", str(FRANKA_MESH_ROOT / "robosuite_panda_assets/meshes" / Path(file_name).name))
    elif "/robots/panda/obj_meshes/" in file_name:
        rel = file_name.split("/robots/panda/obj_meshes/", 1)[1]
        node.set("file", str(FRANKA_MESH_ROOT / "robosuite_panda_assets/obj_meshes" / rel))
    elif "panda_gripper/hand_vis.stl" in file_name:
        node.set("file", str(FRANKA_MESH_ROOT / "panda_gripper/hand_vis.stl"))
    elif "panda_gripper/finger_vis.stl" in file_name:
        node.set("file", str(FRANKA_MESH_ROOT / "panda_gripper/finger_vis.stl"))
    elif "panda_gripper/finger_longer.stl" in file_name:
        node.set("file", str(FRANKA_MESH_ROOT / "panda_gripper/finger_longer.stl"))
    elif "panda_gripper/hand.stl" in file_name:
        node.set("file", str(FRANKA_MESH_ROOT / "panda_gripper/hand.stl"))
    elif "panda_gripper/finger.stl" in file_name:
        node.set("file", str(FRANKA_MESH_ROOT / "panda_gripper/finger.stl"))
    elif "/bases/meshes/rethink_mount/pedestal.stl" in file_name:
        node.set("file", str(FRANKA_MESH_ROOT / "bases/rethink_mount/pedestal.stl"))


def _replace_arm_torque_actuators_with_position(root: ET.Element) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for child in list(actuator):
        if child.attrib.get("name", "").startswith("robot0_torq_j"):
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
                "kp": os.environ.get("TORC_FRANKA_ARM_POSITION_KP", "650"),
                "ctrllimited": "true",
                "ctrlrange": joint_ranges[joint],
                "forcelimited": "true",
                "forcerange": force_ranges[joint],
            },
        )


def _copy_franka_assets_and_actuators(root: ET.Element) -> None:
    franka_root = ET.parse(FRANKA_COMPILED_XML).getroot()
    target_asset = root.find("asset")
    source_asset = franka_root.find("asset")
    if target_asset is None or source_asset is None:
        raise RuntimeError("missing asset node while composing TORC + Franka scene")
    existing_names = {node.attrib.get("name") for node in target_asset if node.attrib.get("name")}
    for node in list(source_asset):
        if _should_copy_franka_asset(node) and node.attrib.get("name") not in existing_names:
            copied = copy.deepcopy(node)
            _rewrite_franka_mesh_path(copied)
            target_asset.append(copied)
            existing_names.add(node.attrib.get("name"))
    for tag in ("actuator", "sensor"):
        old = root.find(tag)
        if old is not None:
            root.remove(old)
        source = franka_root.find(tag)
        if source is not None:
            root.append(copy.deepcopy(source))
    _replace_arm_torque_actuators_with_position(root)


def _remove_torc_robot(worldbody: ET.Element) -> None:
    for body in list(worldbody):
        if body.attrib.get("name") == "base":
            worldbody.remove(body)
            return
    if worldbody.find(".//body[@name='robot0_base']") is None:
        raise RuntimeError("scene did not contain TORC root body 'base' or Franka robot0_base")


def _add_franka_robot_body(worldbody: ET.Element) -> None:
    if worldbody.find(".//body[@name='robot0_base']") is not None:
        return
    franka_root = ET.parse(FRANKA_COMPILED_XML).getroot()
    robot_base = franka_root.find(".//body[@name='robot0_base']")
    if robot_base is None:
        raise RuntimeError("compiled Franka XML has no robot0_base body")
    robot_base = copy.deepcopy(robot_base)
    robot_base.set("pos", _fmt(_load_franka_base_xyz()))
    for body in robot_base.iter("body"):
        body.set("gravcomp", "1")
    worldbody.insert(2, robot_base)


def _stabilize_franka_arm_joints(root: ET.Element) -> None:
    damping = os.environ.get("TORC_FRANKA_ARM_JOINT_DAMPING", "2.0")
    armature = os.environ.get("TORC_FRANKA_ARM_JOINT_ARMATURE", "1.0")
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name", "")
        if re.fullmatch(r"robot0_joint[1-7]", name):
            joint.set("damping", damping)
            joint.set("armature", armature)


def build_franka_runtime_scene(scene_xml: str, experiment_dir: str | None = None) -> str:
    source = Path(scene_xml).resolve()
    out_dir = Path(experiment_dir).resolve() / "runtime_scene" if experiment_dir else PATCH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source.stem}_franka_runtime.xml"

    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(TORC_LAB_ROOT.resolve()))
    compiler.set("texturedir", str(TORC_LAB_ROOT.resolve()))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("TORC scene XML has no worldbody")
    _remove_torc_robot(worldbody)
    for tag in ("contact", "equality", "tendon", "keyframe"):
        node = root.find(tag)
        if node is not None:
            root.remove(node)
    _copy_franka_assets_and_actuators(root)
    _add_franka_robot_body(worldbody)
    _stabilize_franka_arm_joints(root)

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_vis = visual.find("global")
    if global_vis is None:
        global_vis = ET.SubElement(visual, "global")
    global_vis.set("offwidth", "1280")
    global_vis.set("offheight", "720")

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return str(out_path)
