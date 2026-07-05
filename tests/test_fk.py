from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "original_torc" / "lab_vbnpm" / "scripts"


def _read(path: Path) -> str:
    assert path.exists(), f"missing required file: {path}"
    return path.read_text()


def _load_geometry_diff() -> dict:
    path = PROJECT_ROOT / "geometry_diff.json"
    assert path.exists(), "Phase 2 geometry_diff.json is required before Phase 3 tests"
    return json.loads(path.read_text())


def _xml(path: Path):
    assert path.exists(), f"missing XML asset: {path}"
    return ET.parse(path).getroot()


def _joint_ranges(root):
    ranges = {}
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        rng = joint.get("range")
        if name and rng:
            ranges[name] = [float(x) for x in rng.split()]
    return ranges


def test_franka_asset_joint_ranges_match_geometry_diff():
    geometry = _load_geometry_diff()
    expected = geometry["robot_models"]["franka"]["joint_ranges_rad"]
    root = _xml(PROJECT_ROOT / "assets/franka/mjcf/robosuite/panda_robot.xml")
    ranges = _joint_ranges(root)

    for joint, expected_range in expected.items():
        assert joint in ranges, f"Franka source MJCF missing joint {joint}"
        assert ranges[joint] == expected_range

    assert len([name for name in ranges if name.startswith("joint")]) == 7


def test_franka_robot_fk_interface_is_present_and_robot_specific():
    path = SCRIPTS_ROOT / "robot_interface" / "franka_robot.py"
    text = _read(path)

    assert re.search(r"class\s+FrankaRobot\s*\(", text), "FrankaRobot class must exist"
    for method in ("fk", "ik", "collision_model", "make_adapter"):
        assert re.search(rf"def\s+{method}\s*\(", text), f"FrankaRobot must implement {method}()"

    required_names = [
        "robot0_joint1",
        "robot0_joint7",
        "gripper0_right_finger_joint1",
        "gripper0_right_finger_joint2",
        "gripper0_right_grip_site",
    ]
    for name in required_names:
        assert name in text, f"FrankaRobot must declare or map runtime name {name}"

    forbidden_runtime_names = ["motoman_right_ee", "robotiq_2f85"]
    for name in forbidden_runtime_names:
        assert name not in text, f"FrankaRobot must not depend on TORC/Motoman name {name}"


def test_fk_contract_declares_curobo_and_mujoco_name_maps():
    path = SCRIPTS_ROOT / "robot_interface" / "franka_robot.py"
    text = _read(path)

    assert "planner_names" in text, "FrankaRobot must expose CuRobo/planner name map"
    assert "sim_names" in text, "FrankaRobot must expose MuJoCo/sim name map"
    assert "panda_tcp" in text, "planner TCP alias panda_tcp must be defined"
    assert "gripper0_right_grip_site" in text, "MuJoCo TCP site must be defined"


def test_fk_tests_are_guarding_against_motoman_fallback():
    selector = SCRIPTS_ROOT / "task_planner" / "robot_selector.py"
    text = _read(selector)

    assert "franka" in text.lower(), "robot selector must include Franka path"
    assert "FrankaRobot" in text, "robot selector must instantiate/select FrankaRobot"
    assert "MotomanSDA10F" in text, "old Motoman path may remain for baseline parity"
