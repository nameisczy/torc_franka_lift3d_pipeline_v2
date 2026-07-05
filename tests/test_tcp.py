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


def _find_body(root, name):
    for body in root.findall(".//body"):
        if body.get("name") == name:
            return body
    return None


def _find_site(root, name):
    for site in root.findall(".//site"):
        if site.get("name") == name:
            return site
    return None


def _vec(text):
    return [float(x) for x in text.split()]


def test_franka_tcp_site_exists_in_source_and_compiled_mjcf():
    source = ET.parse(PROJECT_ROOT / "assets/franka/mjcf/robosuite/panda_gripper.xml").getroot()
    compiled = ET.parse(PROJECT_ROOT / "assets/franka/mjcf/robosuite/panda_lift_compiled.xml").getroot()

    assert _find_body(source, "eef") is not None
    assert _find_site(source, "grip_site") is not None
    assert _find_body(compiled, "gripper0_right_eef") is not None
    assert _find_site(compiled, "gripper0_right_grip_site") is not None


def test_franka_tcp_offset_matches_geometry_diff():
    geometry = _load_geometry_diff()
    expected = geometry["robot_models"]["franka"]["tcp_contract"]["gripper_body_to_eef_offset_m"]

    source = ET.parse(PROJECT_ROOT / "assets/franka/mjcf/robosuite/panda_gripper.xml").getroot()
    eef = _find_body(source, "eef")
    assert eef is not None
    assert _vec(eef.get("pos")) == expected

    compiled = ET.parse(PROJECT_ROOT / "assets/franka/mjcf/robosuite/panda_lift_compiled.xml").getroot()
    compiled_eef = _find_body(compiled, "gripper0_right_eef")
    assert compiled_eef is not None
    assert _vec(compiled_eef.get("pos")) == expected


def test_tcp_correctness_contract_is_canonical_to_adapter_only():
    adapter_path = SCRIPTS_ROOT / "robot_interface" / "franka_adapter.py"
    text = _read(adapter_path)

    assert "CanonicalGrasp" in text, "Franka adapter must accept CanonicalGrasp"
    assert "RobotGraspCommand" in text, "Franka adapter must output RobotGraspCommand"
    assert "gripper0_right_grip_site" in text, "adapter must map to Franka TCP site"

    forbidden = ["hand_depth", "0.11", "motoman_right_ee", "robotiq"]
    for token in forbidden:
        assert token not in text.lower(), f"TCP adapter must not contain old TORC token {token}"


def test_no_pre_ik_translation_hack_remains_in_grasp_code():
    grasp_dir = SCRIPTS_ROOT / "grasp_planner"
    offenders = []
    for path in grasp_dir.glob("*.py"):
        text = path.read_text(errors="ignore")
        if "pose_t[:3, 3] += displace_t * approach_t" in text:
            offenders.append(path)
        if "self.hand_depth = 0.11" in text:
            offenders.append(path)

    assert not offenders, (
        "TORC pre-IK hand-depth translation must be removed from active grasp planner files: "
        + ", ".join(str(p) for p in offenders)
    )


def test_task_frames_do_not_use_motoman_tcp_names_for_franka():
    franka_files = [
        SCRIPTS_ROOT / "robot_interface" / "franka_robot.py",
        SCRIPTS_ROOT / "robot_interface" / "franka_adapter.py",
        SCRIPTS_ROOT / "motion_planner" / "franka_curobo_planner.py",
    ]
    for path in franka_files:
        text = _read(path)
        assert "motoman_right_ee" not in text
        assert "base_link -> motoman_right_ee" not in text
