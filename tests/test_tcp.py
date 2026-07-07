from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET
import importlib.util
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "original_torc" / "lab_vbnpm" / "scripts"


def _read(path: Path) -> str:
    assert path.exists(), f"missing required file: {path}"
    return path.read_text()


def _load_geometry_diff() -> dict:
    path = PROJECT_ROOT / "geometry_diff.json"
    assert path.exists(), "Phase 2 geometry_diff.json is required before Phase 3 tests"
    return json.loads(path.read_text())


def _load_module(path: Path, module_name: str):
    scripts_root = str(SCRIPTS_ROOT)
    if scripts_root not in sys.path:
        sys.path.insert(0, scripts_root)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_franka_tcp_contact_depth_is_derived_from_current_pad_geometry():
    adapter_path = SCRIPTS_ROOT / "robot_interface" / "franka_adapter.py"
    module = _load_module(adapter_path, "franka_adapter_contract")
    retreat = float(module.RobotAdapter._derive_tcp_contact_retreat_m())

    gripper = ET.parse(PROJECT_ROOT / "assets/franka/mjcf/robosuite/panda_gripper.xml").getroot()
    eef = _find_body(gripper, "eef")
    leftfinger = _find_body(gripper, "leftfinger")
    tip = _find_body(gripper, "finger_joint1_tip")
    pad = None
    for geom in gripper.findall(".//geom"):
        if geom.get("name") == "finger1_pad_collision":
            pad = geom
            break
    assert eef is not None and leftfinger is not None and tip is not None and pad is not None

    tcp_z = _vec(eef.get("pos"))[2]
    pad_front_z = (
        _vec(leftfinger.get("pos"))[2]
        + _vec(tip.get("pos"))[2]
        + _vec(pad.get("pos"))[2]
        + _vec(pad.get("size"))[2]
    )
    expected_retreat = (pad_front_z - tcp_z) + 0.001
    assert abs(retreat - expected_retreat) < 1e-6
    assert abs(module.RobotAdapter.derive_tcp_pad_front_m() - (pad_front_z - tcp_z)) < 1e-6


def test_franka_adapter_maps_canonical_closing_axis_to_current_mujoco_local_x():
    adapter_path = SCRIPTS_ROOT / "robot_interface" / "franka_adapter.py"
    module = _load_module(adapter_path, "franka_adapter_axis_contract")
    adapter = module.RobotAdapter()
    frame = adapter._frame_from([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    assert frame[:, 0].tolist() == [1.0, 0.0, 0.0]
    assert frame[:, 2].tolist() == [0.0, 0.0, 1.0]


def test_franka_adapter_does_not_apply_fixed_opening_width_lateral_shift():
    adapter_path = SCRIPTS_ROOT / "robot_interface" / "franka_adapter.py"
    grasp_path = SCRIPTS_ROOT / "grasp_representation" / "canonical_grasp.py"
    adapter_module = _load_module(adapter_path, "franka_adapter_centering_contract")
    grasp_module = _load_module(grasp_path, "canonical_grasp_centering_contract")

    grasp = grasp_module.CanonicalGrasp(
        contact_pose=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        approach_direction=[0.0, 0.0, 1.0],
        grasp_axis=[1.0, 0.0, 0.0],
        opening_width=0.08,
        score=1.0,
        object_id=1,
    )
    command = adapter_module.RobotAdapter().adapt(grasp)
    retreat = adapter_module.RobotAdapter._derive_tcp_contact_retreat_m()
    assert abs(command.tcp_contact_pose_world[0, 3]) < 1e-9
    assert abs(command.tcp_contact_pose_world[2, 3] + retreat) < 1e-9


def test_franka_adapter_scan_offsets_move_tcp_in_local_x_and_z(monkeypatch):
    adapter_path = SCRIPTS_ROOT / "robot_interface" / "franka_adapter.py"
    grasp_path = SCRIPTS_ROOT / "grasp_representation" / "canonical_grasp.py"
    adapter_module = _load_module(adapter_path, "franka_adapter_scan_offset_contract")
    grasp_module = _load_module(grasp_path, "canonical_grasp_scan_offset_contract")

    monkeypatch.setenv("TORC_FRANKA_TCP_LATERAL_OFFSET_M", "0.008")
    monkeypatch.setenv("TORC_FRANKA_TCP_APPROACH_OFFSET_M", "0.006")
    grasp = grasp_module.CanonicalGrasp(
        contact_pose=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        approach_direction=[0.0, 0.0, 1.0],
        grasp_axis=[1.0, 0.0, 0.0],
        opening_width=0.08,
        score=1.0,
        object_id=1,
    )
    command = adapter_module.RobotAdapter().adapt(grasp)
    retreat = adapter_module.RobotAdapter._derive_tcp_contact_retreat_m()
    assert abs(command.tcp_contact_pose_world[0, 3] - 0.008) < 1e-9
    assert abs(command.tcp_contact_pose_world[2, 3] - (-retreat + 0.006)) < 1e-9


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
