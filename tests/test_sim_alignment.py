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


def _find(root, tag, name):
    for node in root.findall(f".//{tag}"):
        if node.get("name") == name:
            return node
    return None


def _vec(text):
    return [float(x) for x in text.split()]


def test_compiled_franka_mujoco_has_expected_base_tcp_and_actuators():
    root = _xml(PROJECT_ROOT / "assets/franka/mjcf/robosuite/panda_lift_compiled.xml")

    assert _find(root, "body", "robot0_base") is not None
    assert _find(root, "site", "gripper0_right_grip_site") is not None

    for joint in [f"robot0_joint{i}" for i in range(1, 8)]:
        assert _find(root, "joint", joint) is not None

    for joint in ["gripper0_right_finger_joint1", "gripper0_right_finger_joint2"]:
        assert _find(root, "joint", joint) is not None

    for actuator in ["gripper0_right_gripper_finger_joint1", "gripper0_right_gripper_finger_joint2"]:
        assert _find(root, "general", actuator) is not None


def test_base_alignment_solver_replaces_hardcoded_motoman_transform():
    solver_path = SCRIPTS_ROOT / "robot_interface" / "base_placement_solver.py"
    text = _read(solver_path)

    assert "ik_feasible_fraction" in text
    assert "collision_free_fraction" in text
    assert "workspace_coverage" in text
    assert "table_top" in text or "table_height" in text

    forbidden = ["world_to_base_link", "motoman_base", "base_link\" args=\"0 0 0"]
    for token in forbidden:
        assert token not in text, f"base placement solver must not hardcode TORC transform {token}"


def test_franka_launch_uses_franka_node_not_motoman_node():
    launch = PROJECT_ROOT / "original_torc/lab_vbnpm/launch/franka_control_server.launch"
    text = _read(launch)

    assert "franka_node.py" in text
    assert "franka_interface.py" in text
    assert "motoman_node.py" not in text
    assert "motoman_interface.py" not in text
    assert "motoman_sda10f" not in text


def test_gripper_actuation_contract_uses_panda_finger_joints():
    interface = SCRIPTS_ROOT / "execution_scene" / "franka_interface.py"
    node = SCRIPTS_ROOT / "execution_scene" / "franka_node.py"
    combined = _read(interface) + "\n" + _read(node)

    for token in [
        "gripper0_right_finger_joint1",
        "gripper0_right_finger_joint2",
        "gripper0_right_gripper_finger_joint1",
        "gripper0_right_gripper_finger_joint2",
    ]:
        assert token in combined

    for token in ["robotiq", "command_robotiq_action", "finger_joint\""]:
        assert token not in combined.lower(), f"Franka gripper actuation must not use {token}"


def test_task_success_oracle_is_object_and_lift_based():
    oracle_path = SCRIPTS_ROOT / "task_success" / "task_success_oracle.py"
    text = _read(oracle_path)

    required = ["TaskSuccessOracle", "object", "contact", "table", "lift"]
    for token in required:
        assert token.lower() in text.lower(), f"TaskSuccessOracle missing {token}"

    forbidden = ["robotiq_2f85", "gripper0_right_finger1_pad_collision", "gripper0_right_finger2_pad_collision"]
    for token in forbidden:
        assert token not in text, f"TaskSuccessOracle must not hardcode gripper geom {token}"


def test_mujoco_and_curobo_share_tcp_and_joint_order_contract():
    geometry = _load_geometry_diff()
    expected_joints = geometry["robot_models"]["franka"]["arm_joint_names_compiled"]

    robot_text = _read(SCRIPTS_ROOT / "robot_interface" / "franka_robot.py")
    planner_text = _read(SCRIPTS_ROOT / "motion_planner" / "franka_curobo_planner.py")
    sim_text = _read(SCRIPTS_ROOT / "execution_scene" / "franka_node.py")

    for text, label in [(robot_text, "FrankaRobot"), (planner_text, "Franka planner"), (sim_text, "Franka sim")]:
        assert "gripper0_right_grip_site" in text or "panda_tcp" in text, f"{label} must declare shared TCP"
        for joint in expected_joints:
            assert joint in text, f"{label} missing joint order token {joint}"


def test_franka_execution_requires_stable_settle_before_next_segment():
    node = SCRIPTS_ROOT / "execution_scene" / "franka_node.py"
    text = _read(node)

    assert "arm_goal_settle_count" in text
    assert "TORC_FRANKA_TRAJ_SETTLE_POS_TOL" in text
    assert "TORC_FRANKA_TRAJ_SETTLE_VEL_TOL" in text
    assert "TORC_FRANKA_TRAJ_SETTLE_STEPS" in text
    assert "TORC_FRANKA_TRAJ_SETTLE_TIMEOUT_S" in text
    assert 'os.environ.get("TORC_FRANKA_TRAJ_SETTLE_TIMEOUT_S", "45.0")' in text
    assert 'os.environ.get("TORC_FRANKA_TRAJ_SETTLE_VEL_TOL", "0.012")' in text


def test_franka_execution_preserves_torc_style_velocity_trajectory_contract():
    interface_text = _read(SCRIPTS_ROOT / "execution_scene" / "franka_interface.py")
    node_text = _read(SCRIPTS_ROOT / "execution_scene" / "franka_node.py")

    assert "franka_use_toppra_retime" in interface_text
    assert "self._linear_retime(" in interface_text
    assert "raw_plan, desired_duration, timestep" in interface_text
    assert "velocities=tuple(qd)" in interface_text
    assert "velocities=tuple([0.0] * len(joint_names))" not in interface_text

    assert "CubicHermiteSpline" not in node_text
    assert "(1.0 - alpha) * positions[idx] + alpha * positions[idx + 1]" in node_text
    assert "sample_positions = path(sample_times)" not in node_text


def test_franka_runtime_scene_uses_asset_consistent_servo_dynamics():
    scene_patch = _read(SCRIPTS_ROOT / "execution_scene" / "franka_scene_patch.py")
    interface_text = _read(SCRIPTS_ROOT / "execution_scene" / "franka_interface.py")

    assert 'os.environ.get("TORC_FRANKA_ARM_POSITION_KP", "320")' in scene_patch
    assert 'os.environ.get("TORC_FRANKA_ARM_JOINT_DAMPING", "8.0")' in scene_patch
    assert '"robot0_joint1": "5"' in scene_patch
    assert '"robot0_joint7": "0.714286"' in scene_patch
    assert 'joint.set("armature", armature_override or asset_armature[name])' in scene_patch
    assert 'os.environ.get("TORC_FRANKA_ARM_JOINT_ARMATURE", "1.0")' not in scene_patch

    assert '"/robot/franka_retime_padding", 2.2' in interface_text
    assert '"/robot/franka_min_segment_duration", 0.18' in interface_text
