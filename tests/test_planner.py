from pathlib import Path
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "original_torc" / "lab_vbnpm" / "scripts"


def _read(path: Path) -> str:
    assert path.exists(), f"missing required file: {path}"
    return path.read_text()


def _load_geometry_diff() -> dict:
    path = PROJECT_ROOT / "geometry_diff.json"
    assert path.exists(), "Phase 2 geometry_diff.json is required before Phase 3 tests"
    return json.loads(path.read_text())


def test_franka_curobo_config_exists_and_uses_franka_frames():
    config = PROJECT_ROOT / "assets/franka/config/curobo/franka_panda.yml"
    text = _read(config)

    required = ["panda", "panda_link0", "panda_tcp", "panda_hand"]
    for token in required:
        assert token in text, f"Franka CuRobo config must contain {token}"

    forbidden = ["motoman", "robotiq", "motoman_right_ee", "arm_right_joint", "finger_joint: 0.0"]
    for token in forbidden:
        assert token not in text, f"Franka CuRobo config must not contain {token}"


def test_planner_consumes_robot_grasp_command_not_raw_grasp_pose():
    planner_path = SCRIPTS_ROOT / "motion_planner" / "franka_curobo_planner.py"
    text = _read(planner_path)

    assert "RobotGraspCommand" in text
    assert "tcp_pregrasp_pose_world" in text
    assert "tcp_contact_pose_world" in text
    assert "joint_name_order" in text

    forbidden = ["p_grasp", "hand_depth", "motoman_right_ee", "robotiq", "pose.translation += 0.11"]
    for token in forbidden:
        assert token not in text, f"Franka planner must not consume old TORC grasp token {token}"


def test_reachability_sweep_exists_and_uses_canonical_grasps():
    path = SCRIPTS_ROOT / "motion_planner" / "franka_reachability.py"
    text = _read(path)

    assert "CanonicalGrasp" in text, "reachability sweep must sample CanonicalGrasp inputs"
    assert "RobotAdapter" in text, "reachability sweep must adapt canonical grasps through adapter"
    assert "ik_feasible_fraction" in text
    assert "collision_free_fraction" in text
    assert "workspace_coverage" in text


def test_collision_feasibility_uses_franka_collision_model_only():
    adapter_path = SCRIPTS_ROOT / "robot_interface" / "franka_adapter.py"
    planner_path = SCRIPTS_ROOT / "motion_planner" / "franka_curobo_planner.py"
    combined = _read(adapter_path) + "\n" + _read(planner_path)

    assert "collision_model" in combined
    assert "franka_panda" in combined or "panda" in combined
    for token in ["robotiq_arg2f", "motoman_right_ee", "left_outer_finger", "right_outer_finger"]:
        assert token not in combined, f"Franka collision feasibility must not use TORC link {token}"


def test_planner_joint_order_matches_geometry_diff_contract():
    geometry = _load_geometry_diff()
    expected = geometry["robot_models"]["franka"]["arm_joint_names_compiled"]

    robot_path = SCRIPTS_ROOT / "robot_interface" / "franka_robot.py"
    text = _read(robot_path)

    for joint in expected:
        assert joint in text, f"FrankaRobot joint map missing {joint}"

    order_match = re.search(r"arm_joint_names\s*=\s*\((.*?)\)", text, flags=re.DOTALL)
    assert order_match, "FrankaRobot must declare arm_joint_names as an ordered tuple"
    declared = order_match.group(1)
    last_index = -1
    for joint in expected:
        idx = declared.find(joint)
        assert idx > last_index, f"joint {joint} is missing or out of order"
        last_index = idx
