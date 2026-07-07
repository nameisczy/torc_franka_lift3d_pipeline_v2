from dataclasses import dataclass
from pathlib import Path
import ast
import importlib.util
import math
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "original_torc" / "lab_vbnpm" / "scripts"


def _read(path: Path) -> str:
    assert path.exists(), f"missing required file: {path}"
    return path.read_text()


def _load_module(path: Path, module_name: str):
    assert path.exists(), f"missing required module: {path}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_canonical_grasp_dataclass_has_only_robot_independent_fields():
    path = SCRIPTS_ROOT / "grasp_representation" / "canonical_grasp.py"
    module = _load_module(path, "canonical_grasp_contract")

    assert hasattr(module, "CanonicalGrasp"), "CanonicalGrasp dataclass is required"
    fields = getattr(module.CanonicalGrasp, "__dataclass_fields__", {})
    assert set(fields) == {
        "contact_pose",
        "approach_direction",
        "grasp_axis",
        "opening_width",
        "score",
        "object_id",
    }

    forbidden_field_tokens = [
        "tcp",
        "ee",
        "robot",
        "franka",
        "panda",
        "motoman",
        "robotiq",
        "hand_depth",
        "joint",
        "geom",
    ]
    for field in fields:
        lowered = field.lower()
        assert not any(token in lowered for token in forbidden_field_tokens)


def test_canonical_grasp_module_rejects_robot_specific_tokens():
    path = SCRIPTS_ROOT / "grasp_representation" / "canonical_grasp.py"
    text = _read(path).lower()

    forbidden = [
        "motoman",
        "robotiq",
        "franka",
        "panda",
        "gripper0",
        "motoman_right_ee",
        "hand_depth",
        "0.11",
        "finger_joint",
    ]
    for token in forbidden:
        assert token not in text, f"CanonicalGrasp layer must not mention {token}"


def test_cgn_to_canonical_is_the_only_grasp_generator_boundary():
    path = SCRIPTS_ROOT / "grasp_representation" / "cgn_to_canonical.py"
    text = _read(path)

    assert "CanonicalGrasp" in text
    assert re.search(r"def\s+.*canonical", text.lower()), (
        "CGN conversion module must expose a conversion function to CanonicalGrasp"
    )
    for token in ["hand_depth", "motoman_right_ee", "robotiq", "gripper0_right", "panda_tcp"]:
        assert token.lower() not in text.lower(), f"CGN canonicalization must not contain {token}"


def test_grasp_filtering_is_split_between_canonical_and_robot_feasibility():
    canonical_filter = SCRIPTS_ROOT / "grasp_representation" / "canonical_filters.py"
    adapter = SCRIPTS_ROOT / "robot_interface" / "franka_adapter.py"

    canonical_text = _read(canonical_filter).lower()
    adapter_text = _read(adapter)

    for token in ["franka", "panda", "motoman", "robotiq", "tcp", "joint", "geom"]:
        assert token not in canonical_text, f"canonical filtering must not contain robot token {token}"

    assert "CanonicalGrasp" in adapter_text
    assert "RobotGraspCommand" in adapter_text
    assert "opening_width" in adapter_text, "adapter must consume canonical opening request"
    assert "0.076" in adapter_text or "validated_open" in adapter_text, (
        "Franka opening limit must be applied below RobotAdapter"
    )


def test_active_grasp_pipeline_does_not_consume_raw_cgn_pose_for_ik():
    open_loop = SCRIPTS_ROOT / "task_planner" / "curobo_open_loop.py"
    text = _read(open_loop)

    required = ["CanonicalGrasp", "RobotAdapter", "RobotGraspCommand"]
    for token in required:
        assert token in text, f"open-loop pipeline must use {token}"

    forbidden_patterns = [
        r"pose_t\[:3,\s*3\]\s*\+=",
        r"hand_depth\s*=\s*0\.11",
        r"pose_motion_plan\([^)]*p_grasp",
    ]
    for pattern in forbidden_patterns:
        assert not re.search(pattern, text, flags=re.DOTALL), (
            f"open-loop pipeline still contains forbidden TORC grasp path: {pattern}"
        )


def test_franka_grasp_geometry_lives_below_planner_boundary():
    planner = SCRIPTS_ROOT / "grasp_planner" / "curobo_grasp_planner.py"
    adapter = SCRIPTS_ROOT / "robot_interface" / "franka_grasp_adapter.py"
    planner_text = _read(planner)
    adapter_text = _read(adapter)

    assert "FrankaGraspGeometryConfig" in adapter_text
    assert "enclosure_score_weight" in adapter_text
    assert "scan_forward_extent_m: float = 0.020" in adapter_text
    assert "scan_step_m: float = 0.004" in adapter_text
    assert "scan_target_clearance_m: float = 0.001" in adapter_text
    assert "dz_candidates" in adapter_text
    assert "penetration_penalty" in adapter_text
    assert "scan_backward_extent_m: float = 0.002" in adapter_text
    assert "scan_dz_extent_m" not in adapter_text
    assert "TORC_FRANKA_SCAN_FORWARD_EXTENT_M" in adapter_text
    assert "TORC_FRANKA_SCAN_BACKWARD_EXTENT_M" in adapter_text
    assert "FrankaGraspAdapterScorer" in planner_text
    for method in [
        "scan_pose_offset_before_ik",
        "single_object_collides_with",
        "pad_penetrates_object",
        "enclosure_quality_scores",
    ]:
        assert method in adapter_text
        assert f"def _franka_{method}" not in planner_text

    assert "def _franka_single_object_collides_with" not in planner_text
    assert "franka enclosure score" in planner_text
    assert "franka enclosure rejection" not in planner_text
