#!/usr/bin/env python3
"""Write measured old Motoman/Robotiq assumptions from source-reading evidence."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("/mnt/ssd/ziyaochen/torc_franka_clean_results/old_robot_gripper_assumptions.json")
SOURCE_ROOT = Path("/home/ziyaochen/gc6d_lift3d_traj/ros_workspace/src/lab_vbnpm")


def main() -> int:
    data = {
        "schema_version": "1.0.0",
        "evidence_level": "proven",
        "source_root": str(SOURCE_ROOT),
        "old_robot_class": {
            "value": "task_planner.motoman.MotomanSDA10F",
            "source": str(SOURCE_ROOT / "scripts/task_planner/motoman.py"),
            "line_range": "class definition and __init__",
            "evidence_level": "proven",
        },
        "old_ee_link": {
            "value": "motoman_right_ee",
            "sources": [
                str(SOURCE_ROOT / "scripts/task_planner/motoman.py"),
                str(SOURCE_ROOT / "robots/motoman/curobo/motoman_left.yml"),
            ],
            "evidence_level": "proven",
        },
        "old_base_frame": {
            "value": "base_link",
            "source": str(SOURCE_ROOT / "robots/motoman/curobo/motoman_left.yml"),
            "evidence_level": "proven",
        },
        "old_joint_order_and_start_state": {
            "lock_joints": {
                "arm_right_joint_1_s": 1.75,
                "arm_right_joint_2_l": 0.8,
                "arm_right_joint_3_e": 0.0,
                "arm_right_joint_4_u": -0.66,
                "arm_right_joint_5_r": 0.0,
                "arm_right_joint_6_b": 0.0,
                "arm_right_joint_7_t": 0.0,
                "finger_joint": 0.0,
            },
            "source": str(SOURCE_ROOT / "robots/motoman/curobo/motoman_left.yml"),
            "evidence_level": "proven",
        },
        "robotiq_open_close_width_constants_m": {
            "open": 0.085,
            "close": 0.0,
            "source": str(SOURCE_ROOT / "scripts/task_planner/eutils.py"),
            "evidence_level": "proven",
        },
        "grasp_planner_hand_depth_m": {
            "cgn": 0.11,
            "gpd": 0.045,
            "gpg": 0.035,
            "ground_truth": 0.0,
            "source": str(SOURCE_ROOT / "scripts/grasp_planner/curobo_grasp_planner.py"),
            "evidence_level": "proven",
        },
        "contact_graspnet_build_6d_grasp_gripper_depth_m": {
            "value": 0.1034,
            "source": "/home/ziyaochen/gc6d_lift3d_franka_7d_generation/third_party/contact_graspnet/contact_graspnet/contact_graspnet.py",
            "function": "build_6d_grasp",
            "evidence_level": "proven",
        },
        "contact_graspnet_config": {
            "DATA.gripper_width_m": 0.08,
            "TEST.extra_opening_m": 0.005,
            "sources": [
                "/home/ziyaochen/gc6d_lift3d_franka_7d_generation/third_party/contact_graspnet/contact_graspnet/config.yaml",
                "/home/ziyaochen/gc6d_lift3d_traj/ros_workspace/src/contact_graspnet_ros/checkpoints/scene_2048_bs3_rad2_32/config.yaml",
            ],
            "evidence_level": "proven",
        },
        "attach_extra_spheres": {
            "link": "ee_link from RobotWorldConfig, Motoman resolves to motoman_right_ee",
            "spheres": [
                {"center_xyz_m": [0.0, -0.04 + 0.02 * i, 0.0], "radius_m": 0.012}
                for i in range(5)
            ],
            "source": str(SOURCE_ROOT / "scripts/grasp_planner/curobo_grasp_planner.py"),
            "function": "attach_extra_spheres",
            "evidence_level": "proven",
        },
        "is_grasp_above_table_assumptions": {
            "finger_width_m": 0.0065,
            "outer_diameter_m": 0.098,
            "plotter_hand_depth_m": 0.055,
            "half_width_computed_m": 0.04575,
            "default_safety_margin_m": 0.001,
            "approach_axis": "pose_matrix[:3, 2]",
            "closing_axis": "-pose_matrix[:3, 1]",
            "source": str(SOURCE_ROOT / "scripts/grasp_planner/curobo_grasp_planner.py"),
            "function": "is_grasp_above_table",
            "evidence_level": "proven",
        },
        "approach_offset": {
            "value_m": "self.hand_depth added along pose local +Z before IK",
            "cgn_value_m": 0.11,
            "source": str(SOURCE_ROOT / "scripts/grasp_planner/curobo_grasp_planner.py"),
            "evidence_level": "proven",
        },
        "table_clearance_assumptions": {
            "threshold_height": "table_height + safety_margin",
            "default_safety_margin_m": 0.001,
            "geometry_source": "GraspPlotter defaults consumed in is_grasp_above_table",
            "evidence_level": "proven",
        },
        "collision_link_names": {
            "ignore_collision_ee_links_initial": [
                "motoman_right_ee",
                "left_outer_knuckle",
                "left_outer_finger",
                "left_inner_finger",
                "left_inner_finger_pad",
                "left_inner_knuckle",
                "right_outer_knuckle",
                "right_outer_finger",
                "right_inner_finger",
                "right_inner_finger_pad",
                "right_inner_knuckle",
                "robotiq_arg2f_extra_link",
                "robotiq_arg2f_base_link",
                "arm_right_link_7_t",
                "arm_right_link_6_b",
            ],
            "curobo_ignore_collision_ee_links": [
                "robotiq_arg2f_base_link",
                "left_outer_finger",
                "right_outer_finger",
                "right_inner_finger",
                "left_inner_finger",
            ],
            "source": str(SOURCE_ROOT / "scripts/task_planner/motoman.py"),
            "evidence_level": "proven",
        },
        "hardcoded_frames_or_transforms": {
            "world_to_base_link_launch": "0 0 0 0 0 0 1 base base_link",
            "workspace_pose_sim": [0.61, -0.5, 0.90],
            "workspace_size_sim": [0.58, 1.0, 0.5],
            "source": str(SOURCE_ROOT / "scripts/task_planner/motoman.py"),
            "evidence_level": "proven",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
