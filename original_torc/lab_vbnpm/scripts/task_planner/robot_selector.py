"""Opt-in TORC robot selector.

Default behavior remains Motoman.  Franka is selected only when
``TORC_ROBOT=franka`` is present in the environment.
"""

from __future__ import annotations

import os
from typing import Any


MOTOMAN_RESET_JOINTS = {
    "torso_joint_b1": 0,
    "arm_left_joint_1_s": 1.75,
    "arm_left_joint_2_l": 0.8,
    "arm_left_joint_3_e": 0,
    "arm_left_joint_4_u": -0.66,
    "arm_left_joint_5_r": 0,
    "arm_left_joint_6_b": 0,
    "arm_left_joint_7_t": 0,
    "arm_right_joint_1_s": 0.2,
    "arm_right_joint_2_l": -0.7,
    "arm_right_joint_3_e": 0.0,
    "arm_right_joint_4_u": -1.7,
    "arm_right_joint_5_r": 0,
    "arm_right_joint_6_b": -1.3,
    "arm_right_joint_7_t": 0.0,
}

FRANKA_RESET_JOINTS = {
    "robot0_joint1": 0.0,
    "robot0_joint2": -1.3,
    "robot0_joint3": 0.0,
    "robot0_joint4": -2.5,
    "robot0_joint5": 0.0,
    "robot0_joint6": 1.5,
    "robot0_joint7": 0.8,
}


def get_torc_robot_type() -> str:
    robot_type = os.environ.get("TORC_ROBOT", "motoman").strip().lower()
    if robot_type in ("", "motoman", "sda10f"):
        return "motoman"
    if robot_type in ("franka", "panda"):
        return "franka"
    raise ValueError(
        f"Unsupported TORC_ROBOT={robot_type!r}. Expected 'motoman' or 'franka'."
    )


def get_robot_reset_joints(robot_type: str | None = None) -> dict[str, float]:
    robot_type = robot_type or get_torc_robot_type()
    if robot_type == "motoman":
        return dict(MOTOMAN_RESET_JOINTS)
    if robot_type == "franka":
        return dict(FRANKA_RESET_JOINTS)
    raise ValueError(f"Unsupported robot_type={robot_type!r}")


def make_robot(is_sim: bool, ground_truth: bool, robot_type: str | None = None) -> Any:
    robot_type = robot_type or get_torc_robot_type()
    if robot_type == "motoman":
        from task_planner.motoman import MotomanSDA10F

        return MotomanSDA10F(is_sim, ground_truth)
    if robot_type == "franka":
        from robot_interface.franka_robot import FrankaRobot
        from task_planner.franka import FrankaTORCRobot

        return FrankaTORCRobot(is_sim, ground_truth)
    raise ValueError(f"Unsupported robot_type={robot_type!r}")
