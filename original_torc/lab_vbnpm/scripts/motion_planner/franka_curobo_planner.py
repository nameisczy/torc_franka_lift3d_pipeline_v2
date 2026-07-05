from dataclasses import dataclass

import numpy as np

try:
    from robot_interface.franka_adapter import RobotGraspCommand
    from robot_interface.franka_robot import FrankaRobot
except ImportError:
    from ..robot_interface.franka_adapter import RobotGraspCommand
    from ..robot_interface.franka_robot import FrankaRobot


@dataclass(frozen=True)
class PlannerResult:
    success: bool
    joint_name_order: tuple
    waypoints: tuple
    tcp_pregrasp_pose_world: np.ndarray
    tcp_contact_pose_world: np.ndarray
    collision_model: str
    planner_backend: str
    config_path: str
    diagnostics: dict


class FrankaCuroboPlanner:
    config_path = "assets/franka/config/curobo/franka_panda.yml"
    tcp_frame = "panda_tcp"
    mujoco_tcp_site = "gripper0_right_grip_site"
    joint_name_order = (
        "robot0_joint1",
        "robot0_joint2",
        "robot0_joint3",
        "robot0_joint4",
        "robot0_joint5",
        "robot0_joint6",
        "robot0_joint7",
    )

    def __init__(self, robot=None, model_xml_path=None, config_path=None):
        self.robot = robot or FrankaRobot(model_xml_path=model_xml_path)
        self.config_path = config_path or self.config_path
        self.collision_model = self.robot.collision_model()
        self.backend = "curobo_contract_with_mujoco_ik_fallback"

    def _lift_pose(self, pose: np.ndarray, lift_height: float = 0.18) -> np.ndarray:
        lifted = np.array(pose, dtype=float, copy=True)
        lifted[2, 3] += float(lift_height)
        return lifted

    def plan_grasp(self, command: RobotGraspCommand, seed=None):
        pre = self.robot.ik(command.tcp_pregrasp_pose_world, seed=seed)
        contact = self.robot.ik(command.tcp_contact_pose_world, seed=pre["q"])
        lift_pose = self._lift_pose(command.tcp_contact_pose_world)
        lift = self.robot.ik(lift_pose, seed=contact["q"])
        ok = bool(pre["success"] and contact["success"] and lift["success"])
        return PlannerResult(
            success=ok,
            joint_name_order=self.joint_name_order,
            waypoints=(pre["q"], contact["q"], lift["q"]),
            tcp_pregrasp_pose_world=command.tcp_pregrasp_pose_world,
            tcp_contact_pose_world=command.tcp_contact_pose_world,
            collision_model=command.collision_model,
            planner_backend=self.backend,
            config_path=self.config_path,
            diagnostics={"pregrasp": pre, "contact": contact, "lift": lift},
        )


def plan_robot_grasp(command: RobotGraspCommand, seed=None):
    return FrankaCuroboPlanner().plan_grasp(command, seed=seed)
