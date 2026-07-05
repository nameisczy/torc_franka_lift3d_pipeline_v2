from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlannerNames:
    base_frame: str
    tcp_frame: str
    joint_name_order: tuple


@dataclass(frozen=True)
class SimNames:
    base_body: str
    tcp_site: str
    arm_joint_names: tuple
    finger_joint_names: tuple
    finger_actuator_names: tuple


class RobotBase:
    def fk(self, q):
        raise NotImplementedError

    def ik(self, tcp_pose_world, seed=None):
        raise NotImplementedError

    def collision_model(self):
        raise NotImplementedError

    def make_adapter(self):
        raise NotImplementedError


def as_pose(matrix):
    pose = np.asarray(matrix, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError("expected 4x4 pose")
    return pose
