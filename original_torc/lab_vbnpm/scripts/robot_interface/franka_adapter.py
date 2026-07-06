from dataclasses import dataclass
import math

import numpy as np

try:
    from grasp_representation.canonical_grasp import CanonicalGrasp
except ImportError:
    from ..grasp_representation.canonical_grasp import CanonicalGrasp


@dataclass(frozen=True)
class RobotGraspCommand:
    tcp_contact_pose_world: np.ndarray
    tcp_pregrasp_pose_world: np.ndarray
    gripper_opening_width: float
    tcp_site: str
    collision_model: str


class RobotAdapter:
    tcp_site = "gripper0_right_grip_site"
    validated_open_width = 0.08
    opening_clearance_m = 0.012
    pregrasp_distance = 0.10
    tcp_contact_depth_m = -0.0036
    collision_model = "franka_panda_collision_spheres"

    def _frame_from(self, approach_direction, grasp_axis):
        z_axis = np.asarray(approach_direction, dtype=float).reshape(3)
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_seed = np.asarray(grasp_axis, dtype=float).reshape(3)
        x_seed = x_seed / np.linalg.norm(x_seed)
        x_axis = x_seed - z_axis * float(np.dot(x_seed, z_axis))
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        return np.stack([x_axis, y_axis, z_axis], axis=1)

    def adapt(self, grasp: CanonicalGrasp) -> RobotGraspCommand:
        grasp.validate()
        contact = np.array(grasp.contact_pose, dtype=float, copy=True)
        contact[:3, :3] = self._frame_from(grasp.approach_direction, grasp.grasp_axis)
        contact[:3, 3] += 0.5 * float(grasp.opening_width) * contact[:3, 0]
        contact[:3, 3] -= self.tcp_contact_depth_m * contact[:3, 2]
        pregrasp = contact.copy()
        pregrasp[:3, 3] -= self.pregrasp_distance * contact[:3, 2]
        requested = float(grasp.opening_width)
        opening = min(max(requested + self.opening_clearance_m, 0.0), self.validated_open_width)
        if not math.isfinite(opening):
            raise ValueError("opening must be finite")
        return RobotGraspCommand(
            tcp_contact_pose_world=contact,
            tcp_pregrasp_pose_world=pregrasp,
            gripper_opening_width=opening,
            tcp_site=self.tcp_site,
            collision_model=self.collision_model,
        )
