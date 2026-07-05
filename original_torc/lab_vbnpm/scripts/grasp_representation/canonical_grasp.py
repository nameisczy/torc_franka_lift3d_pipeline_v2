from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CanonicalGrasp:
    contact_pose: np.ndarray
    approach_direction: np.ndarray
    grasp_axis: np.ndarray
    opening_width: float
    score: float
    object_id: int

    def validate(self):
        contact_pose = np.asarray(self.contact_pose, dtype=float)
        approach_direction = np.asarray(self.approach_direction, dtype=float).reshape(3)
        grasp_axis = np.asarray(self.grasp_axis, dtype=float).reshape(3)
        if contact_pose.shape != (4, 4):
            raise ValueError("contact_pose must be 4x4")
        if not np.allclose(contact_pose[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("contact_pose must be homogeneous")
        for name, vec in (("approach_direction", approach_direction), ("grasp_axis", grasp_axis)):
            norm = float(np.linalg.norm(vec))
            if not np.isfinite(norm) or norm <= 0.0:
                raise ValueError(f"{name} must be nonzero")
        if float(self.opening_width) <= 0.0:
            raise ValueError("opening_width must be positive")
        return self
