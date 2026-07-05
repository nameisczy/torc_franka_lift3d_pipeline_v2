from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskSuccessOracle:
    lift_height: float = 0.08
    require_contact: bool = True

    def evaluate(self, object_pose_initial, object_pose_current, object_contact_state, table_height):
        initial = np.asarray(object_pose_initial, dtype=float).reshape(4, 4)
        current = np.asarray(object_pose_current, dtype=float).reshape(4, 4)
        lifted = float(current[2, 3]) >= max(float(table_height) + self.lift_height, float(initial[2, 3]) + self.lift_height)
        contact_ok = bool(object_contact_state) if self.require_contact else True
        return {
            "success": bool(lifted and contact_ok),
            "object_lift_delta": float(current[2, 3] - initial[2, 3]),
            "table_height": float(table_height),
            "contact": bool(object_contact_state),
            "lift": bool(lifted),
        }
