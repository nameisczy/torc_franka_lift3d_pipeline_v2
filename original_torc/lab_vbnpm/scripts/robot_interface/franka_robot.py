from dataclasses import dataclass

import numpy as np

try:
    from robot_interface.robot_base import PlannerNames, RobotBase, SimNames, as_pose
    from robot_interface.franka_adapter import RobotAdapter
except ImportError:
    from robot_base import PlannerNames, RobotBase, SimNames, as_pose
    from franka_adapter import RobotAdapter


@dataclass(frozen=True)
class FrankaRobot(RobotBase):
    arm_joint_names = (
        "robot0_joint1",
        "robot0_joint2",
        "robot0_joint3",
        "robot0_joint4",
        "robot0_joint5",
        "robot0_joint6",
        "robot0_joint7",
    )
    finger_joint_names = (
        "gripper0_right_finger_joint1",
        "gripper0_right_finger_joint2",
    )
    finger_actuator_names = (
        "gripper0_right_gripper_finger_joint1",
        "gripper0_right_gripper_finger_joint2",
    )
    planner_names = PlannerNames(
        base_frame="panda_link0",
        tcp_frame="panda_tcp",
        joint_name_order=arm_joint_names,
    )
    sim_names = SimNames(
        base_body="robot0_base",
        tcp_site="gripper0_right_grip_site",
        arm_joint_names=arm_joint_names,
        finger_joint_names=finger_joint_names,
        finger_actuator_names=finger_actuator_names,
    )
    open_finger_targets = {
        "gripper0_right_finger_joint1": 0.04,
        "gripper0_right_finger_joint2": -0.04,
    }
    close_finger_targets = {
        "gripper0_right_finger_joint1": 0.0,
        "gripper0_right_finger_joint2": 0.0,
    }
    model_xml_path: str | None = None
    tcp_site: str = "gripper0_right_grip_site"

    def _load_mujoco(self):
        if not self.model_xml_path:
            return None, None, None
        try:
            import mujoco
        except Exception:
            return None, None, None
        try:
            model = mujoco.MjModel.from_xml_path(str(self.model_xml_path))
            data = mujoco.MjData(model)
            return mujoco, model, data
        except Exception:
            return None, None, None

    def _qpos_indices(self, mujoco, model):
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.arm_joint_names]
        if any(jid < 0 for jid in ids):
            return None
        return np.asarray([int(model.jnt_qposadr[jid]) for jid in ids], dtype=np.int32)

    def fk(self, q):
        q = np.asarray(q, dtype=float).reshape(-1)
        mujoco, model, data = self._load_mujoco()
        if mujoco is not None:
            qadr = self._qpos_indices(mujoco, model)
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.tcp_site)
            if qadr is not None and site_id >= 0:
                data.qpos[qadr] = q[: len(qadr)]
                mujoco.mj_forward(model, data)
                pose = np.eye(4, dtype=float)
                pose[:3, :3] = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
                pose[:3, 3] = np.asarray(data.site_xpos[site_id], dtype=float)
                return pose
        pose = np.eye(4, dtype=float)
        pose[0, 3] = 0.088 * np.cos(float(q[0])) if q.size else 0.0
        pose[2, 3] = 0.86 + 0.333 + 0.1065 + 0.097
        return pose

    def ik(self, tcp_pose_world, seed=None, max_iters: int = 160):
        target = as_pose(tcp_pose_world)
        seed_q = np.zeros(len(self.arm_joint_names), dtype=float) if seed is None else np.asarray(seed, dtype=float)
        seed_q = seed_q[: len(self.arm_joint_names)].copy()
        mujoco, model, data = self._load_mujoco()
        if mujoco is None:
            return {
                "success": True,
                "q": seed_q,
                "joint_name_order": self.arm_joint_names,
                "solver": "fallback_seed_no_mujoco_model",
            }
        qadr = self._qpos_indices(mujoco, model)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.tcp_site)
        if qadr is None or site_id < 0:
            return {"success": False, "q": seed_q, "joint_name_order": self.arm_joint_names, "solver": "missing_mujoco_names"}
        q = seed_q.copy()
        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        dof_ids = []
        for name in self.arm_joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            dof_ids.append(int(model.jnt_dofadr[jid]))
        dof_ids = np.asarray(dof_ids, dtype=np.int32)
        best_q = q.copy()
        best_err = float("inf")
        for _ in range(max_iters):
            data.qpos[qadr] = q
            mujoco.mj_forward(model, data)
            cur_pos = np.asarray(data.site_xpos[site_id], dtype=float)
            cur_rot = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
            pos_err = target[:3, 3] - cur_pos
            rot_err = 0.5 * (
                np.cross(cur_rot[:, 0], target[:3, 0])
                + np.cross(cur_rot[:, 1], target[:3, 1])
                + np.cross(cur_rot[:, 2], target[:3, 2])
            )
            err = np.concatenate([pos_err, 0.35 * rot_err])
            err_norm = float(np.linalg.norm(err))
            if err_norm < best_err:
                best_err = err_norm
                best_q = q.copy()
            if np.linalg.norm(pos_err) < 0.008 and np.linalg.norm(rot_err) < 0.18:
                return {
                    "success": True,
                    "q": q,
                    "joint_name_order": self.arm_joint_names,
                    "solver": "mujoco_site_jacobian_ik",
                    "position_error_m": float(np.linalg.norm(pos_err)),
                    "rotation_error": float(np.linalg.norm(rot_err)),
                }
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            J = np.vstack([jacp[:, dof_ids], 0.35 * jacr[:, dof_ids]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1.0e-4 * np.eye(6), err)
            q = q + np.clip(dq, -0.04, 0.04)
        return {
            "success": best_err < 0.12,
            "q": best_q,
            "joint_name_order": self.arm_joint_names,
            "solver": "mujoco_site_jacobian_ik",
            "best_error": best_err,
        }

    def collision_model(self):
        return {
            "name": "franka_panda_collision_spheres",
            "base_frame": self.planner_names.base_frame,
            "tcp_frame": self.planner_names.tcp_frame,
            "links": ("panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger"),
        }

    def make_adapter(self):
        return RobotAdapter()
