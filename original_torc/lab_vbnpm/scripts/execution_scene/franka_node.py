#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import deque
import copy
import os
import time

import numpy as np
import rospy
from control_msgs.msg import FollowJointTrajectoryResult
from industrial_msgs.msg import RobotStatus
from lab_vbnpm.srv import ExperimentResult, ExperimentResultResponse
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, UInt32
from trajectory_msgs.msg import JointTrajectoryPoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROS execution node for Franka MuJoCo simulation.")
    parser.add_argument("--scene", type=str, help="Path to the scene XML file.")
    parser.add_argument("--gui", type=lambda x: x.lower() in ("true", "t", "y", "yes"), default=False)
    parser.add_argument("--save-image-dir", type=str, default=None)
    parser.add_argument("--server-address", type=str, default="tcp://*:5858")
    parser.add_argument("--mj-pickle", type=lambda x: x.lower() in ("true", "t", "y", "yes"), default=False)
    args, _unknown = parser.parse_known_args()
    os.environ["MUJOCO_GL"] = "glfw" if args.gui else "egl"
    import mujoco  # noqa: F401
    if args.save_image_dir and args.save_image_dir.startswith("_"):
        args.save_image_dir = None

from execution_scene.execution_node import ExecutionNode
from execution_scene.franka_scene_patch import build_franka_runtime_scene


class FrankaNode(ExecutionNode):
    tcp_site_name = "gripper0_right_grip_site"
    arm_joint_names = [
        "robot0_joint1",
        "robot0_joint2",
        "robot0_joint3",
        "robot0_joint4",
        "robot0_joint5",
        "robot0_joint6",
        "robot0_joint7",
    ]
    finger_joint_names = [
        "gripper0_right_finger_joint1",
        "gripper0_right_finger_joint2",
    ]
    finger_actuator_joint_names = [
        "gripper0_right_gripper_finger_joint1",
        "gripper0_right_gripper_finger_joint2",
    ]

    def __init__(self, address: str = "tcp://*:5858"):
        self.joint_names = list(self.arm_joint_names)
        self.init_joints = {
            "robot0_joint1": 0.0,
            "robot0_joint2": -1.3,
            "robot0_joint3": 0.0,
            "robot0_joint4": -2.5,
            "robot0_joint5": 0.0,
            "robot0_joint6": 1.5,
            "robot0_joint7": 0.8,
        }
        self.open_finger_qpos = {
            "gripper0_right_finger_joint1": 0.04,
            "gripper0_right_finger_joint2": -0.04,
        }
        self.close_finger_qpos = {
            "gripper0_right_finger_joint1": 0.0,
            "gripper0_right_finger_joint2": 0.0,
        }
        super(FrankaNode, self).__init__(address)
        self.experiment_result_srv = rospy.Service(
            "/experiment_result",
            ExperimentResult,
            self.experiment_result_cb,
        )
        self.all_pub = rospy.Publisher("/joint_states", JointState, queue_size=5)
        self.all_full_pub = rospy.Publisher("/joint_states_all", JointState, queue_size=5)
        self.robot_status_pub = rospy.Publisher("/robot_status", RobotStatus, queue_size=5)
        self.robot_state_pub = rospy.Publisher("/robot_transfer_state", UInt32, queue_size=5)
        self.gripper_sub = rospy.Subscriber("/franka/gripper_width", Float64, self.gripper_width_cb)
        self.vel_deque = deque(maxlen=10)
        self.last_arm_qpos = np.asarray(list(self.init_joints.values()), dtype=float)
        self.last_status_time = time.time()
        self.gripper_geom_ids = set()
        self.arm_goal = None
        self.arm_goal_start_time = None
        self.arm_goal_settle_count = 0

    def reset(self, xml_file, gui=True, save_image_dir=None, experiment_dir=None, mj_pickle: bool = False):
        franka_xml = build_franka_runtime_scene(xml_file, experiment_dir)
        super().reset(franka_xml, gui, save_image_dir, experiment_dir, mj_pickle)
        self._set_joint_positions(self.open_finger_qpos, set_qpos=True, set_ctrl=True, zero_qvel=True)
        self.gripper_geom_ids = self._find_gripper_geom_ids()
        self.vel_deque.clear()
        self.last_arm_qpos = np.asarray(list(self.init_joints.values()), dtype=float)
        self.last_status_time = time.time()
        self.arm_goal = None
        self.arm_goal_start_time = None
        self.arm_goal_settle_count = 0

    def _joint_to_actuator(self):
        mapping = {}
        for aid in range(self.model.nu):
            jid = int(self.model.actuator_trnid[aid, 0])
            if jid >= 0:
                name = self.model.joint(jid).name
                if name and name not in mapping:
                    mapping[name] = aid
        return mapping

    def _set_joint_positions(self, joints, set_qpos=False, set_ctrl=True, zero_qvel=False):
        amap = self._joint_to_actuator() if set_ctrl else {}
        for name, value in joints.items():
            value = float(value)
            if set_qpos:
                jid = self.model.joint(name).id
                self.data.qpos[self.model.jnt_qposadr[jid]] = value
                if zero_qvel:
                    self.data.qvel[self.model.jnt_dofadr[jid]] = 0.0
            if set_ctrl and name in amap:
                lo, hi = self.model.actuator_ctrlrange[amap[name]]
                self.data.ctrl[amap[name]] = float(np.clip(value, lo, hi))

    def set_joint_angle(self, joints):
        if not isinstance(joints, dict):
            joints = dict(zip(self.joint_names, joints))
        self._set_joint_positions(joints, set_qpos=True, set_ctrl=True, zero_qvel=True)
        import mujoco

        mujoco.mj_forward(self.model, self.data)

    def get_ctrl_indices(self, joints, prefix="", replace=""):
        amap = self._joint_to_actuator()
        return np.asarray([amap[j] for j in joints], dtype=int)

    def gripper_width_cb(self, msg):
        if self.dense_video_recorder is not None:
            self.dense_video_recorder.start_recording()
        width = float(np.clip(msg.data, 0.0, 0.04))
        start = {
            name: self.get_joint_state([name])[0][0]
            for name in self.finger_joint_names
        }
        target = {
            "gripper0_right_finger_joint1": width,
            "gripper0_right_finger_joint2": -width,
        }
        steps = max(20, int(0.6 / self.model.opt.timestep))
        self.ee_trajectory = []
        for alpha in np.linspace(0.0, 1.0, steps):
            cmd = {
                name: (1.0 - alpha) * start[name] + alpha * target[name]
                for name in self.finger_joint_names
            }
            self.ee_trajectory.append(cmd)

    def _interpolate_points(self, joint_names, points):
        if not points:
            return []
        if len(points) == 1:
            return [(joint_names, np.asarray(points[0].positions, dtype=float), np.zeros(len(joint_names)), 0.0)]

        times = np.asarray([point.time_from_start.to_sec() for point in points], dtype=float)
        positions = np.asarray([point.positions for point in points], dtype=float)
        velocities = []
        for point in points:
            if len(point.velocities) == len(point.positions):
                velocities.append(point.velocities)
            else:
                velocities.append(np.zeros(len(point.positions)))
        velocities = np.asarray(velocities, dtype=float)

        keep = np.ones(len(times), dtype=bool)
        keep[1:] = np.diff(times) > 1e-9
        times = times[keep]
        positions = positions[keep]
        velocities = velocities[keep]
        if len(times) == 1:
            return [(joint_names, positions[0], velocities[0], float(times[0]))]

        if times[0] != 0.0:
            times = times - times[0]
        timestep = self.model.opt.timestep
        out = []
        for idx in range(len(times) - 1):
            t0 = float(times[idx])
            t1 = float(times[idx + 1])
            if t1 <= t0:
                continue
            segment_times = np.arange(t0, t1, timestep)
            if idx == 0 and (len(segment_times) == 0 or segment_times[0] != t0):
                segment_times = np.insert(segment_times, 0, t0)
            for t in segment_times:
                alpha = float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
                q = (1.0 - alpha) * positions[idx] + alpha * positions[idx + 1]
                qd = (1.0 - alpha) * velocities[idx] + alpha * velocities[idx + 1]
                out.append((joint_names, q, qd, float(t)))
        out.append((joint_names, positions[-1], velocities[-1], float(times[-1])))
        return out

    def follow_trajectory_cb(self):
        goal_command = self.follow_trajectory_as.accept_new_goal()
        traj = goal_command.trajectory
        joint_names = list(traj.joint_names)
        points = copy.deepcopy(traj.points)
        if points:
            first = np.asarray(points[0].positions, dtype=float)
            last = np.asarray(points[-1].positions, dtype=float)
            rospy.loginfo(
                "Franka goal received: joints=%s points=%d first_last_delta=%.6f duration=%.6f",
                joint_names,
                len(points),
                float(np.linalg.norm(last - first)),
                float(points[-1].time_from_start.to_sec()),
            )
        else:
            rospy.loginfo("Franka goal received with zero points")
        self.arm_trajectory = self._interpolate_points(joint_names, points)
        self.arm_goal = (
            dict(zip(joint_names, np.asarray(points[-1].positions, dtype=float)))
            if points
            else None
        )
        self.arm_goal_start_time = time.time() if self.arm_goal is not None else None
        rospy.loginfo(
            "Franka interpolated trajectory points=%d",
            len(self.arm_trajectory),
        )
        if self.arm_trajectory and self.dense_video_recorder is not None:
            self.dense_video_recorder.start_recording()
        if not self.arm_trajectory:
            result = FollowJointTrajectoryResult()
            result.error_code = 0
            self.follow_trajectory_as.set_succeeded(result)

    def _arm_goal_settled(self):
        if self.arm_goal is None:
            self.arm_goal_settle_count = 0
            return True
        names = list(self.arm_goal.keys())
        target = np.asarray([self.arm_goal[name] for name in names], dtype=float)
        qpos, qvel = self.get_joint_state(names)
        pos_err = float(np.max(np.abs(np.asarray(qpos, dtype=float) - target)))
        vel_err = float(np.max(np.abs(np.asarray(qvel, dtype=float))))
        pos_tol = float(os.environ.get("TORC_FRANKA_TRAJ_SETTLE_POS_TOL", "0.004"))
        vel_tol = float(os.environ.get("TORC_FRANKA_TRAJ_SETTLE_VEL_TOL", "0.012"))
        required = int(os.environ.get("TORC_FRANKA_TRAJ_SETTLE_STEPS", "24"))
        if pos_err <= pos_tol and vel_err <= vel_tol:
            self.arm_goal_settle_count += 1
        else:
            self.arm_goal_settle_count = 0
        if self.arm_goal_settle_count >= required:
            rospy.loginfo(
                "Franka trajectory settled: max_pos_err=%.6f max_vel=%.6f stable_steps=%d",
                pos_err,
                vel_err,
                self.arm_goal_settle_count,
            )
            return True
        timeout_s = float(os.environ.get("TORC_FRANKA_TRAJ_SETTLE_TIMEOUT_S", "45.0"))
        if self.arm_goal_start_time is not None and time.time() - self.arm_goal_start_time > timeout_s:
            rospy.logwarn(
                "Franka trajectory settle timeout: max_pos_err=%.6f max_vel=%.6f stable_steps=%d",
                pos_err,
                vel_err,
                self.arm_goal_settle_count,
            )
            return True
        return False

    def _finish_arm_goal_if_ready(self):
        if self.follow_trajectory_as.is_active() and self._arm_goal_settled():
            rospy.loginfo("Franka trajectory done")
            result = FollowJointTrajectoryResult()
            result.error_code = 0
            self.follow_trajectory_as.set_succeeded(result)
            self.arm_goal = None
            self.arm_goal_start_time = None
            self.arm_goal_settle_count = 0

    def do_traj(self):
        if self.arm_trajectory:
            joint_names, q, _qd, _t = self.arm_trajectory.pop(0)
            self._set_joint_positions(dict(zip(joint_names, q)), set_qpos=False, set_ctrl=True)
        elif self.arm_goal is not None:
            self._set_joint_positions(self.arm_goal, set_qpos=False, set_ctrl=True)
            self._finish_arm_goal_if_ready()
        if self.ee_trajectory:
            cmd = self.ee_trajectory.pop(0)
            self._set_joint_positions(cmd, set_qpos=False, set_ctrl=True)

    def _find_gripper_geom_ids(self):
        ids = set()
        for gid in range(self.model.ngeom):
            geom_name = self.model.geom(gid).name or ""
            body_name = self.model.body(self.model.geom(gid).bodyid[0]).name or ""
            if geom_name.startswith("gripper0_right_") or body_name.startswith("gripper0_right_"):
                ids.add(gid)
        return ids

    def experiment_result_cb(self, req):
        with self.mj_lock:
            dropped = []
            for i in range(self.model.nbody):
                body = self.model.body(i)
                name = body.name
                if (name.startswith(("object_", "obj_")) or (name and name[0] == "0")) and name != req.target:
                    if self.data.body(i).xpos[2] < 0.5:
                        dropped.append(name)

            grasping = set()
            for con_idx in range(self.data.ncon):
                con = self.data.contact[con_idx]
                g1, g2 = int(con.geom1), int(con.geom2)
                if g1 in self.gripper_geom_ids and g2 not in self.gripper_geom_ids:
                    body_id = int(self.model.geom(g2).bodyid[0])
                elif g2 in self.gripper_geom_ids and g1 not in self.gripper_geom_ids:
                    body_id = int(self.model.geom(g1).bodyid[0])
                else:
                    continue
                body_name = self.model.body(body_id).name
                if body_name.startswith(("object_", "obj_", "0")):
                    grasping.add(body_name)

            target = self.model.body(int(req.target)).name
            result = ExperimentResultResponse()
            result.success = target in grasping
            result.dropped = dropped
            result.grasping = sorted(grasping)
            return result

    def publish_joint_state(self, now):
        arm_position, arm_velocity = self.get_joint_state(self.arm_joint_names)
        finger_position, finger_velocity = self.get_joint_state(self.finger_joint_names)

        full = JointState()
        full.header.stamp = now
        full.name = self.arm_joint_names
        full.position = list(arm_position)
        full.velocity = list(arm_velocity)
        self.all_full_pub.publish(full)

        alias = JointState()
        alias.header.stamp = now
        alias.name = self.arm_joint_names + ["panda_gripper_width"]
        alias.position = list(arm_position) + [float(abs(finger_position[0] - finger_position[1]))]
        alias.velocity = list(arm_velocity) + [float(abs(finger_velocity[0] - finger_velocity[1]))]
        self.all_pub.publish(alias)

        t = time.time()
        dt = max(t - self.last_status_time, 1e-3)
        arm_q = np.asarray(arm_position, dtype=float)
        arm_speed = float(np.linalg.norm(arm_q - self.last_arm_qpos) / dt)
        self.last_arm_qpos = arm_q
        self.last_status_time = t
        self.vel_deque.append(arm_speed)

        status = RobotStatus()
        status.header.stamp = now
        status.error_code = int(np.mean(self.vel_deque) * 1e6) if self.vel_deque else 0
        moving = bool(self.arm_trajectory or self.ee_trajectory or (self.vel_deque and np.mean(self.vel_deque) > 2e-2))
        status.in_motion.val = moving
        status.in_error.val = False
        self.robot_status_pub.publish(status)
        self.robot_state_pub.publish(UInt32(int(moving)))


if __name__ == "__main__":
    rospy.init_node("execution_node")
    franka_node = FrankaNode(args.server_address)
    if args.scene:
        franka_node.reset(
            args.scene,
            gui=args.gui,
            save_image_dir=args.save_image_dir,
            mj_pickle=args.mj_pickle,
        )
    franka_node.run()
