#!/usr/bin/env python
from __future__ import annotations

import copy
import sys

import numpy as np
import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from lab_vbnpm.srv import EEControlResponse, ExecuteTrajectoryResponse
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from execution_scene.execution_interface import ExecutionInterface


class FrankaInterface(ExecutionInterface):
    joint_names = [
        "robot0_joint1",
        "robot0_joint2",
        "robot0_joint3",
        "robot0_joint4",
        "robot0_joint5",
        "robot0_joint6",
        "robot0_joint7",
    ]

    def __init__(self):
        super(FrankaInterface, self).__init__()
        self.follow_trajectory_client = actionlib.SimpleActionClient(
            "/joint_trajectory_action",
            FollowJointTrajectoryAction,
        )
        self.gripper_pub = rospy.Publisher("/franka/gripper_width", Float64, queue_size=3)

    def _get_param(self, name, default):
        try:
            return rospy.get_param(name, default)
        except Exception:
            return default

    def _segment_duration(self, q0, q1):
        vel_lim = float(self._get_param("/robot/vel_ang_lim", 20.0)) * np.pi / 180.0
        acc_lim = float(self._get_param("/robot/acc_ang_lim", 850.0)) * np.pi / 180.0
        vel_lim = max(vel_lim, 1e-3)
        acc_lim = max(acc_lim, 1e-3)
        padding = float(self._get_param("/robot/franka_retime_padding", 2.2))
        min_segment = float(self._get_param("/robot/franka_min_segment_duration", 0.18))
        dq = float(np.max(np.abs(np.asarray(q1, dtype=float) - np.asarray(q0, dtype=float))))
        if dq <= 1e-8:
            return 0.02
        # Trapezoid lower bound, padded so MuJoCo position actuators settle on short approach moves.
        by_vel = dq / vel_lim
        by_acc = 2.0 * np.sqrt(dq / acc_lim)
        return max(min_segment, padding * max(by_vel, by_acc))

    def _linear_retime(self, raw_plan, desired_duration, timestep):
        segment_times = np.asarray(
            [self._segment_duration(raw_plan[i], raw_plan[i + 1]) for i in range(len(raw_plan) - 1)],
            dtype=float,
        )
        scale = desired_duration / max(float(np.sum(segment_times)), 1e-6)
        knot_times = np.concatenate(([0.0], np.cumsum(segment_times * scale)))
        sample_times = np.arange(0.0, knot_times[-1] + timestep, timestep)
        if len(sample_times) == 0 or sample_times[-1] < knot_times[-1]:
            sample_times = np.append(sample_times, knot_times[-1])
        sample_times[-1] = knot_times[-1]
        positions = np.vstack(
            [np.interp(sample_times, knot_times, raw_plan[:, j]) for j in range(raw_plan.shape[1])]
        ).T
        velocities = np.gradient(positions, sample_times, axis=0, edge_order=1)
        velocities[0] = 0.0
        velocities[-1] = 0.0
        return positions, velocities, sample_times, knot_times

    def _retime_points(self, joint_names, points):
        if len(points) <= 1:
            return points
        raw_plan = np.asarray([point.positions for point in points], dtype=float)
        if raw_plan.shape[0] < 2:
            return points

        vel_lim = float(self._get_param("/robot/vel_ang_lim", 20.0)) * np.pi / 180.0
        acc_lim = float(self._get_param("/robot/acc_ang_lim", 850.0)) * np.pi / 180.0
        vel_lim = max(vel_lim, 1e-3)
        acc_lim = max(acc_lim, 1e-3)

        requested_duration = float(points[-1].time_from_start.to_sec() - points[0].time_from_start.to_sec())
        lower_bound = sum(
            self._segment_duration(raw_plan[i], raw_plan[i + 1])
            for i in range(len(raw_plan) - 1)
        )
        desired_duration = max(requested_duration, lower_bound, 0.02)
        timestep = float(self._get_param("/robot/franka_retime_timestep", 0.02))
        use_toppra = bool(int(self._get_param("/robot/franka_use_toppra_retime", 0)))
        if use_toppra:
            try:
                import toppra as ta
                import toppra.algorithm as ta_algo
                import toppra.constraint as ta_constraint

                path_param = np.linspace(0.0, 1.0, len(raw_plan))
                path = ta.SplineInterpolator(path_param, raw_plan)
                constraints = [
                    ta_constraint.JointVelocityConstraint([[-vel_lim, vel_lim]] * raw_plan.shape[1]),
                    ta_constraint.JointAccelerationConstraint([[-acc_lim, acc_lim]] * raw_plan.shape[1]),
                ]
                instance = ta_algo.TOPPRAsd(constraints, path)
                instance.set_desired_duration(desired_duration)
                trajectory = instance.compute_trajectory(0.0, 0.0)
                if trajectory is None:
                    raise RuntimeError("TOPPRA returned no trajectory")

                sample_times = np.arange(0.0, trajectory.duration + timestep, timestep)
                if len(sample_times) == 0 or sample_times[-1] < trajectory.duration:
                    sample_times = np.append(sample_times, trajectory.duration)
                sample_times[-1] = trajectory.duration
                positions = trajectory(sample_times)
                velocities = trajectory(sample_times, 1)
                knot_times = np.linspace(0.0, float(sample_times[-1]), len(raw_plan))
            except Exception as exc:
                rospy.logwarn(
                    "Franka TOPPRA retime unavailable (%s); using shape-preserving linear retime",
                    repr(exc),
                )
                positions, velocities, sample_times, knot_times = self._linear_retime(
                    raw_plan, desired_duration, timestep
                )
        else:
            positions, velocities, sample_times, knot_times = self._linear_retime(
                raw_plan, desired_duration, timestep
            )

        if len(raw_plan) >= 2:
            lo = np.minimum(raw_plan[:-1], raw_plan[1:])
            hi = np.maximum(raw_plan[:-1], raw_plan[1:])
            seg_idx = np.searchsorted(knot_times[1:], sample_times, side="right")
            seg_idx = np.clip(seg_idx, 0, len(raw_plan) - 2)
            positions = np.minimum(np.maximum(positions, lo[seg_idx]), hi[seg_idx])

        retimed = []
        for q, qd, t in zip(positions, velocities, sample_times):
            retimed.append(
                JointTrajectoryPoint(
                    positions=tuple(q),
                    velocities=tuple(qd),
                    time_from_start=rospy.Duration.from_sec(float(t)),
                )
            )
        return retimed

    def execute_trajectory(self, req):
        points = copy.deepcopy(req.trajectory.points)
        if len(points) == 0:
            return ExecuteTrajectoryResponse(None, -1)

        joint_names = list(req.trajectory.joint_names)
        first = np.asarray(points[0].positions, dtype=float)
        last = np.asarray(points[-1].positions, dtype=float)
        rospy.loginfo(
            "FrankaInterface execute request: joints=%s points=%d first_last_delta=%.6f duration=%.6f",
            joint_names,
            len(points),
            float(np.linalg.norm(last - first)),
            float(points[-1].time_from_start.to_sec()),
        )
        try:
            joint_state = rospy.wait_for_message("/joint_states_all", JointState, timeout=5)
            if points:
                indices = [joint_state.name.index(name) for name in joint_names]
                points[0].positions = tuple(np.asarray(joint_state.position)[indices])
                points[0].velocities = tuple([0.0] * len(indices))
        except Exception:
            pass
        if req.retime and len(points) > 1:
            points = self._retime_points(joint_names, points)
            rospy.loginfo(
                "FrankaInterface retimed trajectory: points=%d duration=%.6f",
                len(points),
                float(points[-1].time_from_start.to_sec()),
            )

        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = joint_names
        traj.points = points

        client = self.follow_trajectory_client
        if not client.wait_for_server(timeout=rospy.Duration(10)):
            print("Franka trajectory action server unavailable", file=sys.stderr)
            return ExecuteTrajectoryResponse(None, -2)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj
        client.send_goal(goal)
        rospy.loginfo("FrankaInterface goal sent")
        client.wait_for_result()
        result = client.get_result()
        error_code = int(result.error_code) if result is not None else 0
        rospy.loginfo("FrankaInterface goal result error_code=%d", error_code)
        return ExecuteTrajectoryResponse(None, error_code)

    def ee_control(self, req):
        if req.name not in ("panda", "franka"):
            rospy.logerr(f"No such gripper:{req.name}!")
            return EEControlResponse(False)
        width = float(np.clip(req.control, 0.0, 0.04))
        deadline = rospy.Time.now() + rospy.Duration(2.0)
        while self.gripper_pub.get_num_connections() == 0 and rospy.Time.now() < deadline:
            rospy.sleep(0.02)
        self.gripper_pub.publish(Float64(width))
        return EEControlResponse(True)

    def reset(self, init_joint_dict: dict = None):
        return None


if __name__ == "__main__":
    rospy.init_node("franka_interface")
    interface = FrankaInterface()
    interface.run()
