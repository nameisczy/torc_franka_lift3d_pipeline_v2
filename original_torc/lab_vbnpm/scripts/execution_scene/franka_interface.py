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
        dq = float(np.max(np.abs(np.asarray(q1, dtype=float) - np.asarray(q0, dtype=float))))
        if dq <= 1e-8:
            return 0.02
        # Trapezoid lower bound, padded slightly so MuJoCo position actuators settle.
        by_vel = dq / vel_lim
        by_acc = 2.0 * np.sqrt(dq / acc_lim)
        return max(0.08, 1.35 * max(by_vel, by_acc))

    def _retime_points(self, joint_names, points):
        if len(points) <= 1:
            return points
        retimed = []
        elapsed = 0.0
        prev = np.asarray(points[0].positions, dtype=float)
        retimed.append(
            JointTrajectoryPoint(
                positions=tuple(prev),
                velocities=tuple([0.0] * len(joint_names)),
                time_from_start=rospy.Duration(0.0),
            )
        )
        for point in points[1:]:
            target = np.asarray(point.positions, dtype=float)
            elapsed += self._segment_duration(prev, target)
            retimed.append(
                JointTrajectoryPoint(
                    positions=tuple(target),
                    velocities=tuple([0.0] * len(joint_names)),
                    time_from_start=rospy.Duration.from_sec(elapsed),
                )
            )
            prev = target
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
