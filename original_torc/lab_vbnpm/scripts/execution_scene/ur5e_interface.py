#!/usr/bin/env python
import os
import sys
import copy
import time
import queue
import random
import threading
from collections import deque

import numpy as np
import toppra as ta
import toppra.algorithm as ta_algo
import toppra.constraint as ta_constraint
from scipy.interpolate import CubicSpline

import rospy
import rospkg
import actionlib
import message_filters

from std_msgs.msg import UInt32
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryFeedback,
    FollowJointTrajectoryGoal,
)
# from onrobot_vg_control.srv import SetCommand, SetCommandResponse
from ur_msgs.srv import SetIO, SetIORequest

from lab_vbnpm.srv import EEControl, EEControlResponse
from lab_vbnpm.srv import GetNextPlanningPoint, GetNextPlanningPointResponse
from lab_vbnpm.srv import (
    ExecuteTrajectory,
    ExecuteTrajectoryRequest,
    ExecuteTrajectoryResponse,
)

# rp = rospkg.RosPack()
# package_path = rp.get_path('lab_vbnpm')
# sys.path.insert(0, os.path.join(package_path, 'scripts'))

from execution_scene.execution_interface import ExecutionInterface
from utils.conversions import joint_state_to_dict, float_to_ros_duration

GOAL_ERROR_MSG = """
******
Plan is not continuous!
Probably sent plan too close to next execution window.
******"""
EXEC_ERROR_MSG = (
    "Execution queue is empty but robot is still moving! Cannot execute new plan!"
)


class Ur5eInterface(ExecutionInterface):

    def __init__(self):
        super(Ur5eInterface, self).__init__()
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        self.update_params()

        # subscribe to split joint state topics
        self.joint_state_sub = rospy.Subscriber(
            "/joint_states",
            JointState,
            self.joint_state_callback,
        )

        # publishers
        self.joint_state_pub = rospy.Publisher(
            "/joint_states_all",
            JointState,
            queue_size=5,
        )

        # action clients
        ctrlrt = rospy.get_param("/scaled_pos_joint_traj_controller/type", None)
        if ctrlrt is not None:
            self.follow_trajectory_client = actionlib.SimpleActionClient(
                "/scaled_pos_joint_traj_controller/follow_joint_trajectory",
                FollowJointTrajectoryAction
            )
        else:
            self.follow_trajectory_client = actionlib.SimpleActionClient(
                "/joint_trajectory_action", FollowJointTrajectoryAction
            )

    def update_params(self):
        self.vel_ang_lim = rospy.get_param("/robot/vel_ang_lim", 10)
        self.acc_ang_lim = rospy.get_param("/robot/acc_ang_lim", 10)

    def joint_state_callback(self, joints_msg):
        # total_joint_msg = JointState()
        # total_joint_msg.name = list(joints_msg.name)
        # total_joint_msg.position = list(joints_msg.position)
        # total_joint_msg.velocity = list(joints_msg.velocity)
        # sec = joints_msg.header.stamp.secs,
        # nsec = joints_msg.header.stamp.nsecs
        # total_joint_msg.header.stamp = rospy.Time(int(sec), int(nsec))
        self.joint_state_pub.publish(joints_msg)

    @staticmethod
    def retime_trajectory(
        points,
        timestep,
        vel_ang_lim=5,
        acc_ang_lim=10,
        start_vel=0,
        end_vel=0,
    ):
        vel_lim = vel_ang_lim * np.pi / 180
        acc_lim = acc_ang_lim * np.pi / 180
        vel_limit = [-vel_lim, vel_lim]
        acc_limit = [-acc_lim, acc_lim]

        raw_plan = [p.positions for p in points]
        # np.random.seed(0)
        # random.seed(0)
        ss = np.linspace(0, 1, len(raw_plan))
        path = ta.SplineInterpolator(ss, raw_plan)
        vlims = [vel_limit] * len(raw_plan[0])
        alims = [acc_limit] * len(raw_plan[0])
        pc_vel = ta_constraint.JointVelocityConstraint(vlims)
        pc_acc = ta_constraint.JointAccelerationConstraint(alims)
        instance = ta_algo.TOPPRAsd([pc_vel, pc_acc], path)
        t_0 = points[0].time_from_start.to_sec()
        t_1 = points[-1].time_from_start.to_sec()
        instance.set_desired_duration(t_1 - t_0)
        jnt_traj = instance.compute_trajectory(start_vel, end_vel)
        if jnt_traj is None:
            rospy.logwarn(f"Error Code: {instance.problem_data.return_code}")
            # rospy.logwarn(f'K: {instance.problem_data.K}')
            return None, None, None, None
        t_1 = t_0 + jnt_traj.duration
        # print('t_0:', t_0, 't_1:', t_1)
        times = np.linspace(
            0,
            jnt_traj.duration,
            np.ceil((jnt_traj.duration / timestep)).astype(int),
        )
        positions = jnt_traj(times)
        velocities = jnt_traj(times, 1)
        times += t_0
        return positions, velocities, times, jnt_traj

    @staticmethod
    def retime_trajectory_fixed_step(
        points,
        timestep,
        vel_ang_lim=5,
        acc_ang_lim=10,
        start_vel=0,
        end_vel=0,
    ):
        """
        Alternative retime trajectory method that:
        1. Finds the fastest total trajectory time to keep all points within velocity limits
        2. Cubically interpolates trajectory points to be a fixed time step apart

        Args:
            points: List of JointTrajectoryPoint objects
            timestep: Fixed time step for output trajectory
            vel_ang_lim: Angular velocity limit in degrees/second
            acc_ang_lim: Angular acceleration limit in degrees/second^2 (unused in this method)
            start_vel: Starting velocity (scalar or array)
            end_vel: Ending velocity (scalar or array)

        Returns:
            positions: Array of joint positions
            velocities: Array of joint velocities
            times: Array of time stamps
            trajectory_obj: None (no trajectory object for this method)
        """
        if len(points) < 2:
            rospy.logwarn("Need at least 2 points for cubic interpolation")
            return None, None, None, None

        # Convert velocity limit from degrees to radians
        vel_lim = vel_ang_lim * np.pi / 180

        # Extract positions from trajectory points
        raw_positions = np.array([p.positions for p in points])
        n_joints = raw_positions.shape[1]
        n_points = raw_positions.shape[0]

        # Step 1: Find the minimum trajectory time using binary search optimization
        # This finds the fastest time that keeps all velocities within limits

        # Lower bound: theoretical minimum based on largest displacement
        displacements = np.abs(raw_positions[1:] - raw_positions[:-1])
        min_theoretical_time = np.max(np.max(displacements, axis=1) / vel_lim)

        # Upper bound: conservative estimate (sum of all segment times)
        max_segment_times = np.max(displacements, axis=1) / vel_lim
        max_time = np.sum(max_segment_times)

        # Binary search for optimal time
        tolerance = 0.001  # 1ms precision
        max_iterations = 20

        def check_velocity_feasibility(test_time):
            """Check if given total time satisfies velocity constraints"""
            if test_time <= 0:
                return False, float("inf")

            # Create uniform time parameterization
            test_times = np.linspace(0, test_time, n_points)

            # Create splines for this timing
            max_observed_vel = 0.0
            for joint_idx in range(n_joints):
                joint_positions = raw_positions[:, joint_idx]

                # Handle boundary conditions
                if np.isscalar(start_vel):
                    start_vel_joint = start_vel
                else:
                    start_vel_joint = (
                        start_vel[joint_idx]
                        if len(start_vel) > joint_idx else 0
                    )

                if np.isscalar(end_vel):
                    end_vel_joint = end_vel
                else:
                    end_vel_joint = (
                        end_vel[joint_idx] if len(end_vel) > joint_idx else 0
                    )

                try:
                    spline = CubicSpline(
                        test_times,
                        joint_positions,
                        bc_type=((1, start_vel_joint), (1, end_vel_joint)),
                    )

                    # Sample velocities densely to find maximum
                    check_times = np.linspace(
                        0, test_time, max(200, int(test_time / 0.005))
                    )
                    velocities = np.abs(spline(check_times, 1))
                    max_vel_this_joint = np.max(velocities)
                    max_observed_vel = max(max_observed_vel, max_vel_this_joint)

                except Exception:
                    return False, float("inf")

            return max_observed_vel <= vel_lim, max_observed_vel

        # Binary search for minimum feasible time
        lower_bound = max(min_theoretical_time, 0.1)  # At least 100ms
        upper_bound = max_time

        rospy.loginfo(
            f"Binary search range: [{lower_bound:.3f}, {upper_bound:.3f}]s"
        )

        optimal_time = upper_bound
        for iteration in range(max_iterations):
            mid_time = (lower_bound + upper_bound) / 2.0

            is_feasible, max_vel_observed = check_velocity_feasibility(mid_time)

            if is_feasible:
                # This time works, try to go faster
                optimal_time = mid_time
                upper_bound = mid_time
                rospy.logdebug(
                    f"Iteration {iteration+1}: {mid_time:.3f}s feasible (max vel: {max_vel_observed:.3f})"
                )
            else:
                # This time is too fast, need more time
                lower_bound = mid_time
                rospy.logdebug(
                    f"Iteration {iteration+1}: {mid_time:.3f}s too fast (max vel: {max_vel_observed:.3f})"
                )

            # Check convergence
            if (upper_bound - lower_bound) < tolerance:
                break

        total_time = optimal_time
        rospy.loginfo(
            f"Optimized trajectory time: {total_time:.3f}s (theoretical min: {min_theoretical_time:.3f}s)"
        )

        # Step 2: Create final time parameterization with optimized time
        original_times = np.linspace(0, total_time, n_points)

        # Get initial time offset from first point
        t_0 = points[0].time_from_start.to_sec()

        # Step 3: Create cubic spline interpolation for each joint
        splines = []
        for joint_idx in range(n_joints):
            joint_positions = raw_positions[:, joint_idx]

            # Set up boundary conditions for cubic spline
            # Handle start_vel and end_vel (can be scalar or array)
            if np.isscalar(start_vel):
                start_vel_joint = start_vel
            else:
                start_vel_joint = (
                    start_vel[joint_idx] if len(start_vel) > joint_idx else 0
                )

            if np.isscalar(end_vel):
                end_vel_joint = end_vel
            else:
                end_vel_joint = end_vel[joint_idx] if len(
                    end_vel
                ) > joint_idx else 0

            # Create cubic spline with boundary conditions
            spline = CubicSpline(
                original_times,
                joint_positions,
                bc_type=(
                    (1, start_vel_joint),
                    (1, end_vel_joint),
                ),  # 1 means first derivative
            )
            splines.append(spline)

        # Step 4: Generate fixed timestep trajectory
        # Calculate number of output points
        n_output_points = int(np.ceil(total_time / timestep)) + 1
        output_times = np.linspace(0, total_time, n_output_points)

        # Evaluate splines at fixed timesteps
        output_positions = np.zeros((n_output_points, n_joints))
        output_velocities = np.zeros((n_output_points, n_joints))

        for joint_idx, spline in enumerate(splines):
            output_positions[:, joint_idx] = spline(output_times)
            output_velocities[:, joint_idx] = spline(
                output_times, 1
            )  # First derivative

        # Add initial time offset
        output_times += t_0

        rospy.loginfo(
            f"Generated {n_output_points} points with timestep {timestep:.3f}s"
        )

        return output_positions, output_velocities, output_times, None

    def execute_trajectory(self, req):
        """
        Trajectory should specify all joint names to be tracked.
        """

        t0 = rospy.Time.now().to_sec()
        joint_names = copy.deepcopy(req.trajectory.joint_names)
        points = copy.deepcopy(req.trajectory.points)
        if len(points) == 0:
            return ExecuteTrajectoryResponse(None, -1)

        # make index map from joint_names to self.joint_names
        ordered_indices = list(map(joint_names.index, self.joint_names))

        # execute trajectory synchronously if requested or window = 0
        use_action_server = req.mode == ExecuteTrajectoryRequest.SYNCHRONOUS
        if req.execution_window.to_nsec() == 0:
            rospy.logwarn("Execution window is 0! Using action server...")
            use_action_server = True

        if use_action_server:
            if req.retime:
                self.update_params()
                positions, velocities, times, _jt = self.retime_trajectory(
                    points, 0.02, self.vel_ang_lim, self.acc_ang_lim
                )
                if positions is None:
                    rospy.logwarn("Trying simpler retime method...")
                    positions, velocities, times, _jt = (
                        self.retime_trajectory_fixed_step(
                            points, 0.02, self.vel_ang_lim, self.acc_ang_lim
                        )
                    )
                if positions is not None:
                    points.clear()
                    prev_tfs = None
                    rospy.logerr(str(times))
                    for i in range(len(times)):
                        tfs = float_to_ros_duration(times[i])
                        if tfs == prev_tfs:
                            continue
                        points.append(
                            JointTrajectoryPoint(
                                positions=positions[i],
                                velocities=velocities[i],
                                time_from_start=tfs,
                            )
                        )
                        prev_tfs = tfs
                else:
                    rospy.logerr("Could not retime the trajectory!")

            # debug displacement
            joint_state = rospy.wait_for_message(
                "/joint_states_all", JointState
            )
            rospy.logerr(
                "diff positions" +
                str(np.subtract(points[0].positions, joint_state.position))
            )
            # rospy.logwarn(
            #     "joint names" + str(joint_state.name) + str(joint_names)
            # )
            points[0].positions = joint_state.position
            points[0].velocities = [0] * len(joint_state.position)

            traj = JointTrajectory()
            traj.joint_names = joint_names
            traj.points = points
            client = self.follow_trajectory_client
            client.wait_for_server()
            action_goal = FollowJointTrajectoryGoal()
            action_goal.trajectory = traj
            self.tas_done = False

            def feedback(feedback: FollowJointTrajectoryFeedback):
                pass

            def done(state, result):
                rospy.loginfo(
                    "Trajectory Action Server is Done." +
                    f"State: {state}, Result: {result}" +
                    f"Types: {type(state)}, {type(result)}"
                )
                self.tas_done = True

            client.send_goal(action_goal, feedback_cb=feedback, done_cb=done)
            rospy.logdebug("Executing Trajectory in Open-Loop!")

            # result = None
            # rate = rospy.Rate(30)
            # while not self.tas_done:
            #     rate.sleep()
            client.wait_for_result()
            result = client.get_result()
            rospy.logwarn(f"Error code (0 = Success): {result}")
            return ExecuteTrajectoryResponse(None, result.error_code)

        # TODO: Implement Point Streaming Modes
        assert False, "Point Streaming Modes Not Implemented Yet!"

    def ee_control(self, req):
        ee_name = req.name
        if ee_name == "onrobot_vgc10":
            # rospy.wait_for_service("/onrobot_vg/set_command", timeout=5)
            # try:
            #     set_command = rospy.ServiceProxy(
            #         "/onrobot_vg/set_command", SetCommand
            #     )
            #     result = set_command("r" if req.control == 0 else "g")
            # except rospy.ServiceException as e:
            #     print(f"Service call failed: {e}", file=sys.stderr)
            #     return EEControlResponse(False)

            rospy.wait_for_service('/ur_hardware_interface/set_io', timeout=5)
            try:
                set_io = rospy.ServiceProxy(
                    '/ur_hardware_interface/set_io', SetIO
                )
                req_send = SetIORequest()
                req_send.fun = 1  # 1 = Digital Output
                req_send.pin = 0  # Pin 0
                req_send.state = req.control  # 1.0 is on 0.0 is off
                result = set_io(req_send)
            except rospy.ServiceException as e:
                print(f"Service call failed: {e}", file=sys.stderr)
                return EEControlResponse(False)

            return EEControlResponse(result.success)
        else:
            rospy.logerr(f"No such gripper:{ee_name}!")
            return EEControlResponse(False)

    def reset(self, init_joint_dict: dict = None):
        """
        reset the robot to the given init_joint_dict
        """
        if init_joint_dict is None:
            init_joint_dict = {
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -2.2,
                "elbow_joint": 1.9,
                "wrist_1_joint": -1.383,
                "wrist_2_joint": -1.57,
                "wrist_3_joint": 0.00,
            }
        # get current state
        cur_joint_state = rospy.wait_for_message(
            "/joint_states_all", JointState
        )
        cur_joint_dict, _ = joint_state_to_dict(cur_joint_state)
        joint_names = list(init_joint_dict.keys())
        positions = list(init_joint_dict.values())
        cur_joint_pos = [cur_joint_dict[name] for name in joint_names]
        # if the current position is close enough to the init, don't need to track
        diff = np.array(positions) - np.array(cur_joint_pos)
        if np.linalg.norm(diff, ord=np.inf) <= 1e-2:
            return
        # linearly interpolate between current and target
        interpolated_pos = np.linspace(cur_joint_pos, positions, 20)
        msg = JointTrajectory()
        msg.joint_names = joint_names
        pts = []
        for i in range(len(interpolated_pos)):
            pt = JointTrajectoryPoint()
            pt.positions = interpolated_pos[i]
            pts.append(pt)
        msg.points = pts
        rospy.wait_for_service("execute_trajectory")
        execute_traj = rospy.ServiceProxy(
            "execute_trajectory", ExecuteTrajectory
        )
        req = ExecuteTrajectoryRequest()
        req.trajectory = msg
        execute_traj(req)


if __name__ == "__main__":
    rospy.init_node("execution_interface")
    rospy.on_shutdown(lambda: os.system("pkill -9 -f ur5e_interface"))
    # rospy.sleep(1.0)
    execution_interface = Ur5eInterface()
    # reset the trajectory
    rospy.sleep(1.0)
    # execution_interface.reset()
    execution_interface.run()
