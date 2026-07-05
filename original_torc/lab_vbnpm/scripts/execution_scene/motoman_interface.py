#!/usr/bin/env python
"""
NOTE:
motoman robot has to specify the full joint names to track the trajectory.
"""
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
from industrial_msgs.msg import RobotStatus
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from motoman_msgs.srv import CmdJointTrajectoryEx
from motoman_msgs.msg import (
    DynamicJointTrajectory,
    DynamicJointPoint,
    DynamicJointsGroup,
)
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryFeedback,
    FollowJointTrajectoryGoal,
)
from robotiq_2f_gripper_msgs.msg import (
    CommandRobotiqGripperAction,
    CommandRobotiqGripperFeedback,
    CommandRobotiqGripperGoal,
)
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


class MotomanInterface(ExecutionInterface):

    def __init__(self):
        super(MotomanInterface, self).__init__()
        self.joint_names = [
            "arm_left_joint_1_s",
            "arm_left_joint_2_l",
            "arm_left_joint_3_e",
            "arm_left_joint_4_u",
            "arm_left_joint_5_r",
            "arm_left_joint_6_b",
            "arm_left_joint_7_t",
            "arm_right_joint_1_s",
            "arm_right_joint_2_l",
            "arm_right_joint_3_e",
            "arm_right_joint_4_u",
            "arm_right_joint_5_r",
            "arm_right_joint_6_b",
            "arm_right_joint_7_t",
            "torso_joint_b1",
            "torso_joint_b2",  # this is a dummy joint. We only use for tracking
        ]

        self.update_params()

        # subscribe to split joint state topics
        self.left_state_sub = message_filters.Subscriber(
            "/sda10f/sda10f_r1_controller/joint_states", JointState
        )
        self.right_state_sub = message_filters.Subscriber(
            "/sda10f/sda10f_r2_controller/joint_states", JointState
        )
        self.torso_b1_sub = message_filters.Subscriber(
            "/sda10f/sda10f_b1_controller/joint_states", JointState
        )
        self.joint_state_sub = message_filters.ApproximateTimeSynchronizer(
            [self.left_state_sub, self.right_state_sub, self.torso_b1_sub],
            10,
            0.02,
        )
        self.joint_state_sub.registerCallback(self.joint_state_callback)

        # publishers
        self.joint_state_pub = rospy.Publisher(
            "/joint_states_all", JointState, queue_size=5
        )
        self.joint_command_pub = rospy.Publisher(
            "/joint_command", DynamicJointTrajectory, queue_size=20
        )

        # action clients
        self.robotiq_client = actionlib.SimpleActionClient(
            "/command_robotiq_action", CommandRobotiqGripperAction
        )
        self.follow_trajectory_client = actionlib.SimpleActionClient(
            "/joint_trajectory_action", FollowJointTrajectoryAction
        )

        # services
        self.next_point_srv = rospy.Service(
            "get_next_planning_point",
            GetNextPlanningPoint,
            self.next_point_cb,
        )
        self.wait_for = 0
        self.jnt_cmd_q = deque()
        self.lock = threading.Lock()

    def update_params(self):
        self.vel_ang_lim = rospy.get_param("/robot/vel_ang_lim", 10)
        self.acc_ang_lim = rospy.get_param("/robot/acc_ang_lim", 10)

    def run(self):
        while not rospy.is_shutdown():
            t0 = rospy.Time.now().to_sec()
            while rospy.Time.now().to_sec() - t0 < self.wait_for:
                rospy.sleep(0.001)
            while len(self.jnt_cmd_q) == 0:
                rospy.sleep(0.001)
            with self.lock:
                joint_commands = self.jnt_cmd_q.popleft()
                # print('Publishing:', len(joint_commands))
                for dyn_pt, jt_pnt in joint_commands:
                    if dyn_pt is None:
                        self.go_to_idle()
                    else:
                        dyn_traj = DynamicJointTrajectory()
                        dyn_traj.header.stamp = rospy.Time.now()
                        dyn_traj.joint_names = self.joint_names
                        dyn_traj.points = [dyn_pt]
                        self.joint_command_pub.publish(dyn_traj)

    def go_to_idle(self):
        self.joint_command_pub.publish(DynamicJointTrajectory())

    def get_next_point(self):
        if len(self.jnt_cmd_q) == 0:
            rospy.logwarn("Queue Empty!")
            return None
        else:
            rospy.logwarn("Next Point!")
            return self.jnt_cmd_q[0][0][1]

    def next_point_cb(self, req=None):
        with self.lock:
            return GetNextPlanningPointResponse(
                joint_names=self.joint_names[:-1],
                point=self.get_next_point(),
            )

    def joint_state_callback(self, left_msg, right_msg, torso_b1_msg):
        joint_names = list(left_msg.name)
        joint_names += list(right_msg.name)
        joint_names += list(torso_b1_msg.name)
        joint_pos = list(left_msg.position)
        joint_pos += list(right_msg.position)
        joint_pos += list(torso_b1_msg.position)
        joint_vels = list(left_msg.velocity)
        joint_vels += list(right_msg.velocity)
        joint_vels += list(torso_b1_msg.velocity)
        total_joint_msg = JointState()
        total_joint_msg.name = joint_names
        total_joint_msg.position = joint_pos
        total_joint_msg.velocity = joint_vels
        sec = max(
            left_msg.header.stamp.secs,
            right_msg.header.stamp.secs,
            torso_b1_msg.header.stamp.secs,
        )
        nsec = max(
            left_msg.header.stamp.nsecs,
            right_msg.header.stamp.nsecs,
            torso_b1_msg.header.stamp.nsecs,
        )
        total_joint_msg.header.stamp = rospy.Time(int(sec), int(nsec))
        self.joint_state_pub.publish(total_joint_msg)

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
                return False, float('inf')
                
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
                    start_vel_joint = start_vel[joint_idx] if len(start_vel) > joint_idx else 0
                    
                if np.isscalar(end_vel):
                    end_vel_joint = end_vel  
                else:
                    end_vel_joint = end_vel[joint_idx] if len(end_vel) > joint_idx else 0
                
                try:
                    spline = CubicSpline(
                        test_times,
                        joint_positions,
                        bc_type=((1, start_vel_joint), (1, end_vel_joint))
                    )
                    
                    # Sample velocities densely to find maximum
                    check_times = np.linspace(0, test_time, max(200, int(test_time / 0.005)))
                    velocities = np.abs(spline(check_times, 1))
                    max_vel_this_joint = np.max(velocities)
                    max_observed_vel = max(max_observed_vel, max_vel_this_joint)
                    
                except Exception:
                    return False, float('inf')
            
            return max_observed_vel <= vel_lim, max_observed_vel
        
        # Binary search for minimum feasible time
        lower_bound = max(min_theoretical_time, 0.1)  # At least 100ms
        upper_bound = max_time
        
        rospy.loginfo(f"Binary search range: [{lower_bound:.3f}, {upper_bound:.3f}]s")
        
        optimal_time = upper_bound
        for iteration in range(max_iterations):
            mid_time = (lower_bound + upper_bound) / 2.0
            
            is_feasible, max_vel_observed = check_velocity_feasibility(mid_time)
            
            if is_feasible:
                # This time works, try to go faster
                optimal_time = mid_time
                upper_bound = mid_time
                rospy.logdebug(f"Iteration {iteration+1}: {mid_time:.3f}s feasible (max vel: {max_vel_observed:.3f})")
            else:
                # This time is too fast, need more time
                lower_bound = mid_time
                rospy.logdebug(f"Iteration {iteration+1}: {mid_time:.3f}s too fast (max vel: {max_vel_observed:.3f})")
            
            # Check convergence
            if (upper_bound - lower_bound) < tolerance:
                break
        
        total_time = optimal_time
        rospy.loginfo(f"Optimized trajectory time: {total_time:.3f}s (theoretical min: {min_theoretical_time:.3f}s)")
        
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
                start_vel_joint = start_vel[joint_idx] if len(start_vel) > joint_idx else 0
                
            if np.isscalar(end_vel):
                end_vel_joint = end_vel  
            else:
                end_vel_joint = end_vel[joint_idx] if len(end_vel) > joint_idx else 0
            
            # Create cubic spline with boundary conditions
            spline = CubicSpline(
                original_times,
                joint_positions,
                bc_type=((1, start_vel_joint), (1, end_vel_joint))  # 1 means first derivative
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
            output_velocities[:, joint_idx] = spline(output_times, 1)  # First derivative
        
        # Add initial time offset
        output_times += t_0
        
        rospy.loginfo(f"Generated {n_output_points} points with timestep {timestep:.3f}s")
        
        return output_positions, output_velocities, output_times, None

    @staticmethod
    def make_dynamic_joint_point(pos, vel, duration):
        return DynamicJointPoint(
            num_groups=4,
            groups=[
                DynamicJointsGroup(
                    group_number=0,
                    num_joints=7,
                    valid_fields=0,
                    positions=pos[:7],
                    velocities=vel[:7],
                    accelerations=[0.0] * 7,
                    effort=[0.0] * 7,
                    time_from_start=duration,
                ),
                DynamicJointsGroup(
                    group_number=1,
                    num_joints=7,
                    valid_fields=0,
                    positions=pos[7:14],
                    velocities=vel[7:14],
                    accelerations=[0.0] * 7,
                    effort=[0.0] * 7,
                    time_from_start=duration,
                ),
                DynamicJointsGroup(
                    group_number=2,
                    num_joints=1,
                    valid_fields=0,
                    positions=pos[15:16],
                    velocities=vel[15:16],
                    accelerations=[0.0],
                    effort=[0.0],
                    time_from_start=duration,
                ),
                DynamicJointsGroup(
                    group_number=3,
                    num_joints=1,
                    valid_fields=0,
                    positions=pos[15:16],
                    velocities=vel[15:16],
                    accelerations=[0.0],
                    effort=[0.0],
                    time_from_start=duration,
                ),
            ],
        )

    def execute_trajectory(self, req):
        """
        Trajectory should specify all joint names to be tracked.
        Motoman has one dummy joint name torso_joint_b2.
        This does not need to be passed as the same
        value as torso_joint_b1 will be used for it.
        """

        t0 = rospy.Time.now().to_sec()
        joint_names = copy.deepcopy(req.trajectory.joint_names)
        points = copy.deepcopy(req.trajectory.points)
        if len(points) == 0:
            self.go_to_idle()
            return ExecuteTrajectoryResponse(None, -1)

        # set torso_joint_b2 as torso_joint_b1 if not specified
        if "torso_joint_b2" not in (joint_names):
            joint_names.append("torso_joint_b2")
            b1_idx = joint_names.index("torso_joint_b1")
            for i in range(len(points)):
                points[i].positions += (points[i].positions[b1_idx], )
                if points[i].velocities:
                    points[i].velocities += (points[i].velocities[b1_idx], )

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
                    positions, velocities, times, _jt = self.retime_trajectory_fixed_step(
                        points, 0.02, self.vel_ang_lim, self.acc_ang_lim
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
            if "torso_joint_b2" not in (joint_state.name):
                joint_state.name.append("torso_joint_b2")
                b1_idx = joint_state.name.index("torso_joint_b1")
                joint_state.position += (joint_state.position[b1_idx], )
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

        # retime the trajectory if requested and more than one point is passed
        positions = None
        if req.retime and len(points) > 1:
            self.update_params()
            # ensure at most 10 points per window
            timestep = req.execution_window.to_sec() / 5
            self.update_params()
            positions, velocities, times, _jt = self.retime_trajectory(
                points, timestep, self.vel_ang_lim, self.acc_ang_lim
            )
            if positions is None:
                rospy.logwarn("Trying simpler retime method...")
                positions, velocities, times, _jt = self.retime_trajectory_fixed_step(
                    points, timestep, self.vel_ang_lim, self.acc_ang_lim
                )
            if positions is None:
                rospy.logerr("Could not retime the trajectory!")
        if positions is None:
            positions = np.zeros((len(points), len(points[0].positions)))
            velocities = np.zeros((len(points), len(points[0].velocities)))
            times = np.zeros(len(points))
            for i, p in enumerate(points):
                positions[i] = p.positions
                if len(p.velocities) == len(p.positions):
                    velocities[i] = p.velocities
                times[i] = p.time_from_start.to_sec()

        rospy.logwarn(str(times))

        # format the position and velocities in the order of self.joint_names
        positions = positions[:, ordered_indices].astype(np.float32)
        velocities = velocities[:, ordered_indices].astype(np.float32)

        # stream points directly to robot bypassing jnt_cmd_q if requested
        # or only one point is passed
        stream_points = req.mode == ExecuteTrajectoryRequest.STREAM
        if len(points) == 1:
            rospy.logwarn("Only one point set! Will stream to robot directly.")
            stream_points = True

        if stream_points:
            # robot_status = rospy.wait_for_message('/robot_status', RobotStatus)
            # restart_streaming = robot_status.in_motion.val == 0
            robot_transfer_state = rospy.wait_for_message(
                "/robot_transfer_state", UInt32
            )
            restart_streaming = robot_transfer_state.data == 0
            if restart_streaming:
                # self.go_to_idle()
                rospy.logwarn("***Robot is stopped, restarting streaming.***")
                joint_state = rospy.wait_for_message(
                    "/joint_states_all", JointState
                )
                joint_state.name.append("torso_joint_b2")
                indx = joint_state.name.index("torso_joint_b1")
                joint_state.position += (joint_state.position[indx], )
                inds = list(map(joint_state.name.index, self.joint_names))
                pos0 = tuple(np.array(joint_state.position)[inds])
                vel0 = (0.0, ) * len(pos0)
                dur0 = float_to_ros_duration(0)
                rospy.logwarn("Dur0: " + str(dur0))
                dyn_pt0 = self.make_dynamic_joint_point(pos0, vel0, dur0)
                dyn_traj0 = DynamicJointTrajectory()
                dyn_traj0.header.stamp = rospy.Time.now()
                dyn_traj0.joint_names = self.joint_names
                dyn_traj0.points = [dyn_pt0]
                self.joint_command_pub.publish(dyn_traj0)
                times = np.arange(1, len(times) + 1).astype(float)
                times *= req.execution_window.to_sec()

            for pos, vel, duration in zip(positions, velocities, times):
                pos = tuple(pos)
                vel = tuple(vel)
                duration = float_to_ros_duration(duration)
                rospy.logwarn("Duration: " + str(duration))
                dyn_pt = self.make_dynamic_joint_point(pos, vel, duration)
                dyn_traj = DynamicJointTrajectory()
                dyn_traj.header.stamp = rospy.Time.now()
                dyn_traj.joint_names = self.joint_names
                dyn_traj.points = [dyn_pt]
                self.joint_command_pub.publish(dyn_traj)
            return ExecuteTrajectoryResponse(None, restart_streaming)

        if req.mode != ExecuteTrajectoryRequest.INTERUPT:
            rospy.logerr(f"Invalid mode: {req.mode}! Aborting...")
            return ExecuteTrajectoryResponse(None, -1)

        # create the dynamic joint trajectory and
        # the return list of interrupt points
        interrupt_points = JointTrajectory()
        interrupt_points.header.stamp = rospy.Time.now()
        interrupt_points.joint_names = self.joint_names[:-1]
        win_end = float_to_ros_duration(times[0]) + req.execution_window
        windows = [[]]
        for i in range(len(times)):
            duration = float_to_ros_duration(times[i])
            if duration >= win_end:
                interrupt_points.points.append(windows[-1][0][1])
                win_end += req.execution_window
                windows.append([])

            njnts = len(self.joint_names) - 1
            zeros = tuple([0.0] * njnts)
            pos = tuple(positions[i])
            vel = tuple(velocities[i])
            jt_pnt = JointTrajectoryPoint(
                positions=pos[:-1],
                velocities=vel[:-1],
                accelerations=zeros,
                effort=zeros,
                time_from_start=duration,
            )
            dyn_pt = self.make_dynamic_joint_point(pos, vel, duration)
            windows[-1].append((dyn_pt, jt_pnt))

        # add point to signal the end of the trajectory
        # so the controller goes to idle state
        windows[-1].append((None, None))

        if interrupt_points.points:
            interrupt_points.points.pop(0)
        t1 = rospy.Time.now().to_sec()
        rospy.logwarn("Time to retime:" + str(t1 - t0))
        with self.lock:
            # sanity check if goal is valid
            p0 = self.get_next_point()
            p1 = windows[0][0][1]
            if p0 is not None:
                at = p0.time_from_start.to_sec()
                bt = p1.time_from_start.to_sec()
                a = np.array(p0.positions)
                b = np.array(p1.positions)
                c = np.array(p0.velocities)
                d = np.array(p1.velocities)
                rospy.logwarn("distances:\n" + str(a - b))
                rospy.logwarn("velocities:\n" + str(c - d))
                rospy.logwarn(f"times: {at}, {bt}")
                if at != bt and np.linalg.norm(a - b) != 0:
                    rospy.logerr(GOAL_ERROR_MSG)
                    return ExecuteTrajectoryResponse(None, -5)
            else:
                robot_status = rospy.wait_for_message(
                    "/robot_status", RobotStatus
                )
                if robot_status.in_motion.val != 0:

                    rospy.logerr(EXEC_ERROR_MSG)
                    return ExecuteTrajectoryResponse(None, -5)
                # TODO: Check for the following:
                # Can fail for a short time after stopping
                # but within the time that a point stream
                # is still expected to be continued

            self.jnt_cmd_q.clear()
            # print('Window Size:', len(windows))
            for windowed_points in windows:
                # print('Queueing Points:', len(windowed_points))
                self.jnt_cmd_q.append(windowed_points)

            self.wait_for = req.execution_window.to_sec()

        return ExecuteTrajectoryResponse(interrupt_points, 0)

    def robotiq_gripper_control(self, req):
        control = req.control

        client = self.robotiq_client
        client.wait_for_server()
        action_goal = CommandRobotiqGripperGoal()
        action_goal.position = control
        action_goal.force = 35

        self.gas_done = False

        def feedback_cb(feedback):
            # rospy.loginfo('Receiving Gripper Feedback...')
            pass

        def done_cb(state, result):
            rospy.loginfo(
                f"""Gripper Action Server is Done.
                State: {state}, Result: {result.fault_status}"""
            )
            self.gas_done = True

        client.send_goal(action_goal, feedback_cb=feedback_cb, done_cb=done_cb)
        rospy.logdebug("Gripper Goal is Sent!")

        rate = rospy.Rate(30)
        while not self.gas_done:
            rate.sleep()

        result = client.get_result()
        rospy.logdebug(f"Done! {result.fault_status}")
        return EEControlResponse(result.fault_status == 0)

    def ee_control(self, req):
        ee_name = req.name
        if ee_name == "robotiq":
            return self.robotiq_gripper_control(req)
        else:
            rospy.logerr(f"No such gripper:{ee_name}!")
            return EEControlResponse(False)

    def reset(self, init_joint_dict: dict = None):
        """
        reset the robot to the given init_joint_dict
        """
        if init_joint_dict is None:
            init_joint_dict = {
                "torso_joint_b1": 0,
                "arm_left_joint_1_s": 1.75,
                "arm_left_joint_2_l": 0.8,
                "arm_left_joint_3_e": 0,
                "arm_left_joint_4_u": -0.66,
                "arm_left_joint_5_r": 0,
                "arm_left_joint_6_b": 0,
                "arm_left_joint_7_t": 0,
                # "arm_right_joint_1_s": 1.75,
                # "arm_right_joint_2_l": 0.8,
                # "arm_right_joint_3_e": 0,
                # "arm_right_joint_4_u": -0.66,
                # "arm_right_joint_5_r": 0,
                # "arm_right_joint_6_b": 0,
                # "arm_right_joint_7_t": 0
                "arm_right_joint_1_s": -0.2,
                "arm_right_joint_2_l": 0,
                "arm_right_joint_3_e": 0.2,
                "arm_right_joint_4_u": -0.8,
                "arm_right_joint_5_r": -0.25,
                "arm_right_joint_6_b": -1.85,
                "arm_right_joint_7_t": 0,
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
    rospy.on_shutdown(lambda: os.system("pkill -9 -f motoman_interface"))
    # rospy.sleep(1.0)
    execution_interface = MotomanInterface()
    # reset the trajectory
    rospy.sleep(1.0)
    # execution_interface.reset()
    execution_interface.run()
