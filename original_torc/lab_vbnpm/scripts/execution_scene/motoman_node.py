#!/usr/bin/env python
"""
motoman execution node inheriting general execution_node
NOTE:
on the real robot, torso_joint_b2 mimics torso_joint_b1.
This only affects when we execute the trajectory.
We can add b2 if b1 is in the execute_trajectory
"""
import argparse
import os
import sys
import time
import copy
import numpy as np
from collections import deque
from scipy.interpolate import CubicHermiteSpline

import rospy
import actionlib
from std_msgs.msg import UInt32
from sensor_msgs.msg import JointState
from industrial_msgs.msg import RobotStatus
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from motoman_msgs.msg import (
    DynamicJointTrajectory,
    DynamicJointPoint,
    DynamicJointsGroup,
)
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryFeedback,
    FollowJointTrajectoryResult,
)
from robotiq_2f_gripper_msgs.msg import (
    CommandRobotiqGripperFeedback,
    CommandRobotiqGripperResult,
    CommandRobotiqGripperAction,
    CommandRobotiqGripperGoal,
)

from lab_vbnpm.srv import ExperimentResult, ExperimentResultResponse

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="ROS execution node for simulation.")

    # Define arguments
    parser.add_argument("--scene", type=str, help="Path to the scene XML file.")
    parser.add_argument(
        "--gui",
        type=lambda x: x.lower() in ("true", "t", "y", "yes"),
        default=False,
        help="Enable or disable the GUI. (true/false)",
    )
    parser.add_argument(
        "--save-image-dir",
        type=str,
        default=None,
        help="Directory to save simulation images.",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default="tcp://*:5858",
        help="Address of the ZMQ server, which can receive reset requests for the simulation.",
    )
    parser.add_argument(
        "--mj-pickle",
        type=lambda x: x.lower() in ("true", "t", "y", "yes"),
        default=False,
        help="Whether to save Mujoco state as pickle in experiment_result_cb",
    )

    # Parse arguments
    args, unknown = parser.parse_known_args()

    # Configure MUJOCO_GL and immediately import mujoco to
    # make it take effect
    os.environ["MUJOCO_GL"] = "glfw" if args.gui else "egl"
    import mujoco

    # Handle the save_image_dir logic
    if args.save_image_dir and args.save_image_dir.startswith("_"):
        args.save_image_dir = None


from execution_scene.execution_node import ExecutionNode


class MotomanNode(ExecutionNode):

    def __init__(self, address: str = "tcp://*:5858"):
        """
        define motoman-specific fields and interfaces
        """
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
        ]
        self.init_joints = {
            "torso_joint_b1": 0,
            "arm_left_joint_1_s": 1.75,
            "arm_left_joint_2_l": 0.8,
            "arm_left_joint_3_e": 0,
            "arm_left_joint_4_u": -0.66,
            "arm_left_joint_5_r": 0,
            "arm_left_joint_6_b": 0,
            "arm_left_joint_7_t": 0,
            # "arm_right_joint_1_s": 0.75,
            # "arm_right_joint_2_l": 0,
            # "arm_right_joint_3_e": -0.6,
            # "arm_right_joint_4_u": -1.15,
            # "arm_right_joint_5_r": 0,
            # "arm_right_joint_6_b": -1.3,
            # "arm_right_joint_7_t": 0.0,
            "arm_right_joint_1_s": 0.2,
            "arm_right_joint_2_l": -0.7,
            "arm_right_joint_3_e": 0.0,
            "arm_right_joint_4_u": -1.7,
            "arm_right_joint_5_r": 0,
            "arm_right_joint_6_b": -1.3,
            "arm_right_joint_7_t": 0.0,
        }
        self.gripper_name = "left_driver_joint"
        # self.ee_names = [self.gripper_name]

        super(MotomanNode, self).__init__(address)
        self.max_stroke = rospy.get_param("~stroke", 0.085)

        self.experiment_result_srv = rospy.Service(
            "/experiment_result",
            ExperimentResult,
            self.experiment_result_cb,
        )

        # * init the motoman-specific interfaces
        self.left_pub = rospy.Publisher(
            "/sda10f/sda10f_r1_controller/joint_states", JointState, queue_size=5
        )
        self.right_pub = rospy.Publisher(
            "/sda10f/sda10f_r2_controller/joint_states", JointState, queue_size=5
        )
        self.torso_b1_pub = rospy.Publisher(
            "/sda10f/sda10f_b1_controller/joint_states", JointState, queue_size=5
        )
        self.all_pub = rospy.Publisher(
            "/joint_states",
            JointState,
            queue_size=5,
        )
        self.robot_status_pub = rospy.Publisher(
            "/robot_status", RobotStatus, queue_size=5
        )
        self.robot_state_pub = rospy.Publisher(
            "/robot_transfer_state", UInt32, queue_size=5
        )
        self.vel_deque = deque(maxlen=10)

        # ** subscribers
        self.follow_trajectory_sub = rospy.Subscriber(
            "/joint_command", DynamicJointTrajectory, self.stream_trajectory_cb
        )
        self.last_t = 0

        # ** action servers
        # gripper
        self.robotiq_gripper_as = actionlib.SimpleActionServer(
            "command_robotiq_action",
            CommandRobotiqGripperAction,
            execute_cb=self.robotiq_gripper_cb,
            auto_start=False,
        )
        self.robotiq_gripper_as.start()

    def reset(
        self,
        xml_file,
        gui=True,
        save_image_dir=None,
        experiment_dir=None,
        mj_pickle: bool = False,
    ):
        super().reset(xml_file, gui, save_image_dir, experiment_dir, mj_pickle)

        self.vel_deque.clear()

        # * init robotiq geoms
        def has_ancestor(geom, name):
            parent = self.model.body(geom.bodyid[0])
            while parent.parentid:
                if parent.name == name:
                    return True
                parent = self.model.body(parent.parentid[0])
            return False

        self.gripper_geom_ids = set(
            filter(
                lambda i: has_ancestor(self.model.geom(i), "robotiq_2f85"),
                range(self.model.ngeom),
            )
        )

    def experiment_result_cb(self, req):
        """
        detect and return the experiment result
        success
        dropped
        grasping
        """
        with self.mj_lock:
            # DEBUG: Save the state right before running cb
            def debug_save_state(extra_data_dict: dict = {}):
                print("debug_save_state:")
                if self.experiment_dir is None:
                    print("  experiment_dir is None, nothing saved")
                    return
                import pickle

                # We want to save the complete state of the simulation, so we use mjSTATE_INTEGRATION
                raw_state = np.empty(
                    mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION)
                )
                mujoco.mj_getState(
                    self.model,
                    self.data,
                    raw_state,
                    mujoco.mjtState.mjSTATE_INTEGRATION,
                )

                # Get next available numbered file name
                def get_next_file_name(prefix: str, suffix: str) -> str:
                    i = 1
                    while True:
                        file_name = f"{self.experiment_dir}/{prefix}{i}{suffix}"
                        if not os.path.exists(file_name):
                            return file_name
                        i += 1

                state_file = get_next_file_name("state_", ".pkl")

                with open(state_file, "wb") as file:
                    pickle.dump(
                        {
                            "mujoco_state": raw_state,
                            "scene_xml": self.ws_xml_path,
                            "target": req.target,
                            **extra_data_dict,
                        },
                        file,
                    )
                print(f"  Saved state to {state_file}")

            dropped = []
            for i in range(self.model.nbody):
                body = self.model.body(i)
                dbody = self.data.body(i)
                name = body.name
                if (
                    name[:7] == "object_" or name[:4] == "obj_" or name[0] == "0"
                ) and name != req.target:
                    if dbody.xpos[2] < 0.5:
                        dropped.append(name)

            # Recursive gripper collision check: find objects in direct or indirect contact with gripper
            grasping = set()
            visited_body_ids = set()
            current_geom_ids = self.gripper_geom_ids.copy()

            while current_geom_ids:
                next_geom_ids = set()
                for geom_1_id, geom_2_id in self.data.contact.geom:
                    g1_in = geom_1_id in current_geom_ids
                    g2_in = geom_2_id in current_geom_ids

                    if g2_in and not g1_in:
                        g1_in, g2_in = g2_in, g1_in
                        geom_1_id, geom_2_id = geom_2_id, geom_1_id
                    if g1_in and not g2_in:
                        body_id = self.model.geom(geom_2_id).bodyid[0]
                        body_name = self.model.body(body_id).name
                        if (
                            body_id not in visited_body_ids
                            and geom_2_id not in self.gripper_geom_ids
                            and body_name.startswith(("object_", "obj_", "0"))
                        ):
                            visited_body_ids.add(body_id)
                            grasping.add(body_name)

                            # Add all geoms of this body for next iteration
                            for geom_idx in range(self.model.ngeom):
                                if self.model.geom(geom_idx).bodyid[0] == body_id:
                                    next_geom_ids.add(geom_idx)

                current_geom_ids = next_geom_ids

            grasping = list(grasping)

            target = self.model.body(int(req.target)).name

            # Consider a pick that grasps the pick target object as a success
            # This includes picks with multiple grasped objects.
            success = target in grasping

            result = ExperimentResultResponse()
            result.success = success
            result.dropped = dropped
            result.grasping = grasping

            if self.mj_pickle:
                debug_save_state(
                    {
                        "experiment_result": {
                            "success": success,
                            "dropped": dropped,
                            "grasping": grasping,
                        }
                    }
                )
            return result

    def stream_trajectory_cb(self, msg):
        if not msg.points:
            self.last_t = 0
            return

        point = msg.points[0]

        p1 = []
        v1 = []
        t1 = 0
        for g in point.groups:
            if g.group_number < 3:
                p1.extend(g.positions)
                v1.extend(g.velocities)
                t1 = g.time_from_start.to_sec()

        need_to_pop = False
        if not self.arm_trajectory:
            self.arm_trajectory = []
            p0, v0 = self.get_joint_state(self.joint_names)
        else:
            need_to_pop = True
            p0 = self.arm_trajectory[-1][1]
            v0 = self.arm_trajectory[-1][2]
        t = t1 - self.last_t
        self.last_t = t1

        if t <= 0:
            return  # ignore the first point

        times = np.array([0, t])
        positions = np.array([p0, p1])
        velocities = np.array([v0, v1])

        # interpolate by spline interpolation
        path = CubicHermiteSpline(times, positions, velocities)
        timestep = self.model.opt.timestep
        times = np.arange(0, times[-1] + timestep, timestep)
        positions = path(times)
        velocities = path(times, 1)
        # velocities[-1] = vel[-1]

        if need_to_pop:
            times = times[1:]
            positions = positions[1:]
            velocities = velocities[1:]

        self.arm_trajectory.extend(
            zip([self.joint_names] * len(times), positions, velocities, times)
        )

    def follow_trajectory_cb(self):
        goal_command = self.follow_trajectory_as.accept_new_goal()
        # goal_command = goal_command.get_goal()
        points = goal_command.trajectory.points
        joint_names = goal_command.trajectory.joint_names
        # change torso_joint_b2 to torso_joint_b1
        rospy.logdebug(f"New goal received! Trajectory Length:{len(points)}")

        times = np.zeros(len(points))
        positions = np.zeros((len(points), len(points[0].positions)))
        velocities = np.zeros((len(points), len(points[0].velocities)))
        for i, p in enumerate(points):
            times[i] = p.time_from_start.to_sec()
            positions[i] = p.positions
            velocities[i] = p.velocities

        # interpolate by spline interpolation
        path = CubicHermiteSpline(times, positions, velocities)
        timestep = self.model.opt.timestep
        new_times = np.arange(0, times[-1] + timestep, timestep)
        new_pos = path(new_times)
        new_vel = path(new_times, 1)
        new_vel[-1] = velocities[-1]
        new_acc = path(new_times, 2)

        times = new_times
        positions = new_pos
        velocities = new_vel

        # if 'torso_joint_b2' is in the joint_name, change to b1
        mjk_joint_names = copy.deepcopy(joint_names)
        if "torso_joint_b2" in joint_names:
            idx = joint_names.index("torso_joint_b2")
            mjk_joint_names[idx] = "torso_joint_b1"

        self.arm_trajectory = list(
            zip([mjk_joint_names] * len(times), positions, velocities, times)
        )

        return
        # While moving provide feedback and check for result
        rate = rospy.Rate(30)
        start_time = time.time()
        while not rospy.is_shutdown() and len(self.arm_trajectory) > 0:
            positions, velocities = self.get_joint_state(mjk_joint_names)
            actual = JointTrajectoryPoint()
            actual.positions = positions
            actual.velocities = velocities
            cur_time = time.time() - start_time
            actual.time_from_start = rospy.Time(cur_time)
            desired = JointTrajectoryPoint()
            desired.positions = path([cur_time])[0]
            desired.velocities = path([cur_time], 1)[0]

            feedback = FollowJointTrajectoryFeedback()
            feedback.joint_names = joint_names
            feedback.actual = actual
            feedback.desired = desired
            self.follow_trajectory_as.publish_feedback(feedback)
            rate.sleep()

        result.error_code = 0
        self.follow_trajectory_as.set_succeeded(result)

    def robotiq_gripper_cb(self, goal_command):
        """
        ref: https://github.com/Danfoa/robotiq_2finger_grippers/blob/master/robotiq_2f_gripper_control/scripts/robotiq_2f_action_server.py
        goal_command:
            position: 0 ~ max_stroke. 0 means closing.
            (in Mujoco we control in range [0,255], where 0 means open and 255 means closing)
        NOTE:
        updated code: gripper xml, control. Need testing.
        """
        result = CommandRobotiqGripperResult()
        if self.ee_trajectory:
            rospy.logdebug("Already controlling a gripper!")
            result.fault_status = 1
            self.follow_trajectory_as.set_aborted(result)
            return
        gctrl = self.get_ctrl_indices([self.gripper_name])
        # * set the trajectory
        # get the start position in [0,255]
        start_position = self.get_joint_state([self.gripper_name])[0][0]
        start_position = np.interp(start_position, [0.004, 0.79], [0, 255])
        requested_pos = np.interp(
            goal_command.position,
            [0, self.max_stroke],
            [255, 0],
        )
        requested_vel = goal_command.speed if goal_command.speed > 0 else 1
        requested_vel *= 255 / self.max_stroke
        requested_time = np.linalg.norm(start_position - requested_pos) / requested_vel
        positions = np.linspace(
            start_position,
            requested_pos,
            int(requested_time / self.model.opt.timestep),
        )

        self.ee_trajectory = list(
            zip([[self.gripper_name]] * len(positions), positions)
        )

        # Wait until goal is achieved and provide feedback
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            position = self.get_joint_state([self.gripper_name])[0][0]
            position = np.interp(position, [0.004, 0.79], [0, 255])
            position = np.interp(position, [0, 255], [self.max_stroke, 0])
            feedback = CommandRobotiqGripperFeedback()
            feedback.requested_position = goal_command.position
            feedback.position = position
            feedback.is_moving = True
            self.robotiq_gripper_as.publish_feedback(feedback)
            rate.sleep()
            if len(self.ee_trajectory) == 0:
                n_position = self.get_joint_state([self.gripper_name])[0][0]
                n_position = np.interp(n_position, [0.004, 0.79], [0, 255])
                n_position = np.interp(n_position, [0, 255], [self.max_stroke, 0])
                if np.allclose(position, n_position, atol=1e-4):
                    result.requested_position = goal_command.position
                    result.position = n_position
                    result.is_moving = False
                    break

        self.robotiq_gripper_as.set_succeeded(result)

    def publish_joint_state(self, now):
        """
        obtain joint state from Mujoco and publish
        """
        torso_b1 = ["torso_joint_b1"]
        left = [
            "arm_left_joint_1_s",
            "arm_left_joint_2_l",
            "arm_left_joint_3_e",
            "arm_left_joint_4_u",
            "arm_left_joint_5_r",
            "arm_left_joint_6_b",
            "arm_left_joint_7_t",
        ]
        right = [
            "arm_right_joint_1_s",
            "arm_right_joint_2_l",
            "arm_right_joint_3_e",
            "arm_right_joint_4_u",
            "arm_right_joint_5_r",
            "arm_right_joint_6_b",
            "arm_right_joint_7_t",
        ]
        gripper = ["finger_joint"]
        pub_jnts = [
            (self.torso_b1_pub, torso_b1),
            (self.left_pub, left),
            (self.right_pub, right),
            # (self.all_pub, torso_b1 + left + right),
        ]
        for pub, joint_names in pub_jnts:
            position, velocitiy = self.get_joint_state(joint_names)
            msg = JointState()
            msg.name = joint_names
            msg.position = position
            msg.velocity = velocitiy
            msg.header.stamp = rospy.Time.now()
            pub.publish(msg)

        # publish combined joint state (to make moveit happy)
        all_names = torso_b1 + left + right + [self.gripper_name]
        position, velocitiy = self.get_joint_state(all_names)
        msg = JointState()
        msg.name = torso_b1 + left + right + gripper
        msg.position = list(position)
        msg.velocity = list(velocitiy)
        msg.header.stamp = rospy.Time.now()
        self.all_pub.publish(msg)

        # publish robot status
        status = RobotStatus()
        status.header.stamp = rospy.Time.now()
        self.vel_deque.append(np.linalg.norm(velocitiy[:-1]))
        status.error_code = int(np.mean(self.vel_deque) * 1e6)
        if self.arm_trajectory or np.mean(self.vel_deque) > 2e-2:
            status.in_motion.val = True
            status.in_error.val = int(bool(self.arm_trajectory))
        else:
            status.in_motion.val = False
        self.robot_status_pub.publish(status)
        self.robot_state_pub.publish(UInt32(int(status.in_motion.val)))


if __name__ == "__main__":
    # Instantiate and run the node
    rospy.init_node("execution_node")
    motoman_node = MotomanNode(args.server_address)
    if args.scene:
        motoman_node.reset(args.scene, args.gui, args.save_image_dir, args.mj_pickle)
    motoman_node.run()
