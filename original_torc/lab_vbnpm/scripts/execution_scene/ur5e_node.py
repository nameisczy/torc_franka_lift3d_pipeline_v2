#!/usr/bin/env python
"""
ur5e execution node inheriting general execution_node
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
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryFeedback,
    FollowJointTrajectoryResult,
)

# from onrobot_vg_control.srv import SetCommand, SetCommandResponse
from ur_msgs.srv import SetIO, SetIOResponse

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


class Ur5eNode(ExecutionNode):

    def __init__(self, address: str = "tcp://*:5858"):
        """
        define ur5e-specific fields and interfaces
        """
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.init_joints = {
            "shoulder_pan_joint": 0,
            "shoulder_lift_joint": -2.2,
            "elbow_joint": 1.9,
            "wrist_1_joint": -1.383,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.00,
        }

        super(Ur5eNode, self).__init__(address)

        self.experiment_result_srv = rospy.Service(
            "/experiment_result",
            ExperimentResult,
            self.experiment_result_cb,
        )

        # ** init the ur5e-specific interfaces
        self.all_pub = rospy.Publisher(
            "/joint_states",
            JointState,
            queue_size=5,
        )

        self.vel_deque = deque(maxlen=10)

        # ** services
        # suction
        # self.set_command_srv = rospy.Service(
        #     "/onrobot_vg/set_command", SetCommand, self.suction_cmd_cb
        # )
        self.set_command_srv = rospy.Service(
            "/ur_hardware_interface/set_io", SetIO, self.suction_cmd_cb
        )

    def suction_cmd_cb(self, req):
        """Handles suction command requests"""
        # rospy.loginfo(str(req.command))
        rospy.loginfo(str(req.state))

        suction_pos = self.data.site_xpos[self.get_site_id("suction")]

        closest_obj = None
        closest_dist = float("inf")
        for i in range(self.model.nsite):
            site = self.model.site(i)
            name = site.name
            if name.startswith("obj_"):
                site_id = site.id
                site_pos = self.data.site_xpos[site_id]
                dist = np.linalg.norm(suction_pos - site_pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_obj = name.replace("_site", "")
        weld_id = self.get_equality_id(closest_obj)

        # sind = self.model.actuator("suction").id
        # rospy.logerr("*** SIND ***: " + str(sind))
        # if req.command == "g":
        if req.state == 1:
            # self.data.ctrl[sind] = 1.0  # activate suction
            self.data.eq_active[weld_id] = 1
        # elif req.command == "r":
        elif req.state == 0:
            # self.data.ctrl[sind] = 0.0  # deactivate suction
            self.data.eq_active[weld_id] = 0

        # return SetCommandResponse(success=True, message=None)
        return SetIOResponse(success=True)

    # override
    def set_joint_angle(self, joints):
        if type(joints) == dict:
            joint_names = list(joints.keys())
            position = np.array(list(joints.values()))
        else:
            joint_names = self.joint_names
            position = np.array(joints)
        # velocity = [0] * len(position)

        iqpos = self.get_qpos_indices(joint_names)
        pctrl = self.get_ctrl_indices(joint_names)
        # vctrl = self.get_ctrl_indices(joint_names, replace='_v')
        # intact = self.get_act_indices(joint_names, replace="_intv")
        # intvctrl = self.get_ctrl_indices(joint_names, replace="_intv")
        self.data.qpos[iqpos] = position
        self.data.ctrl[pctrl] = position
        # self.data.ctrl[vctrl] = velocity
        # self.data.act[intact] = position
        # self.data.ctrl[intvctrl] = velocity
        print("setting joint angle...")

        mujoco.mj_forward(self.model, self.data)

    # override
    def do_traj(self):
        """
        track the trajectory if there are ones inside the saved list
        """
        # * controlling arm
        if self.arm_trajectory:
            joint_names, position, velocity, _time = self.arm_trajectory.pop(0)
            # iqpos = self.get_qpos_indices(joint_names)
            pctrl = self.get_ctrl_indices(joint_names)
            # vctrl = self.get_ctrl_indices(joint_names, replace='_v')
            # intact = self.get_act_indices(joint_names, replace="_intv")
            # intvctrl = self.get_ctrl_indices(joint_names, replace="_intv")
            self.data.ctrl[pctrl] = position
            # self.data.ctrl[vctrl] = velocity
            # self.data.act[intact] = position
            # self.data.ctrl[intvctrl] = velocity
        else:
            # iqpos = self.get_qpos_indices(self.joint_names)
            # pctrl = self.get_ctrl_indices(self.joint_names)
            # vctrl = self.get_ctrl_indices(self.joint_names, replace='_v')
            # intact = self.get_act_indices(self.joint_names, replace="_intv")
            # intvctrl = self.get_ctrl_indices(self.joint_names, replace="_intv")
            # self.data.ctrl[pctrl] = self.data.qpos[iqpos]
            # self.data.ctrl[vctrl] = 0
            # self.data.ctrl[intvctrl] = 0

            is_active = self.arm_trajectory is not None
            is_active &= self.follow_trajectory_as.is_active()
            if is_active:
                rospy.loginfo("Trajectory is done!")
                result = FollowJointTrajectoryResult()
                result.error_code = 0
                self.follow_trajectory_as.set_succeeded(result)
                self.arm_trajectory = None

    def reset(self, xml_file, gui=True, save_image_dir=None):
        super().reset(xml_file, gui, save_image_dir)

        self.vel_deque.clear()

        # * init robotiq geoms
        def has_ancestor(geom, name):
            parent = self.model.body(geom.bodyid[0])
            while parent.parentid:
                if parent.name == name:
                    return True
                parent = self.model.body(parent.parentid[0])
            return False

        self.suction_geom_ids = set(
            filter(
                lambda i: has_ancestor(self.model.geom(i), "onrobot_vgc10_suction_cup"),
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

            grasping = set()
            for g1, g2 in self.data.contact.geom:
                gripisg1 = g1 in self.suction_geom_ids
                gripisg2 = g2 in self.suction_geom_ids
                if gripisg1 and not gripisg2:
                    objid = self.model.geom(g2).bodyid[0]
                    grasping.add(self.model.body(objid).name)
                if gripisg2 and not gripisg1:
                    objid = self.model.geom(g1).bodyid[0]
                    grasping.add(self.model.body(objid).name)
            grasping = list(grasping)

            target = self.model.body(int(req.target)).name

            success = len(grasping) == 1 and grasping[0] == target

            result = ExperimentResultResponse()
            result.success = success
            result.dropped = dropped
            result.grasping = grasping
            return result

    def follow_trajectory_cb(self):
        goal_command = self.follow_trajectory_as.accept_new_goal()
        # goal_command = goal_command.get_goal()
        points = goal_command.trajectory.points
        joint_names = goal_command.trajectory.joint_names
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

        self.arm_trajectory = list(
            zip([joint_names] * len(times), positions, velocities, times)
        )

        return

    def publish_joint_state(self, now):
        """
        obtain joint state from Mujoco and publish
        """
        # publish joint state
        position, velocitiy = self.get_joint_state(self.joint_names)
        msg = JointState()
        msg.name = self.joint_names
        msg.position = list(position)
        msg.velocity = list(velocitiy)
        msg.header.stamp = rospy.Time.now()
        self.all_pub.publish(msg)


if __name__ == "__main__":
    # Instantiate and run the node
    rospy.init_node("execution_node")
    ur5e_node = Ur5eNode(args.server_address)
    if args.scene:
        ur5e_node.reset(args.scene, args.gui, args.save_image_dir)
    ur5e_node.run()
