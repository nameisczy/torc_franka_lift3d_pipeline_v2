import os
import sys
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from industrial_msgs.msg import RobotStatus

from lab_vbnpm.srv import EEControl
from lab_vbnpm.srv import ExecuteTrajectory, ExecuteTrajectoryRequest
from lab_vbnpm.srv import ExperimentResult, ExperimentResultRequest

from utils.conversions import float_to_ros_duration


def get_ee_command_open_close(opening: bool):
    robot_type = os.environ.get("TORC_ROBOT", "motoman").strip().lower()
    if robot_type in ("franka", "panda"):
        return "panda", 0.04 if opening else 0.0
    return "robotiq", 0.085 if opening else 0.0


def wait_till(stop, steps=8, max_steps=500):
    c = 0
    total_steps = 0
    while c < steps and total_steps < max_steps:
        robot_status = rospy.wait_for_message("/robot_status", RobotStatus)
        moving = robot_status.in_motion.val == 1

        if moving != stop:
            c += 1
        else:
            c = 0
        total_steps += 1
    if total_steps >= max_steps:
        print(f"wait_till hit max_steps: {max_steps}!")


def wait_till_gripper(steps=8, max_steps=500):
    joint_name = "finger_joint"

    c = 0
    total_steps = 0
    prev_pos = 0
    while c < steps and total_steps < max_steps:
        msg = rospy.wait_for_message("/joint_states", JointState)
        while joint_name not in msg.name:
            msg = rospy.wait_for_message("/joint_states", JointState)
        idx = msg.name.index(joint_name)
        position = msg.position[idx]

        if np.allclose(position, prev_pos, atol=0.001):
            c += 1
        else:
            c = 0
        prev_pos = position
        total_steps += 1
    if total_steps >= max_steps:
        print(f"wait_till_gripper hit max_steps: {max_steps}")


def execute(trajectory, window=0.1, retime=True, stream=False, wait=False):
    # print('execute_trajectory:')
    # optionally wait for robot to stop
    if wait:
        print("Waiting for robot to stop...")
        wait_till(stop=True)
        print("Stopped!")
        joint_state = rospy.wait_for_message("/joint_states_all", JointState, timeout=5)
        inds = list(map(joint_state.name.index, trajectory.joint_names))
        positions = np.array(joint_state.position)[inds]
        trajectory.points[0].positions = positions

    # figure out mode and set execution window
    if stream:
        mode = ExecuteTrajectoryRequest.STREAM
    elif window == 0:
        mode = ExecuteTrajectoryRequest.SYNCHRONOUS
    else:
        mode = ExecuteTrajectoryRequest.INTERUPT
    window = float_to_ros_duration(window)

    # create service request to execute
    rospy.wait_for_service("execute_trajectory", timeout=5)
    try:
        execute_trajectory = rospy.ServiceProxy("execute_trajectory", ExecuteTrajectory)
        resp = execute_trajectory(trajectory, window, retime, mode)
        print("Moving!")
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}", file=sys.stderr)
        return -11, []

    # optionally wait for robot to stop
    if wait:
        # first wait for robot to move before waiting for it to stop
        # wait_till(stop=False)
        print("Waiting for robot to stop...")
        wait_till(stop=True)
        print("Stopped!")
    return resp.error_code, resp.interrupt_points


def ee_close(wait=True):
    # print('ee_control: close')
    if wait:
        print("Waiting for robot to stop...")
        wait_till(stop=True)
        print("Stopped!")
    rospy.wait_for_service("ee_control", timeout=5)
    try:
        ee_control = rospy.ServiceProxy("ee_control", EEControl)
        gripper_name, width = get_ee_command_open_close(False)
        resp = ee_control(gripper_name, width)
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}", file=sys.stderr)

    if wait:
        wait_till_gripper()
        print("Closed!")


def ee_open(wait=True):
    # print('ee_control: open')
    if wait:
        print("Waiting for robot to stop...")
        wait_till(stop=True)
        print("Stopped!")
    rospy.wait_for_service("ee_control", timeout=5)
    try:
        ee_control = rospy.ServiceProxy("ee_control", EEControl)
        gripper_name, width = get_ee_command_open_close(True)
        resp = ee_control(gripper_name, width)
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}", file=sys.stderr)

    if wait:
        wait_till_gripper()
        print("Opened!")


def ee_suction_on():
    rospy.wait_for_service("ee_control", timeout=5)
    try:
        ee_control = rospy.ServiceProxy("ee_control", EEControl)
        resp = ee_control("onrobot_vgc10", 1.0)
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}", file=sys.stderr)


def ee_suction_off():
    rospy.wait_for_service("ee_control", timeout=5)
    try:
        ee_control = rospy.ServiceProxy("ee_control", EEControl)
        resp = ee_control("onrobot_vgc10", 0.0)
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}", file=sys.stderr)


def get_experiment_result(target, sim=True):
    if sim:
        rospy.wait_for_service("/experiment_result")
        try:
            experiment_result = rospy.ServiceProxy(
                "/experiment_result", ExperimentResult
            )
            resp = experiment_result(str(target))
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}", file=sys.stderr)
            return False, [], []
        return resp.success, resp.grasping, resp.dropped
    else:
        grasping = input(
            f"""Is the robot grasping "{target}"?
Press enter if yes, otherwise list grasped objects.
If no objects are grasped write "n"."""
        )
        dropped = input("List dropped objects (press enter if none)?\n")

        if grasping == "":
            grasping = [str(target)]
        elif grasping == "n":
            grasping = []
        else:
            grasping = grasping.split(",")

        if dropped == "":
            dropped = []
        else:
            dropped = dropped.split(",")

        success = len(grasping) == 1 and grasping[0] == target

        return success, grasping, dropped
