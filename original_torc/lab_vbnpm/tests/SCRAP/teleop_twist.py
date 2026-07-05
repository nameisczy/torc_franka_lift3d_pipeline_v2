import os
import sys
import copy
import pickle
import trimesh
import numpy as np
import transformations as tf

import rospy
import message_filters
from rospkg import RosPack
from moveit_commander import conversions as conv

from sensor_msgs.msg import JointState
from actionlib_msgs.msg import GoalStatusArray, GoalID
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Twist, Point, TransformStamped, PoseStamped, Pose

from lab_vbnpm.srv import EEControl, EEControlResponse
from lab_vbnpm.srv import ExecuteTrajectory, ExecuteTrajectoryResponse
from lab_vbnpm.srv import GetScenePointCloud, GetScenePointCloudResponse

from utils import conversions as conv2
from motion_planner.moveit_planner import MoveitPlanner


def execute(trajectory):
    print('execute_trajectory:')
    rospy.wait_for_service('execute_trajectory', timeout=5)
    try:
        execute_trajectory = rospy.ServiceProxy(
            'execute_trajectory', ExecuteTrajectory
        )
        resp = execute_trajectory(trajectory, True)
        print('Done!')
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def close():
    print('ee_control: close')
    rospy.wait_for_service('ee_control', timeout=60)
    try:
        ee_control = rospy.ServiceProxy('ee_control', EEControl)
        resp = ee_control('robotiq', 0.02)
        print('Done!')
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def open():
    print('ee_control: open')
    rospy.wait_for_service('ee_control', timeout=60)
    try:
        ee_control = rospy.ServiceProxy('ee_control', EEControl)
        resp = ee_control('robotiq', 0.085)
        print('Done!')
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


rospy.init_node("planning")
cancel_pub = rospy.Publisher(
    '/joint_trajectory_action/cancel', GoalID, queue_size=1
)
is_sim = True if len(sys.argv) > 1 else False
## instatiate planner ##
rp = RosPack()
robot_urdf = rp.get_path(
    'motoman_sda10f_moveit_config'
) + '/config/gazebo_motoman_sda10f.urdf'
planner = MoveitPlanner(
    robot_urdf, ['motoman_left_ee', 'motoman_right_ee'],
    ['arm_right', 'arm_left'],
    is_sim=is_sim
)

ik = planner.ik_for_ees['motoman_left_ee']
mg = planner.move_groups['arm_left']


def move_joint_twist(joint_msg, twist_msg):
    joint_indices = list(map(joint_msg.name.index, ik.joint_names))
    start_joint_values = np.array(joint_msg.position)[joint_indices]
    start_pose = ik.fk(start_joint_values)

    start_xyz = tf.translation_from_matrix(start_pose)
    start_ang = tf.euler_from_matrix(start_pose)
    velocity = np.array(
        [twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z]
    )
    speed = np.linalg.norm(velocity)
    if speed == 0:
        print('Speed is zero!')
        return
    rospy.set_param(
        'execution_node/trajectory_timestep', min(0.5, float(speed))
    )
    goal_xyz = start_xyz + 10 * velocity / speed
    goal_ang = start_ang + np.array(
        [twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z]
    )

    goal_pose = tf.euler_matrix(*goal_ang)
    goal_pose[:3, 3] = goal_xyz

    pose_msg = conv2.matrix_to_pose(goal_pose)
    waypoints = [copy.deepcopy(pose_msg)]

    r_start_state = planner.make_robot_state_msg(
        joint_msg, [], None, None, [], False
    )
    mg.set_start_state(r_start_state)
    robot_trajectory, fraction = mg.compute_cartesian_path(
        waypoints, 0.002, 5, avoid_collisions=True
    )
    if len(robot_trajectory.joint_trajectory.points) > 1:
        last = robot_trajectory.joint_trajectory.points[-1]
        goal_joint_values = last.positions
        g_time = last.time_from_start.to_sec()
        dist = np.linalg.norm(goal_joint_values - start_joint_values)
        print(dist, speed, g_time, speed * g_time / dist)
        retimed = mg.retime_trajectory(
            r_start_state, robot_trajectory, speed * g_time / dist
        )
        joint_trajectory = retimed.joint_trajectory
        # joint_trajectory.points = joint_trajectory.points[1:2]
        print('joint_trajectory', len(joint_trajectory.points))
        execute(joint_trajectory)
        return speed
    else:
        print('Hit IK barrier!')

    # goal_joint_values = ik.ik(goal_pose, start_joint_values)
    # dist = np.linalg.norm(goal_joint_values - start_joint_values)
    # print(start_joint_values)
    # print(goal_joint_values)
    # print(start_pose)
    # print(goal_pose)
    # print(dist)
    # print(dt)
    # if goal_joint_values is None:
    #     print('No IK solution found!')
    #     return

    # trajectory = JointTrajectory()
    # trajectory.header = joint_msg.header
    # trajectory.joint_names = ik.joint_names
    # zero = [0.0] * ik.number_of_joints
    # trajectory.points = [
    #     JointTrajectoryPoint(
    #         positions=start_joint_values,
    #         velocities=zero,
    #         accelerations=zero,
    #         time_from_start=rospy.Duration(0)
    #     ),
    #     JointTrajectoryPoint(
    #         positions=goal_joint_values,
    #         velocities=zero,
    #         accelerations=zero,
    #         time_from_start=rospy.Duration(dt)
    #     )
    # ]
    # execute(trajectory)


new_msg = None
old_msg = None


def twist_callback(twist_msg):
    global new_msg
    new_msg = twist_msg


rospy.set_param('execution_node/trajectory_timestep', 0.1)
twist_sub = rospy.Subscriber('/cmd_vel', Twist, twist_callback, queue_size=1)

rate = rospy.Rate(4)
while not rospy.is_shutdown():
    if old_msg != new_msg:
        if old_msg is not None:
            cancel_pub.publish(GoalID())
            rate.sleep()
        joint_state_msg = rospy.wait_for_message(
            '/joint_states_all', JointState, timeout=5
        )
        move_joint_twist(joint_state_msg, new_msg)
        old_msg = new_msg
    rate.sleep()
