import os
import sys
import copy
import pickle
import trimesh
import numpy as np

import rospy
import tf2_ros
import tf2_geometry_msgs
from rospkg import RosPack
from moveit_commander import conversions as conv

np.float = float
import ros_numpy as rnp

from std_msgs.msg import Int64
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointCloud2, JointState
from actionlib_msgs.msg import GoalStatusArray, GoalID
from geometry_msgs.msg import Point, TransformStamped, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from gpd_ros.msg import CloudSources, CloudIndexed, CloudSamples
# from gpd_ros.srv import detect_grasps, detect_graspsRequest, detect_graspsResponse

from lab_vbnpm.msg import ObjectPoses
from lab_vbnpm.srv import EEControl, EEControlResponse
from lab_vbnpm.srv import ExecuteTrajectory, ExecuteTrajectoryResponse
from lab_vbnpm.srv import GetScenePointCloud, GetScenePointCloudResponse

from utils import conversions as conv2
from utils.visual_utils import decode_seg_img_rgb
from grasp_planner.grasp_planner import GraspPlanner
from motion_planner.moveit_planner import MoveitPlanner
from perception.perception_interface import PerceptionInterface

from motoman_msgs.msg import DynamicJointTrajectory, DynamicJointPoint, DynamicJointsGroup


def execute(trajectory):
    print('execute_trajectory:')
    rospy.wait_for_service('execute_trajectory', timeout=5)
    try:
        execute_trajectory = rospy.ServiceProxy(
            'execute_trajectory', ExecuteTrajectory
        )
        resp = execute_trajectory(trajectory, rospy.Duration(0.2))
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


def send_point_to_robot(plan):
    dyn_traj = DynamicJointTrajectory()
    dyn_traj.header.stamp = rospy.Time.now()
    dyn_traj.joint_names = plan.joint_names
    dyn_traj.points = [
        DynamicJointPoint(
            num_groups=4,
            groups=[
                DynamicJointsGroup(
                    group_number=0,
                    num_joints=7,
                    valid_fields=0,
                    positions=p.positions[:7],
                    velocities=p.velocities[:7],
                    accelerations=[0.0] * 7,
                    effort=[0.0] * 7,
                    time_from_start=p.time_from_start,
                ),
                DynamicJointsGroup(
                    group_number=1,
                    num_joints=7,
                    valid_fields=0,
                    positions=p.positions[7:14],
                    velocities=p.velocities[7:14],
                    accelerations=[0.0] * 7,
                    effort=[0.0] * 7,
                    time_from_start=p.time_from_start,
                ),
                DynamicJointsGroup(
                    group_number=2,
                    num_joints=1,
                    valid_fields=0,
                    positions=p.positions[15:16],
                    velocities=p.velocities[15:16],
                    accelerations=[0.0],
                    effort=[0.0],
                    time_from_start=p.time_from_start,
                ),
                DynamicJointsGroup(
                    group_number=3,
                    num_joints=1,
                    valid_fields=0,
                    positions=p.positions[15:16],
                    velocities=p.velocities[15:16],
                    accelerations=[0.0],
                    effort=[0.0],
                    time_from_start=p.time_from_start,
                ),
            ]
        ) for p in plan.points
    ]
    print(dyn_traj)
    joint_cmd_pub.publish(dyn_traj)


rospy.init_node("planning")
cancel_pub = rospy.Publisher(
    '/joint_trajectory_action/cancel', GoalID, queue_size=1
)
joint_cmd_pub = rospy.Publisher(
    '/joint_command', DynamicJointTrajectory, queue_size=1
)

is_sim = True if len(sys.argv) > 1 else False
## instatiate planner ##
rp = RosPack()
robot_urdf = rp.get_path(
    'motoman_sda10f_moveit_config'
) + '/config/gazebo_motoman_sda10f.urdf'
planner = MoveitPlanner(
    None,
    robot_urdf, ['motoman_left_ee', 'motoman_right_ee'],
    ['arm_right', 'arm_left', 'arms'],
    is_sim=is_sim
)

joint_state = rospy.wait_for_message('/joint_states_all', JointState, timeout=5)

if is_sim:
    start = {
        "arm_left_joint_1_s": 1.75,
        "arm_left_joint_2_l": 0.8,
        "arm_left_joint_3_e": 0,
        "arm_left_joint_4_u": -0.66,
        "arm_left_joint_5_r": 0,
        "arm_left_joint_6_b": 0,
        "arm_left_joint_7_t": 0,
        "arm_right_joint_1_s": 0.8227,
        "arm_right_joint_2_l": 0.7819,
        "arm_right_joint_3_e": -1.9520,
        "arm_right_joint_4_u": -0.5853,
        "arm_right_joint_5_r": -1.4359,
        "arm_right_joint_6_b": -1.6822,
        "arm_right_joint_7_t": 0.0,
        "torso_joint_b1": 0,
    }
else:
    start = {
        "arm_left_joint_1_s": 1.75,
        "arm_left_joint_2_l": 0.8,
        "arm_left_joint_3_e": 0,
        "arm_left_joint_4_u": -0.66,
        "arm_left_joint_5_r": 0,
        "arm_left_joint_6_b": 0,
        "arm_left_joint_7_t": 0,
        "arm_right_joint_1_s": 1.2102,
        "arm_right_joint_2_l": -0.0540,
        "arm_right_joint_3_e": 2.3140,
        "arm_right_joint_4_u": 1.4975,
        "arm_right_joint_5_r": -2.7925,
        "arm_right_joint_6_b": -0.9223,
        "arm_right_joint_7_t": 0,
        "torso_joint_b1": 0,
    }

goal = copy.deepcopy(start)
goal["arm_left_joint_6_b"] = -1.57
goal["arm_left_joint_7_t"] = 1.57

plan0 = planner.joint_motion_plan(joint_state, goal, 'arms')
# joint_state1 = JointState(name=goal.keys(), position=goal.values())
n = len(plan0.points)
mid = plan0.points[n // 2]

input("Execute?")
execute(plan0)
input("Continue?")
i = 0
plan0.joint_names = plan0.joint_names[1:] + ['torso_joint_b1', 'torso_joint_b2']
while i < len(plan0.points):
    point = plan0.points[i]
    point.positions = point.positions[1:] + point.positions[:1] + (0.0, )
    point.velocities = point.velocities[1:] + point.velocities[:1] + (0.0, )
    point.accelerations = point.accelerations[1:] + point.accelerations[:1] + (
        0.0,
    )
    print(point)
    send_point_to_robot(
        JointTrajectory(
            joint_names=plan0.joint_names,
            points=[point],
        )
    )
    # rospy.sleep(0.1)
    i += 1
print(i)

input("Next?")
start["torso_joint_b2"] = 0
plan1 = JointTrajectory(
    joint_names=list(start.keys()),
    points=[
        JointTrajectoryPoint(
            positions=list(start.values()),
            velocities=[0.0] * len(start),
            time_from_start=rospy.Duration(15.0),
        ),
    ]
)
send_point_to_robot(
    JointTrajectory(
        joint_names=plan1.joint_names,
        points=plan1.points,
    )
)
