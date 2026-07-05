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
from lab_vbnpm.srv import GetNextPlanningPoint, GetNextPlanningPointResponse

from utils import conversions as conv2
from utils.visual_utils import decode_seg_img_rgb
from grasp_planner.grasp_planner import GraspPlanner
from motion_planner.moveit_planner import MoveitPlanner
from perception.perception_interface import PerceptionInterface

from motoman_msgs.msg import DynamicJointTrajectory, DynamicJointPoint, DynamicJointsGroup

WINDOW = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2


def execute(trajectory):
    print('execute_trajectory:')
    rospy.wait_for_service('execute_trajectory', timeout=5)
    try:
        execute_trajectory = rospy.ServiceProxy(
            'execute_trajectory', ExecuteTrajectory
        )
        resp = execute_trajectory(trajectory, rospy.Duration(WINDOW))
        print('Done!')
        return resp
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return None


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

is_sim = True if len(sys.argv) > 2 else False
## instatiate planner ##
rp = RosPack()
robot_urdf = rp.get_path(
    'motoman_sda10f_moveit_config'
) + '/config/gazebo_motoman_sda10f.urdf'
planner = MoveitPlanner(
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

goal2 = copy.deepcopy(start)
goal2["arm_left_joint_5_r"] = -1.57 / 2
goal2["arm_left_joint_6_b"] = -1.57
goal2["arm_left_joint_7_t"] = 1.57

plan0 = planner.joint_motion_plan(joint_state, goal, 'arms')
input("Execute?")
resp = execute(plan0)
t = rospy.Time.now().to_sec()
if resp.error_code == 0:
    points = resp.interrupt_points
input("Continue?")

t1 = np.inf
planning_point = points.points.pop(0)
tft = planning_point.time_from_start
print('pop:', tft.to_sec())
while tft.to_sec() < 0.2 * WINDOW + (t1 - t):
    print('planning loop')
    try:
        t0 = rospy.Time.now().to_sec()
        while tft.to_sec() < 0.2 * WINDOW + (t0 - t):
            planning_point = points.points.pop(0)
            tft = planning_point.time_from_start
            print('pop:', tft.to_sec())
    except IndexError:
        plan1 = None
        break

    joint_state1 = JointState(
        name=points.joint_names,
        position=planning_point.positions,
        velocity=planning_point.velocities,
    )
    plan1 = planner.joint_motion_plan(joint_state1, goal2, 'arms')
    for i in range(len(plan1.points)):
        plan1.points[i].time_from_start += tft
    t1 = rospy.Time.now().to_sec()
    print('Planning Time:', (t1 - t0))

print('t1-t:', t1 - t)
print('ppt:', tft.to_sec())
print('ppt > t1-t:', tft.to_sec() > t1 - t)
while tft.to_sec() > WINDOW + (t1 - t):
    print('post loop', t1)
    t1 = rospy.Time.now().to_sec()

if plan1:
    execute(plan1)
else:
    print('Ran out of points.')
