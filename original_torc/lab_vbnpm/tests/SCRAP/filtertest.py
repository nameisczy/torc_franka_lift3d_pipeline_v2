import os
import sys
import copy
import time
import pickle
import trimesh
import numpy as np
import open3d as o3d
import transformations as tf

import rospy
import tf2_ros
import tf2_geometry_msgs
from rospkg import RosPack
from moveit_commander import conversions as conv

np.float = float
import ros_numpy as rnp

from std_msgs.msg import Int64, String
from sensor_msgs import point_cloud2 as pc2
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import PointCloud2, JointState
from geometry_msgs.msg import Point, TransformStamped, PoseStamped

from lab_vbnpm.srv import EEControl, EEControlResponse
from lab_vbnpm.srv import ExecuteTrajectory, ExecuteTrajectoryResponse
from segment3d.srv import GetDeticResults, GetDeticResultsRequest, GetDeticResultsResponse

from utils import conversions as conv2
from grasp_planner.grasp_planner import GraspPlanner
from motion_planner.moveit_planner import MoveitPlanner
# from perception.perception_interface import PerceptionInterface
from perception.perception_fast import PerceptionInterface


def execute(trajectory):
    print('execute_trajectory:')
    rospy.wait_for_service('execute_trajectory', timeout=5)
    try:
        execute_trajectory = rospy.ServiceProxy(
            'execute_trajectory', ExecuteTrajectory
        )
        resp = execute_trajectory(trajectory, False)
        print('Done!')
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def close():
    print('ee_control: close')
    rospy.wait_for_service('ee_control', timeout=60)
    try:
        ee_control = rospy.ServiceProxy('ee_control', EEControl)
        resp = ee_control('robotiq', 0.0)
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


pre_grasp_dist = 0.09
lift_height = 0.03
ee_links = [
    "motoman_right_ee",
    "left_outer_knuckle",
    "left_outer_finger",
    "left_inner_finger",
    "left_inner_finger_pad",
    "left_inner_knuckle",
    "right_outer_knuckle",
    "right_outer_finger",
    "right_inner_finger",
    "right_inner_finger_pad",
    "right_inner_knuckle",
    "robotiq_arg2f_extra_link",
    "robotiq_arg2f_base_link",
    "arm_right_link_7_t",
    "arm_right_link_6_b",
]

t0 = time.time()

rospy.init_node("planning")
is_sim = sys.argv[1][0] not in (
    '0',
    'r',
    'R',
    'n',
) if len(sys.argv) > 1 else True

## instatiate planner ##
rp = RosPack()
robot_urdf = rp.get_path(
    'motoman_sda10f_moveit_config'
) + '/config/gazebo_motoman_sda10f.urdf'
planner = MoveitPlanner(
    None,
    robot_urdf,
    ['motoman_left_ee', 'motoman_right_ee'],
    ['arm_right', 'arm_left'],
    is_sim=is_sim,
)

## instatiate grasp planner ##
grasp_planner = GraspPlanner()

if is_sim:
    # init simulated perception interface
    camera = ['camera0', 'camera1']
    perception = PerceptionInterface(
        camera,
        pose_x=0.9 - 0.29,
        pose_y=0 - 0.5,
        pose_z=0.90,
        size_x=0.58,
        size_y=1.0,
        size_z=0.5,
        resolution=0.005,
    )
else:
    # init real perception interface
    camera = ['d455', 'd435']
    pose_x, pose_y, pose_z = rospy.get_param('/workspace/pose', [0.9, 0.8, 1.0])
    size_x, size_y, size_z = rospy.get_param('/workspace/size', [0.5, 1.5, 0.5])
    offsetxF = 0.02
    offsetxB = 0.04
    offsetyR = 0.17
    offsetyL = 0.26
    offsetz = 0.01
    perception = PerceptionInterface(
        camera,
        pose_x - offsetxF,
        pose_y - size_y + offsetyR,
        pose_z - 1 * offsetz,
        size_x + offsetxF + offsetxB,
        size_y - offsetyR - offsetyL,
        size_z - 0 * offsetz,
        resolution=0.005,
    )
t1 = time.time()
print('Init Time:', t1 - t0)

## select object from argument ##
object_to_grasp = sys.argv[2] if len(sys.argv) > 2 else 'tomato_soup_can'

print('Staring!')
t00 = time.time()

## sense the scene ##
t0 = time.time()
# points, colors, target_mask, bg_mask = perception.get_visible_points(
#     camera[0],
#     object_to_grasp,
# )
perception.updated_fused_points(camera[0], object_to_grasp, filter_robot=False)
perception.updated_fused_points(camera[1], object_to_grasp, filter_robot=False)
input('Continue?')
perception.updated_fused_points(camera[0], object_to_grasp, filter_robot=False)
perception.updated_fused_points(camera[1], object_to_grasp, filter_robot=False)
points, colors = perception.get_fused_point_cloud()
# target_points, target_colors = perception.get_fused_target_point_cloud()
bg_points, bg_colors = perception.get_fused_bg_point_cloud()

t2 = time.time()
perception.save_fusion('pc_data', object_to_grasp)
t3 = time.time()

perception.update_occlusion(camera[0])
perception.update_occlusion(camera[1])
occluded_points = perception.get_occlusion_points()
occluded_colors = np.zeros((*occluded_points.shape[:-1], 3))
occluded_colors[:] = [1, 0, 1]
# vx_occluded, vx_resolution, vx_pose, vx_frame_id = perception.get_occlusion_voxels()
t4 = time.time()

# filter combined point cloud
all_points = np.concatenate([points, occluded_points])
all_colors = np.concatenate([colors, occluded_colors])

pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(all_points)
pcl.colors = o3d.utility.Vector3dVector(all_colors)
# pcl, ind = pcl.remove_radius_outlier(10, 0.004)
pcl, ind = pcl.remove_radius_outlier(10, 0.008)
pcl, ind = pcl.remove_statistical_outlier(20, 3)
all_points = np.array(pcl.points)
all_colors = np.array(pcl.colors)
t5 = time.time()

target_points, target_colors = perception.get_largest_target_cluster()
target_mesh, t_pcd = perception.get_symmetric_shape(
    target_points,
    target_colors,
    # True,
)
# target_points = np.array(t_pcd.vertices)
# target_colors = np.array(t_pcd.colors[:, :3]) / 255.0
t6 = time.time()

# get filtered surface points
# p_t = set([tuple(p) for p in points])
# all_p_t = set([tuple(p) for p in all_points])
# surface_points = np.array(list(p_t.intersection(all_p_t)))

t1 = time.time()
print('Fusion Time:', t2 - t0)
print('Occlusion Time:', t4 - t3)
print('Filtering Time:', t5 - t4)
print('Target Shape Estimation:', t6 - t5)
print('Total Perception Time:', (t1 - t0) - (t3 - t2))

trimesh.points.PointCloud(bg_points, bg_colors).show()
trimesh.points.PointCloud(target_points, target_colors).show()
trimesh.scene.Scene([target_mesh, t_pcd]).show()
trimesh.points.PointCloud(all_points, all_colors).show()
# trimesh.points.PointCloud(surface_points).show()
