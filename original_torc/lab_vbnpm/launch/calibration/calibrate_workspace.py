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
from trajectory_msgs.msg import JointTrajectory
from moveit_msgs.msg import MoveGroupActionResult
from sensor_msgs.msg import PointCloud2, JointState
from visualization_msgs.msg import InteractiveMarkerUpdate
from geometry_msgs.msg import Point, TransformStamped, PoseStamped, Pose
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
from task_planner.eutils import execute


rospy.init_node("planning")

goal_traj = JointTrajectory()
bottom_left = Pose()
top_right = Pose()

rospy.sleep(1)
tf_buffer = tf2_ros.Buffer(rospy.Duration(60))
tf_listen = tf2_ros.TransformListener(tf_buffer)


def get_transform(target_frame, source_frame):
    if target_frame == source_frame:
        return None
    try:
        in2out = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rospy.Time(),
            rospy.Duration(1.0),
        )
        return in2out
    except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
    ):
        print(
            f"""
Warning: Couldn't get transform from [{source_frame}] to [{target_frame}]!"
Returning grasps in frame [{source_frame}]
              """
        )
        return None


def move_group_callback(result_msg):
    global goal_traj
    goal_traj = result_msg.result.planned_trajectory.joint_trajectory


def bottom_left_callback(pose_stamped):
    global bottom_left
    bottom_left = tf2_geometry_msgs.do_transform_pose(
        pose_stamped,
        get_transform('world', pose_stamped.header.frame_id),
    ).pose


def top_right_callback(pose_stamped):
    global top_right
    top_right = tf2_geometry_msgs.do_transform_pose(
        pose_stamped,
        get_transform('world', pose_stamped.header.frame_id),
    ).pose


rospy.Subscriber(
    '/move_group/result', MoveGroupActionResult, move_group_callback
)
rospy.Subscriber('/aruco_single_1/pose', PoseStamped, bottom_left_callback)
rospy.Subscriber('/aruco_single_2/pose', PoseStamped, top_right_callback)

rp = RosPack()
robot_urdf = rp.get_path(
    'motoman_sda10f_moveit_config'
) + '/config/gazebo_motoman_sda10f.urdf'
planner = MoveitPlanner(
    robot_urdf,
    ['motoman_left_ee', 'motoman_right_ee'],
    ['arm_right', 'arm_left', 'arms'],
    is_sim=False,
)
planner.scene.remove_world_object()
planner.scene.add_box(
    'safety',
    conv.list_to_pose_stamped([0.97, 0.0, 0, 0, 0, 0], 'world'),
    (0.4, 10, 10),
)

joint_state = rospy.wait_for_message('/joint_states_all', JointState, timeout=5)
goal = {
    'arm_right_joint_1_s': 0.3,
    'arm_right_joint_2_l': 1.6,
    'arm_right_joint_3_e': 2.75,
    'arm_right_joint_4_u': 0.6,
    'arm_right_joint_5_r': -0.5,
    'arm_right_joint_6_b': -1.65,
    'arm_right_joint_7_t': 0,
    'torso_joint_b1': 1.09,
}
plan = planner.joint_motion_plan(joint_state, goal, 'arm_right')
input("Execute 1?")
execute(plan, wait=True)
while input("Detected?") not in ('y', 'Y', 't', '1'):
    execute(goal_traj, wait=True)

joint_state = rospy.wait_for_message('/joint_states_all', JointState, timeout=5)
goal = {
    'arm_right_joint_1_s': 1.9,
    'arm_right_joint_2_l': -0.47,
    'arm_right_joint_3_e': 1.18,
    'arm_right_joint_4_u': 1.6,
    'arm_right_joint_5_r': 0,
    'arm_right_joint_6_b': 0,
    'arm_right_joint_7_t': -2.7,
    'torso_joint_b1': -0.24,
}
plan = planner.joint_motion_plan(joint_state, goal, 'arm_right')
input("Execute 2?")
execute(plan, wait=True)
while input("Detected?") not in ('y', 'Y', 't', '1'):
    execute(goal_traj, wait=True)

bl = np.array(
    (
        bottom_left.position.x,
        bottom_left.position.y - 0.02,  # subtract half of the marker width
        # bottom_left.position.z - 0.02,  # subtract half of the marker height
        bottom_left.position.z - 0.2,
    )
)
tr = np.array(
    (
        top_right.position.x + 0.5,  # add depth of shelf
        top_right.position.y + 0.02,  # add half of the marker width
        # top_right.position.z + 0.06,  # add half of the marker height plus extra
        top_right.position.z + 0.18,
    )
)
size = np.abs(tr - bl)
print(bl, size)

with open('set_workspace_params.py', 'w') as f:
    contents = f"""import rospy
pose, size = (
    {bl.tolist()},
    {size.tolist()},
)
rospy.set_param('/workspace/pose', pose)
rospy.set_param('/workspace/size', size)
rospy.set_param('/robot/vel_ang_lim', 20)
rospy.set_param('/robot/acc_ang_lim', 850)"""
    print(contents, file=f)
