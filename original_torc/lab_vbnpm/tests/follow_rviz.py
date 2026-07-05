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

# np.float = float
# import ros_numpy as rnp

from std_msgs.msg import Int64
from industrial_msgs.msg import RobotStatus
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, CameraInfo
from moveit_msgs.msg import MoveGroupActionResult
from sensor_msgs.msg import PointCloud2, JointState
from visualization_msgs.msg import InteractiveMarkerUpdate
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Point, TransformStamped, PoseStamped, Pose

# from gpd_ros.msg import CloudSources, CloudIndexed, CloudSamples
# from gpd_ros.srv import detect_grasps, detect_graspsRequest, detect_graspsResponse

from lab_vbnpm.msg import ObjectPoses
from lab_vbnpm.srv import EEControl, EEControlResponse
from lab_vbnpm.srv import ExecuteTrajectory, ExecuteTrajectoryResponse
from lab_vbnpm.srv import GetScenePointCloud, GetScenePointCloudResponse
from lab_vbnpm.srv import GetNextPlanningPoint, GetNextPlanningPointResponse

# from utils import conversions as conv2
# from utils.visual_utils import decode_seg_img_rgb
# from grasp_planner.grasp_planner import GraspPlanner
from motion_planner.moveit_planner import MoveitPlanner

# from perception.perception_interface import PerceptionInterface

from task_planner.eutils import ee_open, ee_close, execute

goal_pose = Pose()
plan_traj = JointTrajectory()


def marker_callback(update_msg):
    poses = update_msg.poses
    if len(poses) > 0:
        global goal_pose
        goal_pose = poses[0].pose
        # print(goal_pose)


rospy.init_node("planning")

rospy.Subscriber(
    "/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/update",
    InteractiveMarkerUpdate,
    marker_callback,
)

is_sim = True if len(sys.argv) > 1 and sys.argv[1][0] == "s" else False
group = "arm_left" if len(sys.argv) > 2 else "arm_right"

## instatiate planner ##
rp = RosPack()
robot_urdf = (
    rp.get_path("motoman_sda10f_moveit_config") + "/config/gazebo_motoman_sda10f.urdf"
)
planner = MoveitPlanner(
    robot_urdf,
    ["motoman_left_ee", "motoman_right_ee"],
    ["arm_right", "arm_left"],
    is_sim=is_sim,
)

t1 = None
prev_goal_pose = Pose()
while not rospy.is_shutdown():
    rospy.sleep(0.1)
    if goal_pose == prev_goal_pose:
        continue

    prev_goal_pose = copy.deepcopy(goal_pose)

    robot_status = rospy.wait_for_message("/robot_status", RobotStatus, timeout=5)
    if robot_status.in_motion.val == 0:
        joint_state = rospy.wait_for_message("/joint_states_all", JointState, timeout=5)
        plan = planner.pose_motion_plan(joint_state, prev_goal_pose, group)
        if type(plan) is JointTrajectory:
            execute(plan)
        t = rospy.Time.now().to_sec()
    else:
        error = 888
        while error != 0:
            print("error loop:", error)
            plan = None
            t0 = rospy.Time.now().to_sec()
            if t1 is None or tft.to_sec() < (t1 - t):
                rospy.wait_for_service("get_next_planning_point", timeout=5)
                try:
                    get_next_planning_point = rospy.ServiceProxy(
                        "get_next_planning_point", GetNextPlanningPoint
                    )
                    planning_point = get_next_planning_point()
                    tft = planning_point.point.time_from_start
                    if planning_point.point == JointTrajectoryPoint():
                        t1 = None
                        break
                    print("Next PP: Done!")
                except rospy.ServiceException as e:
                    print("Service call failed: %s" % e)
                    break
                t2 = rospy.Time.now().to_sec() - t0

            joint_state = JointState(
                name=planning_point.joint_names,
                position=planning_point.point.positions,
                velocity=planning_point.point.velocities,
            )
            plan = planner.pose_motion_plan(joint_state, prev_goal_pose, group)
            t1 = rospy.Time.now().to_sec()
            if type(plan) is not JointTrajectory:
                error = -88
                continue
            for i in range(len(plan.points)):
                plan.points[i].time_from_start += tft
            print("Planning Time:", (t1 - t0))
            print("Get Point Time:", t2)
            print("t1-t:", t1 - t)
            print("ppt:", tft.to_sec())
            print("ppt > t1-t:", tft.to_sec() > t1 - t)

            error, points = execute(plan)
            if error == 0:
                t1 = None
