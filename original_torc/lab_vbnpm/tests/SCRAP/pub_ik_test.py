import sys
import time
import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from moveit_commander import conversions as conv
from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory, RobotState

from utils import conversions as conv2
from grasp_planner.grasp_planner import GraspPlanner
from task_planner.eutils import ee_open, ee_close, execute

is_sim = sys.argv[1][0] not in (
    '0',
    'r',
    'R',
    'n',
    'N',
) if len(sys.argv) > 1 else True
place = [0.6, 0, 1.1, 0.5, -0.5, -0.5, -0.5]
# place = [0.6, 0, 1.1]

## init perception and planning interfaces ##
rospy.init_node("planning")
t0 = time.time()
from task_planner.motoman import MotomanSDA10F

robot = MotomanSDA10F(is_sim)
planner = robot.init_motion_planner()
planner.reset(update_moveit=False)
# planner.set_planning_scene([], update_moveit=False, visualize=True)
t1 = time.time()
print('Init Time:', t1 - t0)

speed = 20 * np.pi / 180

joint_state = rospy.wait_for_message('/joint_states_all', JointState, timeout=5)
total_dur = 0
for i in range(50):
    plan = planner.iter_ik_motion_plan(
        joint_state,
        place,
        robot.gripper_group,
        score=10,
        num_iters=1,
        time_step=1.0,
        # look_link=robot.camera_link,
        # lookat_point=np.mean(tgt_pts, axis=0),
        # lookat_axis=robot.camera_axis,
        ee=robot.gripper_link,
        is_diff=False,
    )
    print('Plan A:', len(plan.points))
    plan.points.pop(0)
    print('Plan B:', len(plan.points))

    # set duration (and velocity?)
    inds = list(map(plan.joint_names.index, joint_state.name))
    vals = np.array(plan.points[-1].positions)[inds]
    disp = vals - joint_state.position
    # dist = np.linalg.norm(disp)
    dist = max(np.abs(disp))

    duration = dist / speed
    if duration < 0.1:
        print('Done:', i)
        break
    total_dur += duration
    plan.points[-1].time_from_start = rospy.Duration(total_dur)
    plan.points[-1].velocities = tuple(disp / duration)
    if i % 10 == 0:
        input('Continue?')
    print('Dur, total', duration, total_dur)
    is_stream_new = execute(plan, window=duration, wait=False)
    if is_stream_new == 1:
        total_dur = 0

    # update joint state for next iteration
    joint_state.position = vals
