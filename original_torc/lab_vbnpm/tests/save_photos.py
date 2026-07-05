import sys
import trimesh
import numpy as np

import rospy
import tf2_ros
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from utils.visual_utils import from_color_map
from task_planner.motoman import MotomanSDA10F
from task_planner.eutils import ee_open, ee_close, execute
from task_planner.curobo_open_loop import update_and_get_points

rospy.init_node('save_photos')
rospy.sleep(1)
is_sim = sys.argv[1][0] not in (
    '0',
    'r',
    'R',
    'n',
    'N',
) if len(sys.argv) > 1 else True
gt = sys.argv[2][0] not in (
    '0',
    'r',
    'R',
    'n',
    'N',
) if len(sys.argv) > 2 else True
robot = MotomanSDA10F(is_sim, gt)
perception = robot.init_perception_interface()

i = 0
while True:
    result = update_and_get_points(
        "red yogurt.",
        robot,
        perception,
        camera_inds=[0],
        save_debug='video',
        debug_number=i,
    )
    joint_state = rospy.wait_for_message('/joint_states_all', JointState)
    if i == 0:
        fname = f'/tmp/recording_0000/joints_0.csv'
    else:
        fname = f'/tmp/recording_0000/poses/joints_{i}.csv'
    with open(fname, 'w') as f:
        for name, pos in zip(joint_state.name, joint_state.position):
            f.write(f'{name},{pos}\n')

    pts, rgb, obj_mask, all_pts, all_rgb, all_mask = result
    trimesh.points.PointCloud(all_pts, all_rgb).show()
    if input("Took photo " + str(i) + ", continue? (Y/n)") == 'n':
        break
    i += 1

# unique = list(sorted(set(obj_mask)))
# color_num = list(map(unique.index, obj_mask))
# colors = from_color_map(color_num, 32)
# trimesh.points.PointCloud(pts, colors).show()

# unique = list(sorted(set(all_mask)))
# color_num = list(map(unique.index, all_mask))
# colors = from_color_map(color_num, 32)
# trimesh.points.PointCloud(all_pts, colors).show()
# exit(0)
# input('Continue?')

# num = int(sys.argv[1]) if len(sys.argv) > 1 else 0
# if num > 0:
#     planner = robot.init_motion_planner(planner='curobo')
#     from grasp_planner.curobo_grasp_planner import GraspPlanner
#     grasp_planner = GraspPlanner(
#         robot.curobo_config,
#         planner.static_world_config,
#         robot.urdf,
#         ignore_collision_ee_links=robot.ignore_collision_ee_links,
#     )
#     from task_planner.curobo_active_grasp import active_grasp_views
#     tgt_pts = np.load('test_pts.npy')
#     scores, views, poses, min_max = active_grasp_views(
#         "mustard bottle.",
#         robot,
#         perception,
#         planner,
#         grasp_planner,
#         tgt_pts,
#         downsample=10
#     )
#     for i in range(min(len(views), num)):
#         print(f"view {i} score {scores[i]}")
#         joint_state = rospy.wait_for_message(
#             '/joint_states_all', JointState, timeout=5
#         )
#         plan = planner.pose_motion_plan(
#             joint_state,
#             poses[i],
#         )
#         input("Execute?")
#         execute(JointTrajectory())
#         execute(plan, window=0.1, retime=True)
#         input("Save Photo?")
#         update_and_get_points(
#             "mustard bottle.",
#             robot,
#             perception,
#             camera_inds=[1],
#             save_debug='video',
#             debug_number=i + 1,
#         )
