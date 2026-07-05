import sys
import time
import numpy as np
from curobo.geom.types import PointCloud, Mesh

import rospy
from rospkg import RosPack
from sensor_msgs.msg import JointState

from lab_vbnpm.srv import ExecuteTrajectory
# from grasp_planner.grasp_planner import GraspPlanner
from grasp_planner.curobo_grasp_planner import GraspPlanner
# from motion_planner.moveit_planner import MoveitPlanner
from motion_planner.curobo_planner import CuroboPlanner
from task_planner.open_loop import update_and_get_points
from task_planner.eutils import ee_open, ee_close, execute

from utils import conversions as conv2

rospy.init_node("reset")

## parse args ##
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
object_to_grasp = sys.argv[3] if len(sys.argv) > 3 else 35

## instatiate perception ##
from task_planner.motoman import MotomanSDA10F

robot = MotomanSDA10F(is_sim, gt)
perception = robot.init_perception_interface()

## sense the environment ##
result = update_and_get_points(object_to_grasp, robot, perception)
points, tgt_pts, tgt_rgb, all_pts, all_rgb, ds_all_pts = result

print("Points: ", points.shape)
print("All points:", all_pts.shape)

occl_vol, color_vol = perception.get_occlusion_voxels()
tsdf_vol, color_vol, mask, seen = perception.tsdf_vol.get_volume()
print(tsdf_vol.shape)
# print(set(tsdf_vol.flatten()))
print(occl_vol.shape)
print(set(occl_vol.flatten()))

t0 = rospy.Time.now().to_sec()
mesh = Mesh.from_pointcloud(all_pts, 0.005 * 2)
t1 = rospy.Time.now().to_sec()
print("Mesh time:", t1 - t0)
# mesh.get_trimesh_mesh().show()

tgt_mesh = perception.get_shape_estimate(tgt_pts)

## instatiate planner ##
rp = RosPack()
robot_urdf = rp.get_path('motoman_sda10f_moveit_config')
robot_urdf += '/config/gazebo_motoman_sda10f.urdf'
curobo_config = rp.get_path('lab_vbnpm')
curobo_config += '/robots/motoman/curobo/motoman.yml'
planner = CuroboPlanner(
    robot_urdf,
    ['motoman_left_ee', 'motoman_right_ee'],
    curobo_config,
    [
        "arm_right_link_7_t",
        "robotiq_arg2f_base_link",
        "left_outer_finger",
        "right_outer_finger",
        "right_inner_finger",
        "left_inner_finger",
    ],
    is_sim=is_sim,
    warmup=False,
)
planner.visualize_rviz()

grasp_planner = GraspPlanner(curobo_config, planner.world_config, robot_urdf)

joint_state0 = rospy.wait_for_message(
    '/joint_states_all', JointState, timeout=5
)
planner.visualize_spheres_rviz(joint_state0)
input('next...')

## grasp planning ##
t0 = rospy.Time.now().to_sec()
result = grasp_planner.get_grasp_poses(
    tgt_pts,
    tgt_rgb,
    points,
    all_pts,
    rviz_func=planner.visualize_spheres_rviz,
)
grasps, pre_grasps, grasp_js, pre_grasps_js, scores, samples = result
t1 = rospy.Time.now().to_sec()
print("Grasps: ", len(grasps))
print("Grasp planning time:", t1 - t0)

## generate motion plan ##

# goal = {
#     "torso_joint_b1": 0,
#     "arm_left_joint_1_s": 1.75,
#     "arm_left_joint_2_l": 0.8,
#     "arm_left_joint_3_e": 0,
#     "arm_left_joint_4_u": -0.66,
#     "arm_left_joint_5_r": 0,
#     "arm_left_joint_6_b": 0,
#     "arm_left_joint_7_t": 0,
#     "arm_right_joint_1_s": 0.75,
#     "arm_right_joint_2_l": 0,
#     "arm_right_joint_3_e": -0.6,
#     "arm_right_joint_4_u": -1.15,
#     "arm_right_joint_5_r": 0,
#     "arm_right_joint_6_b": -1.3,
#     "arm_right_joint_7_t": 0.0,
# }
# plan = planner.joint_motion_plan(joint_state, goal, 'arms')

# plan grasp
t0 = rospy.Time.now().to_sec()
planner.set_planning_scene(all_pts)
planner.visualize_rviz()
t1 = rospy.Time.now().to_sec()
result = None
while result is None and len(scores) > 0:
    max_idx = np.argmax(scores)
    print("Score:", scores[max_idx])
    grasp = grasps[max_idx]

    # vis ik solution
    inds = list(map(joint_state0.name.index, planner.motion_gen.joint_names))
    vis_state = JointState()
    vis_state.name = joint_state0.name
    vis_state.position = np.array(joint_state0.position)
    vis_state.position[inds] = pre_grasps_js[max_idx]
    planner.visualize_spheres_rviz(vis_state)
    input('and grasp?')
    vis_state = JointState()
    vis_state.name = joint_state0.name
    vis_state.position = np.array(joint_state0.position)
    vis_state.position[inds] = grasp_js[max_idx]
    planner.visualize_spheres_rviz(vis_state)

    grasp_position = conv2.pose_to_matrix(grasps[max_idx])[:3, 3]
    pre_grasp_pose = conv2.pose_to_matrix(pre_grasps[max_idx])[:3, 3]
    dist = np.linalg.norm(grasp_position - pre_grasp_pose)
    gparams = {
        # 'pre_grasp_state': pre_grasps_js[max_idx],
        'grasp_approach_offset': [0,0,-dist],
    }
    # result = planner.pose_motion_plan(joint_state0, grasp, grasp_params=gparams)
    scores.pop(max_idx)
    grasps.pop(max_idx)
    grasp_js.pop(max_idx)

    input('next pre grasp...')
if result is None:
    print("No valid grasp found")
    sys.exit()
plan0, plan1 = result
t2 = rospy.Time.now().to_sec()
print("Scene setting time:", t1 - t0)
print("Motion planning time:", t2 - t1)

# plan retract
t0 = rospy.Time.now().to_sec()
grasp_state = JointState()
grasp_state.name = plan0.joint_names
grasp_state.position = plan0.points[-1].positions
joint_state1 = JointState()
joint_state1.name = plan1.joint_names
joint_state1.position = plan1.points[-1].positions
planner.set_planning_scene(
    all_pts,
    tgt_mesh,
    grasp_state,
    joint_state1,
    save_scene_file='/tmp/world_config.pth',
)
planner.visualize_rviz()
t1 = rospy.Time.now().to_sec()
# planner.visualize_spheres_rviz(grasp_state)
# input('vis next...')
planner.visualize_spheres_rviz(joint_state1)
place = [0.6, 0, 1.1, 0.5, -0.5, -0.5, -0.5]
place = [1.1, 0.32, 1.1, 0, 0.7, 0, 0.7]
plan2 = planner.pose_motion_plan(joint_state1, place)
t2 = rospy.Time.now().to_sec()
print("Scene setting time:", t1 - t0)
print("Motion planning time:", t2 - t1)

## execute ##
input("Execute?")
execute(plan0, wait=True, window=0.1, retime=True)
ee_close()
execute(plan1, wait=True, window=0.1, retime=True)
if plan2 is None:
    print("No valid retract found")
else:
    execute(plan2, wait=True, window=0.1, retime=True)
ee_open()
