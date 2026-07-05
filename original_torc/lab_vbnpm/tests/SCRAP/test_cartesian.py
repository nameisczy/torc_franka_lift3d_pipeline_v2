import os
import sys
import copy
import time
import rospy

import torch
import pytorch_kinematics as pk

from curobo.types.robot import RobotConfig
from curobo.types.state import JointState as JointState_cu
from curobo.types.file_path import ContentPath
from curobo.cuda_robot_model.util import load_robot_yaml
from curobo.util.trajectory import get_spline_interpolated_trajectory, get_interpolated_trajectory, get_smooth_trajectory

from rospkg import RosPack
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory
from lab_vbnpm.srv import ExecuteTrajectory
# from motion_planner.moveit_planner import MoveitPlanner
from motion_planner.curobo_planner import CuroboPlanner
from task_planner.eutils import ee_open, ee_close, execute

rospy.init_node("reset")
is_sim = True if len(sys.argv) > 1 else False
## instatiate planner ##
# rp = RosPack()
# robot_urdf = rp.get_path(
#     'motoman_sda10f_moveit_config'
# ) + '/config/gazebo_motoman_sda10f.urdf'
# planner = MoveitPlanner(
#     robot_urdf, ['motoman_left_ee', 'motoman_right_ee'],
#     ['arm_right', 'arm_left', 'arms'],
#     is_sim=is_sim
# )
rp = RosPack()
robot_urdf = rp.get_path('motoman_sda10f_moveit_config')
robot_urdf += '/config/gazebo_motoman_sda10f.urdf'
curobo_root = rp.get_path('lab_vbnpm')
curobo_root += '/robots/motoman/curobo/'
content_path = ContentPath(
    robot_config_absolute_path=curobo_root + 'motoman.yml',
    robot_urdf_absolute_path=curobo_root + 'motoman.urdf',
    robot_usd_absolute_path=curobo_root + 'motoman.usd',
    robot_asset_absolute_path=curobo_root,
)
robot_dict = load_robot_yaml(content_path)
curobo_config = robot_dict['robot_cfg']
planner = CuroboPlanner(
    robot_urdf,
    ['motoman_left_ee', 'motoman_right_ee'],
    curobo_config,
    is_sim=is_sim,
    warmup=True,
)

joint_state = rospy.wait_for_message('/joint_states_all', JointState, timeout=5)
g = Pose(
    position=Point(
        x=1.0747823221783712,
        y=-0.14214753946699277,
        z=1.080911760237506,
    ),
    orientation=Quaternion(
        x=0.5888208042179924,
        y=0.4130436989111367,
        z=-0.37556636227564133,
        w=0.5844953984736959,
    )
)

target_poses = [
    [1.1, -0.5, 1.1, 0.5, 0, 0.5, 0],
    [1.1, 0.5, 1.1, 0.5, 0, 0.5, 0],
]

# plan = planner.pose_motion_plan(joint_state, g)
# plan, mgr = planner.pose_motion_plan(joint_state, [0.6, 0, 1.1, 0.5, -0.5, 0.5, 0.5], return_all=True)
# raw_plan = mgr.interpolated_plan
# dt = mgr.interpolation_dt
# print(dt)
# p = get_interpolated_trajectory(raw_plan.position, copy.deepcopy(raw_plan))
# input('interp')
# plan = planner.joint_trajectory_from_curobo(raw_plan, dt, joint_state, visualize=True)
t0 = time.time()
plan = planner.pink_cartesian_motion(
    joint_state,
    # [0.1, -0.7, 1.2, 0., 1., 0., 0.],
    [0.48, -0.1, 1.25, 0.5, -0.5, 0.5, 0.5],
    # precise=False,
)
# plan = planner.pink_cartesian_motion(joint_state, target_poses[0])
t1 = time.time()
input('Go')
print('Cartesian time:', t1 - t0)
if plan:
    ee_open()
    # execute(JointTrajectory())
    execute(plan, window=0, retime=True)
else:
    print('Already there.')

input('Go')
joint_state = rospy.wait_for_message('/joint_states_all', JointState, timeout=5)

goal = {
    "torso_joint_b1": 0,
    "arm_left_joint_1_s": 1.75,
    "arm_left_joint_2_l": 0.8,
    "arm_left_joint_3_e": 0,
    "arm_left_joint_4_u": -0.66,
    "arm_left_joint_5_r": 0,
    "arm_left_joint_6_b": 0,
    "arm_left_joint_7_t": 0,
    "arm_right_joint_1_s": 0.75,
    "arm_right_joint_2_l": 0,
    "arm_right_joint_3_e": -0.6,
    "arm_right_joint_4_u": -1.15,
    "arm_right_joint_5_r": 0,
    "arm_right_joint_6_b": -1.3,
    "arm_right_joint_7_t": 0.0,
}

goal_js = planner.parse_joint_state(goal)
ord_state = goal_js.get_ordered_joint_state(planner.pk_joint_names)
q_0 = ord_state.position.to('cpu').numpy()
m = planner.chain.forward_kinematics(q_0).cuda().get_matrix()
pos = m[0, :3, 3]
rot = pk.matrix_to_quaternion(m[0, :3, :3])

t0 = time.time()
p2, arrived = planner.pink_cartesian_motion(
    joint_state,
    [0.48, 0.1, 1.25, 0.5, -0.5, 0.5, 0.5],
    return_all=True,
)
# p2, arrived = planner.pink_cartesian_motion(joint_state, target_poses[1], return_all=True)
# p2, arrived = planner.pink_cartesian_motion(joint_state, g, return_all=True)
# p2, arrived = planner.pink_cartesian_motion(joint_state, (pos, rot), return_all=True)
t1 = time.time()
print('Cartesian time:', t1 - t0)
print('Arrived:', arrived)
input('Go')
execute(p2, window=0, retime=True)
