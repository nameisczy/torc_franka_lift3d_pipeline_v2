import os
import sys
import rospy

from curobo.geom.types import WorldConfig
from curobo.types.robot import RobotConfig
from curobo.types.file_path import ContentPath
from curobo.cuda_robot_model.util import load_robot_yaml

from rospkg import RosPack
from sensor_msgs.msg import JointState
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
    warmup=False,
)

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
    # "arm_right_joint_1_s": 0.75,
    # "arm_right_joint_2_l": 0,
    # "arm_right_joint_3_e": -0.6,
    # "arm_right_joint_4_u": -1.15,
    # "arm_right_joint_5_r": 0,
    # "arm_right_joint_6_b": -1.3,
    # "arm_right_joint_7_t": 0.0,
    "arm_right_joint_1_s": 0.2,
    "arm_right_joint_2_l": -0.7,
    "arm_right_joint_3_e": 0.0,
    "arm_right_joint_4_u": -1.7,
    "arm_right_joint_5_r": 0,
    "arm_right_joint_6_b": -1.3,
    "arm_right_joint_7_t": 0.0,
}

if os.path.isfile('/tmp/world_config.pth'):
    if os.stat('/tmp/world_config.pth').st_size > 0:
        planner.load_planning_scene('/tmp/world_config.pth')
    else:
        planner.world_config = WorldConfig.from_dict({})
        planner.update_world_motion_gen()
planner.visualize_rviz()
planner.visualize_spheres_rviz(joint_state)
plan = planner.joint_motion_plan(joint_state, goal)

ee_open()
input("Execute?")
execute(JointTrajectory())
execute(plan, window=0.1, retime=True)
