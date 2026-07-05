import os
import sys
import time
import rospy

import pytorch_kinematics as pk

from curobo.types.robot import RobotConfig
from curobo.types.file_path import ContentPath
from curobo.cuda_robot_model.util import load_robot_yaml

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
    warmup=False,
)

joint_state = rospy.wait_for_message('/joint_states_all', JointState, timeout=5)


p0 = planner.pose_motion_plan(joint_state, [1.1, -0.5, 1.1, 0.5, 0, 0.5, 0])

input('Go')
if p0:
    execute(JointTrajectory())
    execute(p0, window=0.1, retime=True)
else:
    print('Already there.')

input('Plan2')
joint_state = rospy.wait_for_message('/joint_states_all', JointState, timeout=5)
p1 = planner.pose_motion_plan(joint_state, [1.1, 0.5, 1.1, 0.5, 0, 0.5, 0])
input('Go')
if p1:
    execute(JointTrajectory())
    execute(p1, window=0.1, retime=True)
else:
    print('Already there.')
