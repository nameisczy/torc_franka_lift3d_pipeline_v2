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

import multiprocessing as mp
from task_planner.curobo_closed_loop import *

if __name__ == '__main__':
    ## init perception and planning interfaces ##
    rospy.init_node("test")
    # t0 = time.time()
    from task_planner.motoman import MotomanSDA10F
    robot = MotomanSDA10F(True, True)
    # planner = robot.init_motion_planner(planner='curobo', warmup=False)
    # t1 = time.time()
    # print('Init Time:', t1 - t0)

    # joint_state = rospy.wait_for_message('/joint_states_all', JointState, timeout=5)
    # p0 = planner.pose_motion_plan(joint_state, [1.1, -0.5, 1.1, 0.5, 0, 0.5, 0])
    # input('Go')
    # if p0:
    #     execute(JointTrajectory())
    #     execute(p0, window=0.1, retime=True)
    # else:
    #     print('Already there.')

    mp.set_start_method('spawn', force=True)
    target_poses = [
        [1.1, -0.5, 1.1, 0.5, 0, 0.5, 0],
        [1.1, 0.5, 1.1, 0.5, 0, 0.5, 0],
    ]
    closed_loop_pick_or_place(
        35,
        robot,
        True,
        target_poses,
    )
