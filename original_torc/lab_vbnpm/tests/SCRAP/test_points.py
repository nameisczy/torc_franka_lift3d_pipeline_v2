import os
import sys
import copy
import pickle
import trimesh
import numpy as np

import rospy
from rospkg import RosPack
from moveit_commander import conversions as conv

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, TransformStamped, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lab_vbnpm.srv import EEControl, EEControlResponse
from lab_vbnpm.srv import ExecuteTrajectory, ExecuteTrajectoryResponse

from utils import conversions as conv2
from motion_planner.moveit_planner import MoveitPlanner

from motoman_msgs.msg import DynamicJointTrajectory, DynamicJointPoint, DynamicJointsGroup

from task_planner.eutils import ee_open, ee_close, execute, wait_till


def send_point_to_robot(plan):
    dyn_traj = DynamicJointTrajectory()
    dyn_traj.header.stamp = rospy.Time.now()
    dyn_traj.joint_names = plan.joint_names
    dyn_traj.points = [
        DynamicJointPoint(
            num_groups=4,
            groups=[
                DynamicJointsGroup(
                    group_number=0,
                    num_joints=7,
                    valid_fields=0,
                    positions=p.positions[:7],
                    velocities=p.velocities[:7],
                    accelerations=[0.0] * 7,
                    effort=[0.0] * 7,
                    time_from_start=p.time_from_start,
                ),
                DynamicJointsGroup(
                    group_number=1,
                    num_joints=7,
                    valid_fields=0,
                    positions=p.positions[7:14],
                    velocities=p.velocities[7:14],
                    accelerations=[0.0] * 7,
                    effort=[0.0] * 7,
                    time_from_start=p.time_from_start,
                ),
                DynamicJointsGroup(
                    group_number=2,
                    num_joints=1,
                    valid_fields=0,
                    positions=p.positions[15:16],
                    velocities=p.velocities[15:16],
                    accelerations=[0.0],
                    effort=[0.0],
                    time_from_start=p.time_from_start,
                ),
                DynamicJointsGroup(
                    group_number=3,
                    num_joints=1,
                    valid_fields=0,
                    positions=p.positions[15:16],
                    velocities=p.velocities[15:16],
                    accelerations=[0.0],
                    effort=[0.0],
                    time_from_start=p.time_from_start,
                ),
            ]
        ) for p in plan.points
    ]
    # print(dyn_traj)
    joint_cmd_pub.publish(dyn_traj)


rospy.init_node("planning")
joint_cmd_pub = rospy.Publisher(
    '/joint_command', DynamicJointTrajectory, queue_size=1
)

goal = {
    "arm_left_joint_1_s": 1.75,
    "arm_left_joint_2_l": 0.8,
    "arm_left_joint_3_e": 0,
    "arm_left_joint_4_u": -0.66,
    "arm_left_joint_5_r": 0,
    "arm_left_joint_6_b": 0,
    "arm_left_joint_7_t": 0.05,
    "arm_right_joint_1_s": 1.2102,
    "arm_right_joint_2_l": -0.0540,
    "arm_right_joint_3_e": 2.3140,
    "arm_right_joint_4_u": 1.4975,
    "arm_right_joint_5_r": -2.7925,
    "arm_right_joint_6_b": -0.9223,
    "arm_right_joint_7_t": 0,
    "torso_joint_b1": 0,
    # "torso_joint_b2": 0,
}

# first point
joint_state = rospy.wait_for_message('/joint_states_all', JointState, timeout=5)
execute(
    JointTrajectory(
        joint_names=joint_state.name,
        points=[
            JointTrajectoryPoint(
                positions=joint_state.position,
                velocities=[0.0] * len(joint_state.position),
                time_from_start=rospy.Duration(0.0),
            ),
        ]
    ),
    wait=False
)

# trajectory
t0 = rospy.Time.now().to_sec()
T = 5
N = 12
for i in range(1, N + 1):
    rospy.sleep(0.1)
    temp = copy.deepcopy(goal)
    temp["arm_left_joint_7_t"] = (i) * (1.57 / N)
    dur = i * (T / N)
    print(temp["arm_left_joint_7_t"])
    print(dur)
    vels = np.zeros(len(temp))
    # if i < N:
    #     vels[list(temp.keys()).index("arm_left_joint_7_t")] = -0.5*(1.57 / T)
    execute(
        JointTrajectory(
            joint_names=list(temp.keys()),
            points=[
                JointTrajectoryPoint(
                    positions=list(temp.values()),
                    velocities=vels,
                    time_from_start=rospy.Duration(dur),
                ),
            ]
        ),
        wait=False,
    )
wait_till(stop=True)
t1 = rospy.Time.now().to_sec()
print(t1 - t0)
rospy.sleep(0.25)
execute(JointTrajectory(), wait=False)
