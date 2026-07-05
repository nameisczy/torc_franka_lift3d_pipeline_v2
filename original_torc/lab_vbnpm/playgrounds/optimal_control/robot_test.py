from robot import Robot, MotomanRobot
from planning_scene import PlanningScene
from geometric_trajopt_ipopt import PoseTrajOpt
import numpy as np
import scipy as sp
import transformations as tf

# test the robot with the scene
# add environment collisions
pcd_total = []
# shelf-bottom
num_points = 1000
position = np.array([0.85, 0, 0.5])
half_size = np.array([0.175, 0.5, 0.5])
pcd = np.random.uniform(low=position-half_size, high=position+half_size, size=(num_points, 3))
pcd_total.append(pcd)
# shelf-top
num_points = 1000
position = np.array([0.85, 0, 1.42])
half_size = np.array([0.175, 0.5, 0.025])
pcd = np.random.uniform(low=position-half_size, high=position+half_size, size=(num_points, 3))
pcd_total.append(pcd)
# shelf-padding-left
num_points = 1000
position = np.array([0.85, -0.475, 1.2])
half_size = np.array([0.175, 0.025, 0.2])
pcd = np.random.uniform(low=position-half_size, high=position+half_size, size=(num_points, 3))
pcd_total.append(pcd)
# shelf-padding-right
num_points = 1000
position = np.array([0.85, 0.475, 1.2])
half_size = np.array([0.175, 0.025, 0.2])
pcd = np.random.uniform(low=position-half_size, high=position+half_size, size=(num_points, 3))
pcd_total.append(pcd)
# shelf-padding-back
num_points = 1000
position = np.array([1.0, 0, 1.2])
half_size = np.array([0.025, 0.5, 0.2])
pcd = np.random.uniform(low=position-half_size, high=position+half_size, size=(num_points, 3))
pcd_total.append(pcd)
pcd_total = np.concatenate(pcd_total, axis=0)


torso_b1 = ["torso_joint_b1"]
left = [
    "arm_left_joint_1_s",
    "arm_left_joint_2_l",
    "arm_left_joint_3_e",
    "arm_left_joint_4_u",
    "arm_left_joint_5_r",
    "arm_left_joint_6_b",
    "arm_left_joint_7_t",
]
right = [
    "arm_right_joint_1_s",
    "arm_right_joint_2_l",
    "arm_right_joint_3_e",
    "arm_right_joint_4_u",
    "arm_right_joint_5_r",
    "arm_right_joint_6_b",
    "arm_right_joint_7_t",
]
robot_joint_names = right#torso_b1 + left + right
robot = MotomanRobot(selected_joint_names=robot_joint_names)
# # scene = PlanningScene(robot, scene_pcd=pcd_total)
# scene = PlanningScene(robot, scene_pcd=None)
# scene.update_scene_pcd(pcd_total)


# set the joint angles of the robot multiple times and visualize
for i in range(10):
    q = np.random.uniform(robot.selected_joint_limits[:, 0], robot.selected_joint_limits[:, 1])
    robot.set_selected_joint_values(q)
    robot.visualize()
    input("Press Enter to continue...")