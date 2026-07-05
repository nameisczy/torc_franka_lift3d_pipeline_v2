import functools
import os
import copy
import torch
import numpy as np
import trimesh as tm
from scipy.spatial import KDTree

from curobo.types.math import Pose
from curobo.types.base import TensorDeviceType
from curobo.geom.sphere_fit import SphereFitType
from curobo.util.logger import setup_curobo_logger
from curobo.geom.sdf.world import CollisionCheckerType, CollisionQueryBuffer
from curobo.types.robot import JointState, RobotConfig
from curobo.geom.types import Cuboid, WorldConfig, Mesh
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, MotionGenStatus
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml

import rospy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose as Pose_MSG
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import JointState as JointState_MSG
from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from motion_planner.motion_planner import MotionPlanner
rospy.init_node('test')

pose = rospy.get_param('/workspace/pose', [0.8, 0.65, 0.9])
size = rospy.get_param('/workspace/size', [0.7, 1.31, 0.55])
padding = 0.1
thick = 0.05
size_top = [size[0], size[1] + 5 * thick, thick]
size_bottom = [size[0], size[1] + 2 * thick, pose[2] + thick]
size_left = [size[0], 3 * thick, size[2] + pose[2] + size_top[2]]
size_right = [size[0], 3 * thick, size[2] + pose[2] + size_top[2]]
size_back = [thick, size[1], size[2]]
pose_top = [
    pose[0] + size[0] - size_top[0] / 2,
    pose[1] - size[1] / 2,
    pose[2] + size[2] + 1.5 * size_top[2],
]
pose_bottom = [
    pose[0] + size_bottom[0] / 2,
    pose[1] - size[1] / 2,
    pose[2] - 0.5 * size_bottom[2] + thick,
]
pose_left = [
    pose[0] + size_left[0] / 2,
    pose[1] + size_left[1] / 2,
    size_left[2] / 2,
]
pose_right = [
    pose[0] + size_right[0] / 2,
    pose[1] - size[1] - size_right[1] / 2,
    size_right[2] / 2,
]
pose_back = [
    pose[0] + size[0] + size_back[0] / 2,
    pose[1] - size[1] / 2,
    pose[2] + size[2] / 2,
]

world_config = {
    # cuboid:
    #   name:
    #       dims: x, y, z
    #       pose: x, y, z, qw, qx, qy, qz
    "cuboid": {
        "shelf_top": {
            "pose": [*pose_top, 1, 0, 0, 0],
            "dims": np.add(size_top, [padding, padding, 0.5 * padding]),
        },
        "shelf_bottom": {
            "pose": [*pose_bottom, 1, 0, 0, 0],
            "dims": np.add(size_bottom, [padding, padding, 0]),
        },
        "shelf_left": {
            "pose": [*pose_left, 1, 0, 0, 0],
            "dims": np.add(size_left, padding),
        },
        "shelf_right": {
            "pose": [*pose_right, 1, 0, 0, 0],
            "dims": np.add(size_right, padding),
        },
        "shelf_back": {
            "pose": [*pose_back, 1, 0, 0, 0],
            "dims": np.add(size_back, padding),
        },
    },
}
world_config = WorldConfig.from_dict(world_config)
collision_cache = world_config.get_cache_dict()
collision_cache["mesh"] = 1

tensor_args=TensorDeviceType()
robot_file = "motoman.yml"
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_file,
    world_config,
    tensor_args,
    use_cuda_graph=True,
    collision_cache=collision_cache,
    collision_checker_type=CollisionCheckerType.MESH,
)
motion_gen = MotionGen(motion_gen_config)
robot_config = RobotWorldConfig.load_from_config(
    robot_file,
    world_config,
)
