import math
import time
import copy
import os
import logging
import torch
import numpy as np
import trimesh as tm
from scipy.spatial import KDTree

from curobo.types.math import Pose
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.geom.types import WorldConfig, Mesh
from curobo.geom.sphere_fit import SphereFitType
from curobo.util.trajectory import InterpolateType
from curobo.types.robot import JointState, RobotConfig
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.util.logger import log_error, log_info, log_warn
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, GraspPlanResult
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import MotionGenConfig, MotionGenPlanConfig

import rospy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose as Pose_MSG
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import JointState as JointState_MSG
from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from motion_planner.motion_planner import MotionPlanner
from task_planner.curobo_visibility_planner import VisibilityPlanner
from execution_scene.motoman_interface import MotomanInterface as MI
from perception.perception_fast import PerceptionInterface
from task_planner.curobo_closed_loop import find_ik_solutions
from utils.conversions import pose_to_matrix
from tracikpy import TracIKSolver

from typing import Any, Dict, List, Optional, Tuple, Union

import qpsolvers
import pinocchio as pin
from spatialmath.base import trinterp
from pink import Configuration, solve_ik
from pink.utils import get_root_joint_dim
from pink.tasks import FrameTask, PostureTask
from pink.exceptions import NotWithinConfigurationLimits


def get_torc_planner_frames() -> dict[str, str]:
    if os.environ.get("TORC_ROBOT", "motoman").strip().lower() in ("franka", "panda"):
        return {
            "base_link": os.environ.get("TORC_CUROBO_BASE_LINK", "panda_link0"),
            "ee_link": os.environ.get("TORC_CUROBO_EE_LINK", "panda_tcp"),
            "mujoco_tool_body": os.environ.get("TORC_MUJOCO_TOOL_BODY", "gripper0_right_grip_site"),
            "contact_proxy": os.environ.get("TORC_FRANKA_CONTACT_PROXY", "gripper0_right_grip_site"),
        }
    return {
        "base_link": "base_link",
        "ee_link": "motoman_right_ee",
        "mujoco_tool_body": "motoman_right_ee",
        "contact_proxy": "robotiq",
    }


def stage_probe(stage: str, detail: str = ""):
    path = os.environ.get("TORC_STAGE_LOG_FILE")
    message = f"[CUROBO_PLANNER_STAGE {time.time():.6f}] {stage}"
    if detail:
        message += f" | {detail}"
    print(message, flush=True)
    if not path:
        return
    try:
        with open(path, "a") as stage_file:
            print(message, file=stage_file, flush=True)
    except Exception:
        pass


# override pinocchio's check_limits to ignore extra joints
def override_check_limits_motoman(self, tol=-1e-3, safety_break=True):
    q_max = self.model.upperPositionLimit
    q_min = self.model.lowerPositionLimit
    root_nq, _ = get_root_joint_dim(self.model)
    for i in range(root_nq, 15):  # only check arm/torso joints
        if q_max[i] <= q_min[i] + tol:  # no limit
            continue
        if self.q[i] < q_min[i] - tol or self.q[i] > q_max[i] + tol:
            if safety_break:
                raise NotWithinConfigurationLimits(
                    i,
                    self.q[i],
                    q_min[i],
                    q_max[i],
                )
            logging.warning(
                "Value %f at index %d is out of limits: [%f, %f]",
                self.q[i],
                i,
                q_min[i],
                q_max[i],
            )


def override_check_limits_ur5e(self, tol=-1e-3, safety_break=True):
    q_max = self.model.upperPositionLimit
    q_min = self.model.lowerPositionLimit
    root_nq, _ = get_root_joint_dim(self.model)
    for i in range(root_nq, 6):  # only check arm/torso joints
        if q_max[i] <= q_min[i] + tol:  # no limit
            continue
        if self.q[i] < q_min[i] - tol or self.q[i] > q_max[i] + tol:
            if safety_break:
                raise NotWithinConfigurationLimits(
                    i,
                    self.q[i],
                    q_min[i],
                    q_max[i],
                )
            logging.warning(
                "Value %f at index %d is out of limits: [%f, %f]",
                self.q[i],
                i,
                q_min[i],
                q_max[i],
            )


def override_check_limits_franka(self, tol=-1e-3, safety_break=True):
    q_max = self.model.upperPositionLimit
    q_min = self.model.lowerPositionLimit
    root_nq, _ = get_root_joint_dim(self.model)
    for i in range(root_nq, root_nq + 7):
        if q_max[i] <= q_min[i] + tol:
            continue
        if self.q[i] < q_min[i] - tol or self.q[i] > q_max[i] + tol:
            if safety_break:
                raise NotWithinConfigurationLimits(
                    i,
                    self.q[i],
                    q_min[i],
                    q_max[i],
                )
            logging.warning(
                "Value %f at index %d is out of limits: [%f, %f]",
                self.q[i],
                i,
                q_min[i],
                q_max[i],
            )


class CuroboPlanner(MotionPlanner):

    def __init__(
        self,
        robot_file,
        end_effector_links,
        curobo_config,
        grasp_ignore_collision_links=[],
        resolution=0.005,
        is_sim=False,
        warmup=True,
    ):
        self.urdf = robot_file
        super().__init__(robot_file, end_effector_links)
        self.curobo_config = curobo_config
        self.disable_collision_links = grasp_ignore_collision_links
        self.resolution = resolution
        self.marker_publisher = rospy.Publisher(
            "/visualization_marker",
            MarkerArray,
            latch=True,
            queue_size=10,
        )
        self.traj_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            DisplayTrajectory,
            latch=True,
            queue_size=10,
        )
        self.tensor_args = TensorDeviceType()
        self.ee = self.curobo_config["kinematics"]["ee_link"]
        self.torc_frames = get_torc_planner_frames()
        self.attach_link = self.torc_frames.get("ee_link", self.ee)
        self.is_franka_robot = (
            os.environ.get("TORC_ROBOT", "motoman").strip().lower()
            in ("franka", "panda")
        )
        self.base_pose_world = np.eye(4, dtype=np.float64)
        if self.is_franka_robot:
            raw_base = os.environ.get(
                "TORC_FRANKA_PLANNER_BASE_POSE_WORLD",
                os.environ.get("TORC_FRANKA_MUJOCO_BASE_POSE_WORLD", "0,0,0.86"),
            )
            values = [float(v) for v in raw_base.replace(";", ",").split(",") if v.strip()]
            if len(values) >= 3:
                self.base_pose_world[:3, 3] = values[:3]
        self.world_to_base = np.linalg.inv(self.base_pose_world)
        self.chain = self.chains[self.ee]
        self.pk_joint_names = [j.name for j in self.chain.get_joints()]
        self.pin_model = pin.buildModelFromUrdf(self.urdf)
        self.pin_data = self.pin_model.createData()
        self.reset(warmup)
        self.set_planning_scene_time = 0
        self.number_steps = 7
        self.ray_cost_enabled = True
        self.point_cost_enabled = True

        if 'motoman' in self.urdf:
            Configuration.check_limits = override_check_limits_motoman
        elif 'ur5e' in self.urdf:
            Configuration.check_limits = override_check_limits_ur5e
        elif self.is_franka_robot:
            Configuration.check_limits = override_check_limits_franka

    def _points_to_planner_frame(self, points):
        if points is None or not self.is_franka_robot:
            return points
        pts = np.asarray(points, dtype=np.float64)
        original_shape = pts.shape
        if pts.size == 0:
            return pts.reshape(original_shape)
        pts = pts.reshape(-1, 3)
        pts_h = np.concatenate(
            [pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)],
            axis=1,
        )
        pts = (pts_h @ self.world_to_base.T)[:, :3]
        return pts.reshape(original_shape)

    def _pose_xyz_to_planner_frame(self, xyz):
        if not self.is_franka_robot:
            return list(xyz)
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        xyz_h = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
        xyz = (self.world_to_base @ xyz_h)[:3]
        return [float(xyz[0]), float(xyz[1]), float(xyz[2])]

    def _world_config_to_planner_frame(self, config):
        if not self.is_franka_robot:
            return config
        cfg = copy.deepcopy(config)
        for group in cfg.values():
            if not isinstance(group, dict):
                continue
            for obstacle in group.values():
                if not isinstance(obstacle, dict) or "pose" not in obstacle:
                    continue
                pose = list(obstacle["pose"])
                pose[:3] = self._pose_xyz_to_planner_frame(pose[:3])
                obstacle["pose"] = pose
        return cfg

    def reset(self, warmup=True):
        pose = rospy.get_param("/workspace/pose", [0.55, 0.655, 1.05])
        size = rospy.get_param("/workspace/size", [0.4, 1.31, 0.52])
        padding = 0.04
        size_top = [size[0], size[1] + 0.2, 0.1]
        size_bottom = [size[0], size[1] + 0.2, pose[2]]
        size_left = [size[0], 0.1, size[2]]
        size_right = [size[0], 0.1, size[2]]
        size_back = [0.05, size[1], size[2]]
        pose_top = [
            pose[0] + 0.5 * size_top[0],
            pose[1] - 0.5 * size[1],
            pose[2] + size[2] + 0.5 * size_top[2],
        ]
        pose_bottom = [
            pose[0] + 0.5 * size_bottom[0],
            pose[1] - 0.5 * size[1],
            pose[2] - 0.5 * size_bottom[2],
        ]
        pose_left = [
            pose[0] + 0.5 * size_left[0],
            pose[1] + 0.5 * size_left[1],
            pose[2] + 0.5 * size_left[2],
        ]
        pose_right = [
            pose[0] + 0.5 * size_right[0],
            pose[1] - size[1] - 0.5 * size_right[1],
            pose[2] + 0.5 * size_right[2],
        ]
        pose_back = [
            pose[0] + size[0] + 0.5 * size_back[0],
            pose[1] - 0.5 * size[1],
            pose[2] + 0.5 * size[2],
        ]

        self.static_world_config = {
            # cuboid:
            #   name:
            #       dims: x, y, z
            #       pose: x, y, z, qw, qx, qy, qz
            "cuboid": {
                "shelf_bottom": {
                    "pose": [*pose_bottom, 1, 0, 0, 0],
                    "dims": np.add(size_bottom, [padding, padding, 0]),
                },
                # "shelf_top": {
                #     "pose": [*pose_top, 1, 0, 0, 0],
                #     "dims": np.add(size_top, [padding, padding, padding / 2]),
                # },
                # "shelf_left": {
                #     "pose": [*pose_left, 1, 0, 0, 0],
                #     "dims": np.add(size_left, padding),
                # },
                # "shelf_right": {
                #     "pose": [*pose_right, 1, 0, 0, 0],
                #     "dims": np.add(size_right, padding),
                # },
                # "shelf_back": {
                #     "pose": [*pose_back, 1, 0, 0, 0],
                #     "dims": np.add(size_back, padding),
                # },
            },
            # "voxel": {
            #     "base": {
            #         "dims": size,
            #         "pose": [0, 0, 0, 0, 1, 0, 0, 0],
            #         "voxel_size": self.resolution,
            #         "feature_dtype": torch.bfloat16,
            #     },
            # }
        }
        self.static_world_config = self._world_config_to_planner_frame(
            self.static_world_config
        )
        self.static_world_config_no_bottom = copy.deepcopy(
            self.static_world_config
        )
        self.static_world_config_no_bottom["cuboid"].pop("shelf_bottom", None)

        self.world_config = WorldConfig.from_dict(self.static_world_config)
        self.world_config_raytrace = WorldConfig.from_dict(
            self.static_world_config_no_bottom
        )
        collision_cache = self.world_config.get_cache_dict()
        collision_cache["mesh"] = 1
        collision_cache["obb"] = 100

        robot_world_cnf = RobotWorldConfig.load_from_config(
            copy.deepcopy(self.curobo_config),
            self.world_config,
            collision_activation_distance=0.0,
        )
        self.robot_world = RobotWorld(robot_world_cnf)

        # motion gen init
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            copy.deepcopy(self.curobo_config),
            self.world_config,
            self.tensor_args,
            use_cuda_graph=True,
            # velocity_scale=0.8,
            # trajopt_dt=0.125,
            # trajopt_tsteps=16,
            # js_trajopt_dt=0.5,
            # js_trajopt_tsteps=36,
            # maximum_trajectory_dt=0.5,
            # interpolation_dt=0.1,
            collision_cache=collision_cache,
            # interpolation_type=InterpolateType.KUNZ_STILMAN_OPTIMAL,
            # trajopt_fix_terminal_action=True, #Important
            # optimize_dt=False,
            # trajopt_seed_ratio={"linear": 0.0, "bias": 1.0},
        )
        self.motion_gen = MotionGen(motion_gen_config)
        zero = torch.tensor(0.0).to("cuda")
        self.set_ray_weight(self.motion_gen.trajopt_solver, zero)
        self.set_point_weight(self.motion_gen.trajopt_solver, zero)
        self.set_ray_weight(self.motion_gen.js_trajopt_solver, zero)
        self.set_point_weight(self.motion_gen.js_trajopt_solver, zero)
        print(
            self.motion_gen.kinematics.kinematics_config.joint_limits.velocity
        )

        if warmup:
            self.motion_gen.warmup(n_goalset=32)

        self.motion_gen_short = self.motion_gen
        # motion_gen_config_short = MotionGenConfig.load_from_robot_config(
        #     copy.deepcopy(self.curobo_config),
        #     self.world_config,
        #     self.tensor_args,
        #     use_cuda_graph=True,
        #     velocity_scale=0.6,
        #     trajopt_dt=0.125,
        #     trajopt_tsteps=7,
        #     collision_cache=collision_cache,
        #     optimize_dt=False,
        # )
        # self.motion_gen_short = MotionGen(motion_gen_config_short)
        # self.set_ray_weight(self.motion_gen_short.trajopt_solver, zero)
        # self.set_point_weight(self.motion_gen_short.trajopt_solver, zero)
        # self.set_ray_weight(self.motion_gen_short.js_trajopt_solver, zero)
        # self.set_point_weight(self.motion_gen_short.js_trajopt_solver, zero)
        # # self.set_line_weight(self.motion_gen_short.trajopt_solver, 1000000)
        # # self.set_line_weight(self.motion_gen_short.js_trajopt_solver, 1000000)
        # if warmup:
        #     self.motion_gen_short.warmup()

        # mpc init
        # self.rays_collision_checker = VisibilityPlanner(
        #     copy.deepcopy(self.curobo_config),
        #     self.world_config,
        #     self.urdf,
        # )
        # https://curobo.org/_api/curobo.wrap.reacher.mpc.html#curobo.wrap.reacher.mpc.MpcSolver.step
        robot_config = RobotConfig.from_dict(
            copy.deepcopy(self.curobo_config),
            self.tensor_args,
        )
        robot_config.kinematics.kinematics_config.joint_limits.velocity *= 0.25
        mpc_config = MpcSolverConfig.load_from_robot_config(
            robot_config,
            self.world_config,
            store_rollouts=True,
            # set metrics false here due to weird tensor being contiguous error
            # also remove line 533 from mpc.py in curobo src directory
            compute_metrics=False,
            step_dt=0.05,
            use_cuda_graph=False,
            particle_opt_iters=None,
            use_lbfgs=False,
        )
        self.mpc = MpcSolver(mpc_config)
        print(self.mpc.kinematics.kinematics_config.joint_limits.velocity)

    def set_init_points(self, solver, point_1, point_2, point_3):
        # solver = self.motion_gen.trajopt_solver.solver
        for optimizer in solver.solver.optimizers:
            optimizer.rollout_fn.init_pos_1.copy_(point_1)
            optimizer.rollout_fn.init_pos_2.copy_(point_2)
            optimizer.rollout_fn.init_pos_3.copy_(point_3)

    def set_ray_weight(self, solver, value):
        if type(value) is not torch.Tensor:
            value = torch.tensor(value).to("cuda")
        for optimizer in solver.solver.optimizers:
            if not hasattr(optimizer.rollout_fn, "ray_cost"):
                if torch.count_nonzero(value).item() == 0:
                    log_warn("ray_cost not present on rollout_fn; skipping zero-weight init")
                    continue
                raise AttributeError("ray_cost not present on rollout_fn for nonzero weight")
            optimizer.rollout_fn.ray_cost.weight.copy_(value)

    def set_point_weight(self, solver, value):
        if type(value) is not torch.Tensor:
            value = torch.tensor(value).to("cuda")
        for optimizer in solver.solver.optimizers:
            if not hasattr(optimizer.rollout_fn, "point_cost"):
                if torch.count_nonzero(value).item() == 0:
                    log_warn("point_cost not present on rollout_fn; skipping zero-weight init")
                    continue
                raise AttributeError("point_cost not present on rollout_fn for nonzero weight")
            optimizer.rollout_fn.point_cost.weight.copy_(value)

    def set_line_weight(self, solver, value):
        if type(value) is not torch.Tensor:
            value = torch.tensor(value).to("cuda")
        for optimizer in solver.solver.optimizers:
            optimizer.rollout_fn.straight_line_cost.weight.copy_(value)

    def set_collision_dist(self, solver, value):
        if type(value) is not torch.Tensor:
            value = torch.tensor(value).to("cuda")
        for opt in solver.solver.optimizers:
            opt.rollout_fn.primitive_collision_cost.activation_distance.copy_(
                value
            )

    def set_mpc_goal_joint_remove_goals(self, goal, start=None):
        self.mpc.enable_pose_cost(enable=False)
        self.mpc.enable_cspace_cost(enable=False)

        goal_state = self.parse_joint_state(goal)
        if start is None:
            start = rospy.wait_for_message(
                "/joint_states_all", JointState_MSG, 5
            )
        start_state = self.parse_joint_state(start)
        curobo_goal = Goal(
            current_state=start_state, goal_state=goal_state.clone()
        )

        goal_buffer = self.mpc.setup_solve_single(curobo_goal, 1)
        self.mpc.update_goal(goal_buffer)

    def set_mpc_goal_pose(self, goal, start=None):
        self.mpc.enable_pose_cost(enable=True)
        self.mpc.enable_cspace_cost(enable=False)

        if start is None:
            start = rospy.wait_for_message(
                "/joint_states_all", JointState_MSG, 5
            )
        goal_pose = self.parse_goal_pose(goal)
        start_state = self.parse_joint_state(start)
        curobo_goal = Goal(
            current_state=start_state,
            goal_state=start_state.clone(),
            goal_pose=goal_pose,
        )

        goal_buffer = self.mpc.setup_solve_single(curobo_goal, 1)
        self.mpc.update_goal(goal_buffer)

    def trajopt_rospy_communication_custom(
        self, start, goal, var_steps_scale=1.0
    ):
        t0 = time.time()
        # Get ray normalized direction vectors and the origin of the rays
        size = 12 * 6
        default = np.zeros((4, size, 3)).tolist()

        # Get rays from perception pipeline
        rays = rospy.get_param("/rays_from_perception", default)  # list
        rays = np.asarray(rays)
        rays_origin = rospy.get_param("/rays_origin_from_perception", [])
        if len(rays_origin) == 0:
            return False, None
        rays_origin = np.asarray(rays_origin)

        # x1 = [3], x2_batch = [num_rays, 3], length=1, spheres=64, r=.01
        ray_endpoints = rospy.get_param(
            "/rays_endpoints_from_perception", default
        )
        ray_endpoints = np.asarray(ray_endpoints)

        # Collision check rays in a loop and once a ray set not fully occluded is found, break the loop
        which_rays = 0
        for i in range(
                rays.shape[0]
        ):  # Won't get to this point if the rospy parameters havent been set yet as rays_origin will have failed out
            which_rays = i
            ray_mask, largest_cone_ray, ray_cone_sizes, largest_cone_ind = (
                self.rays_collision_checker.raytrace_batch(
                    rays_origin[i], ray_endpoints[i], spheres=64, r=6.0
                )
            )
            if len(rays[which_rays, ray_mask.cpu().numpy()]) > 0:
                print("which rays------------------------", which_rays)
                break  # Farthest center point that worked first

        ray_mask_vis = ray_mask.cpu().numpy()
        ray_mask_np = largest_cone_ind.cpu().numpy()
        weight = torch.tensor(1.0).to("cuda")
        if len(rays[which_rays, ray_mask_vis]) == 0:
            print(
                "no valid rays"
            )  # , ray_mask_vis.shape, rays[ray_mask_vis].shape)
            weight *= 0.0
        print("ray mask np shape", ray_mask_np)  # , rays[0], rays_origin)

        # default_value = np.array([1.0, 0.0, 0.0])  #For the ones in collision make them face backwards
        # collision_free_rays = np.full_like(rays, default_value)
        # collision_free_rays[ray_mask_np] = rays[ray_mask_np] #Fill in the correct rays here with the default being backwards rays

        widest_cylinder_ray = np.expand_dims(rays[which_rays, ray_mask_np], 0)
        print("widest_cylinder_ray should be shape [1, 3]", widest_cylinder_ray)
        best_ray_ik_solution = PerceptionInterface.ik_option_points_one_ray(
            rays_origin[which_rays], rays[which_rays, ray_mask_np]
        )[0]
        rospy.set_param("/best_ray_pose", best_ray_ik_solution.tolist())

        rospy.set_param(
            "/rays_collision_free",
            rays[which_rays, ray_mask_vis].reshape((-1, 3)).tolist(),
        )
        rospy.set_param("/ray_largest_cone", widest_cylinder_ray.tolist())
        rospy.set_param(
            "/motion_gen_origin", rays_origin[which_rays].tolist()
        )  # Do this to ensure we use the matching collision free rays with the origin that the rays had
        rospy.set_param("/ray_it_worked", which_rays)
        torch_rays = torch.from_numpy(widest_cylinder_ray).to("cuda")
        self.motion_gen.trajopt_solver.solver.optimizers[
            0
        ].rollout_fn.ray_cost.origin.copy_(
            torch.from_numpy(rays_origin[which_rays]).to("cuda")
        )  # Need to use the .copy_() function not = cuda graph no set new var
        self.motion_gen.trajopt_solver.solver.optimizers[
            0].rollout_fn.ray_cost.rays.copy_(torch_rays.to("cuda"))
        self.motion_gen.trajopt_solver.solver.optimizers[
            0].rollout_fn.ray_cost.weight.copy_(weight * self.ray_cost_enabled)
        self.motion_gen.trajopt_solver.solver.optimizers[
            1
        ].rollout_fn.ray_cost.origin.copy_(
            torch.from_numpy(rays_origin[which_rays]).to("cuda")
        )  # Need to use the .copy_() function not = cuda graph no set new var
        self.motion_gen.trajopt_solver.solver.optimizers[
            1].rollout_fn.ray_cost.rays.copy_(torch_rays.to("cuda"))
        self.motion_gen.trajopt_solver.solver.optimizers[
            1].rollout_fn.ray_cost.weight.copy_(weight * self.ray_cost_enabled)

        # For grasping
        target_position = rospy.get_param("/target_position", [0, 0, 0])
        target_quaternion = rospy.get_param("/target_quaternion", [1, 0, 0, 0])
        self.motion_gen.trajopt_solver.solver.optimizers[
            0].rollout_fn.point_cost.target_position.copy_(
                torch.tensor(target_position).to("cuda")
            )
        self.motion_gen.trajopt_solver.solver.optimizers[
            0].rollout_fn.point_cost.target_quaternion.copy_(
                torch.tensor(target_quaternion).to("cuda")
            )
        self.motion_gen.trajopt_solver.solver.optimizers[
            0].rollout_fn.point_cost.weight.copy_(self.point_cost_enabled)
        self.motion_gen.trajopt_solver.solver.optimizers[
            1].rollout_fn.point_cost.target_position.copy_(
                torch.tensor(target_position).to("cuda")
            )
        self.motion_gen.trajopt_solver.solver.optimizers[
            1].rollout_fn.point_cost.target_quaternion.copy_(
                torch.tensor(target_quaternion).to("cuda")
            )
        self.motion_gen.trajopt_solver.solver.optimizers[
            1].rollout_fn.point_cost.weight.copy_(self.point_cost_enabled)
        print("rospy comm time:", time.time() - t0)

        # Figure step number
        # The actual horizon is horzion - 4

        # start_m = self.chain.forward_kinematics(start.position.cpu().numpy()).get_matrix().numpy()[0]
        start_m = self.motion_gen.kinematics.get_state(
            start.position
        ).ee_pose.get_numpy_matrix()[0]
        goal_m = goal.get_numpy_matrix()[0]
        dist = self.axis_angle(start_m, goal_m)
        dist[3:] *= 0.0
        dist[0:3] *= 1.0
        dist = np.linalg.norm(dist)
        horizon = self.motion_gen.trajopt_solver.action_horizon
        number_steps = int((horizon - 3 - 1) * dist * 2)
        number_steps = np.clip(number_steps, 5, horizon - 3 - 1)
        # number_steps = horizon-3-1
        self.number_steps = number_steps
        print("-------------Dist:", dist, number_steps)
        self.motion_gen.trajopt_solver.solver.newton_optimizer.number_fixed = (
            horizon - number_steps
        )
        self.motion_gen.finetune_trajopt_solver.solver.newton_optimizer.number_fixed = (
            horizon - number_steps
        )
        self.motion_gen.trajopt_solver.solver.optimizers[
            0].rollout_fn.needed_steps = (
                number_steps  # .copy_(torch.tensor(number_steps).to('cuda'))
            )
        self.motion_gen.trajopt_solver.solver.optimizers[
            1].rollout_fn.needed_steps = (
                number_steps  # .copy_(torch.tensor(number_steps).to('cuda'))
            )
        for opt in self.motion_gen.finetune_trajopt_solver.solver.optimizers:  #
            opt.rollout_fn.needed_steps = number_steps
        self.motion_gen.trajopt_solver.needed_steps = number_steps
        self.motion_gen.finetune_trajopt_solver.needed_steps = number_steps

        return True, largest_cone_ray

    def mpc_rospy_communication_custom(self):
        t0 = time.time()
        # Get ray normalized direction vectors and the origin of the rays
        size = 12 * 6
        default = np.zeros((4, size, 3)).tolist()

        # Get rays from perception pipeline
        rays = rospy.get_param("/rays_from_perception", default)  # list
        rays = np.asarray(rays)
        rays_origin = rospy.get_param("/rays_origin_from_perception", [])
        if len(rays_origin) == 0:
            return False, None
        rays_origin = np.asarray(rays_origin)

        # x1 = [3], x2_batch = [num_rays, 3], length=1, spheres=64, r=.01
        ray_endpoints = rospy.get_param(
            "/rays_endpoints_from_perception", default
        )
        ray_endpoints = np.asarray(ray_endpoints)

        # Collision check rays in a loop and once a ray set not fully occluded is found, break the loop
        which_rays = 0
        for i in range(
                rays.shape[0]
        ):  # Won't get to this point if the rospy parameters havent been set yet as rays_origin will have failed out
            which_rays = i
            ray_mask, largest_cone_ray, ray_cone_sizes, largest_cone_ind = (
                self.rays_collision_checker.raytrace_batch(
                    rays_origin[i], ray_endpoints[i], spheres=64, r=6.0
                )
            )
            if len(rays[which_rays, ray_mask.cpu().numpy()]) > 0:
                print("which rays------------------------", which_rays)
                break  # Farthest center point that worked first

        ray_mask_vis = ray_mask.cpu().numpy()
        ray_mask_np = largest_cone_ind.cpu().numpy()
        weight = torch.tensor(1.0).to("cuda")
        if len(rays[which_rays, ray_mask_vis]) == 0:
            print(
                "no valid rays"
            )  # , ray_mask_vis.shape, rays[ray_mask_vis].shape)
            weight *= 0.0
        print("ray mask np shape", ray_mask_np)  # , rays[0], rays_origin)

        # For the ones in collision make them face backwards
        # default_value = np.array([1.0, 0.0, 0.0])
        # collision_free_rays = np.full_like(rays, default_value)
        # Fill in the correct rays here with the default being backwards rays
        # collision_free_rays[ray_mask_np] = rays[ray_mask_np]

        widest_cylinder_ray = np.expand_dims(rays[which_rays, ray_mask_np], 0)
        print("widest_cylinder_ray should be shape [1, 3]", widest_cylinder_ray)
        best_ray_ik_solution = PerceptionInterface.ik_option_points_one_ray(
            rays_origin[which_rays], rays[which_rays, ray_mask_np]
        )[0]
        rospy.set_param("/best_ray_pose", best_ray_ik_solution.tolist())

        rospy.set_param(
            "/rays_collision_free",
            rays[which_rays, ray_mask_vis].reshape((-1, 3)).tolist(),
        )
        rospy.set_param("/ray_largest_cone", widest_cylinder_ray.tolist())
        rospy.set_param(
            "/motion_gen_origin", rays_origin[which_rays].tolist()
        )  # Do this to ensure we use the matching collision free rays with the origin that the rays had
        rospy.set_param("/ray_it_worked", which_rays)
        torch_rays = torch.from_numpy(widest_cylinder_ray).to("cuda")
        self.mpc.solver.optimizers[0].rollout_fn.ray_cost.origin.copy_(
            torch.from_numpy(rays_origin[which_rays]).to("cuda")
        )  # Need to use the .copy_() function not = cuda graph no set new var
        self.mpc.solver.optimizers[0].rollout_fn.ray_cost.rays.copy_(
            torch_rays.to("cuda")
        )
        self.mpc.solver.optimizers[0].rollout_fn.ray_cost.weight.copy_(
            weight * self.ray_cost_enabled
        )

        # For grasping
        target_position = rospy.get_param("/target_position", [0, 0, 0])
        target_quaternion = rospy.get_param("/target_quaternion", [1, 0, 0, 0])
        self.mpc.solver.optimizers[
            0].rollout_fn.point_cost.target_position.copy_(
                torch.tensor(target_position).to("cuda")
            )
        self.mpc.solver.optimizers[
            0].rollout_fn.point_cost.target_quaternion.copy_(
                torch.tensor(target_quaternion).to("cuda")
            )
        self.mpc.solver.optimizers[0].rollout_fn.point_cost.weight.copy_(
            self.point_cost_enabled
        )
        print("rospy comm time:", time.time() - t0)

        return True, largest_cone_ray

    def run_graph_search(self, joint_state):
        joint_config = self.parse_joint_state(joint_state)
        ik = TracIKSolver(
            self.urdf,
            self.torc_frames.get("base_link", "base_link"),
            self.torc_frames.get("ee_link", "motoman_right_ee"),
        )
        goal_config, _ = find_ik_solutions(
            ik
        )  # We pass it the other stuff through rospy
        if goal_config is None:
            return
        goal_config = self.parse_joint_state(
            goal_config, names=joint_state.name
        )
        graph_result = self.motion_gen.graph_search(joint_config, goal_config)
        graph_success = torch.count_nonzero(graph_result.success).item()
        print("graph success", graph_success)

    def make_contiguous(self, plan):
        for attr_name in dir(plan):  # Loop over all attributes
            attr = getattr(plan, attr_name)
            if isinstance(attr, torch.Tensor) and not attr.is_contiguous():
                setattr(plan, attr_name, attr.contiguous())
        return plan

    # If you don't manually update world right before calling this, you will have issues with the collision mesh
    def mpc_step_wrapper(
        self, joint_state, shift_steps, max_attempts, dt=0.05, visualize=True
    ):
        tb = rospy.Time.now().to_sec()
        while (self.set_planning_scene_time == 0
               or tb - self.set_planning_scene_time < 3.0):
            tb = rospy.Time.now().to_sec()
        self.update_world_mpc()  # CRUCIAL
        rays_updated_yet = False
        while rays_updated_yet == False:
            self.update_world_mpc()
            rays_updated_yet, _ = self.mpc_rospy_communication_custom()
        print(
            "MPC TYPES",
            self.mpc.solver.optimizers[0].rollout_fn.ray_cost.origin
        )
        # self.run_graph_search(joint_state)
        # print("goal type", type(goal))
        # print(goal)
        # goal_js = self.parse_joint_state(goal)
        # goal_js = self.parse_joint_state(np.array([0.46989863, 0.91965475, 0.84495721, 2.78612939, -0.13365973, -1.73285056, -1.55046083, 2.9491305]), name=joint_state.name)
        current_state = self.parse_joint_state(joint_state)
        current_state = self.make_contiguous(current_state)
        self.set_mpc_goal_joint_remove_goals(joint_state)
        mpc_output = self.mpc.step(
            current_state=current_state,
            shift_steps=shift_steps,
            max_attempts=max_attempts,
        )
        print("mpc_output", type(mpc_output.action))
        print("position there", mpc_output.action.position.shape)
        plan = mpc_output.action
        plan = self.make_contiguous(plan)
        raw_plan = plan
        joint_trajectory_plan = self.joint_trajectory_from_curobo(
            plan, dt, joint_state, visualize
        )
        return joint_trajectory_plan, raw_plan

    def set_planning_scene(
        self,
        points,
        target_trimesh=None,
        attach_js=None,
        attach_zoffset=0,
        filter_js=None,
        resolution=None,
        visualize=False,
        save_scene_file=None,
        send_to_gpu=True,
        raytrace_version=False,
    ):
        stage_probe(
            "set_planning_scene begin",
            f"points_shape={None if points is None else list(np.asarray(points).shape)} "
            f"target={target_trimesh is not None} attach={attach_js is not None} "
            f"filter={filter_js is not None} franka={self.is_franka_robot}",
        )
        # init world config
        if raytrace_version == False:
            world_config = WorldConfig.from_dict(self.static_world_config)
        else:
            world_config = WorldConfig.from_dict(
                self.static_world_config_no_bottom
            )

        # add estimated target geometry to planning scene
        self.motion_gen.detach_object_from_robot(self.attach_link)
        if target_trimesh is not None:
            # filter out points inside target geometry
            if points is not None:
                points = points[~target_trimesh.contains(points)]

            if attach_js is not None:
                # convert trimesh mesh to curobo mesh
                sf = tm.sample.sample_surface_even(target_trimesh, 10000)
                surface, find = sf
                surface = self._points_to_planner_frame(surface)
                target_mesh = Mesh.from_pointcloud(
                    surface,
                    self.resolution,
                    "target",
                )
                # attach target to robot
                attach_state = self.parse_joint_state(attach_js)

                offset = None
                if attach_zoffset:
                    offset = Pose.from_list(
                        [0, 0, attach_zoffset, 1, 0, 0, 0],
                        self.tensor_args,
                    )
                self.motion_gen.attach_external_objects_to_robot(
                    attach_state,
                    [target_mesh],
                    link_name=self.attach_link,
                    surface_sphere_radius=0.01,
                    sphere_fit_type=SphereFitType.SAMPLE_SURFACE,
                    world_objects_pose_offset=offset,
                )

        # reset world if points is None
        if points is None:
            self.world_config = world_config
            if send_to_gpu:
                self.update_world_motion_gen()
                self.update_world_mpc()
            stage_probe("set_planning_scene done", "points=None")
            return
        points = self._points_to_planner_frame(points)

        # get spheres for desired robot state
        if filter_js is not None:
            remove = self.parse_joint_state(filter_js)
            remove = remove.position
            balls = self.motion_gen.kinematics.get_robot_as_spheres(remove)[0]

            # filter out points that are close robot state(s)
            kdtree = KDTree(points)
            inds_to_remove = set()
            for sphere in balls:
                p = sphere.pose[:3]
                r = sphere.radius + 0.03
                res = kdtree.query_ball_point(p, r)
                inds_to_remove.update(res)
            points = np.delete(points, list(inds_to_remove), axis=0)

        # convert point cloud to mesh
        if resolution is None:
            resolution = self.resolution
        if len(points) > 0:
            scene_mesh = Mesh.from_pointcloud(points, resolution, "world")
            if visualize:
                meshes = [scene_mesh.get_trimesh_mesh()]
                if target_trimesh is not None:
                    meshes.append(target_mesh.get_trimesh_mesh())
                tm.scene.Scene(meshes).show()

            # add to world config
            world_config.add_obstacle(scene_mesh)

        # update collision world
        if raytrace_version == False:
            self.world_config = world_config
        else:
            self.world_config_raytrace = world_config
        if send_to_gpu:
            stage_probe("set_planning_scene update_world begin")
            self.update_world_motion_gen()
            self.update_world_mpc()
            stage_probe("set_planning_scene update_world done")

        if save_scene_file:
            torch.save(self.world_config, save_scene_file)
        if self.set_planning_scene_time == 0:
            self.set_planning_scene_time = rospy.Time.now().to_sec()
        stage_probe(
            "set_planning_scene done",
            f"points_after={len(points) if points is not None else None}",
        )

    def update_world_motion_gen(self):
        self.motion_gen.clear_world_cache()
        self.motion_gen.update_world(self.world_config)
        self.robot_world.update_world(self.world_config)

    def update_world_mpc(self):
        # self.rays_collision_checker.update_world(self.world_config_raytrace)
        self.mpc.update_world(self.world_config)

    def load_planning_scene(self, world_config_file="/tmp/world_config.pth"):
        self.world_config = torch.load(world_config_file)
        self.update_world_motion_gen()

    def visualize_rviz(self, raytrace=False):
        # marker to delete previous visualization
        delete = MarkerArray()
        delete.markers.append(Marker())
        delete.markers[-1].action = Marker.DELETEALL

        # visualize updates
        marker = MarkerArray()
        if raytrace == False:
            wc = self.motion_gen.world_model
        else:
            wc = self.rays_collision_checker.world_config
        wc = wc.get_collision_check_world()
        mesh = WorldConfig.get_scene_graph(wc).to_mesh()
        marker.markers.append(Marker())
        marker.markers[-1].id = 800
        marker.markers[-1].action = Marker.ADD
        marker.markers[-1].header.frame_id = "world"
        marker.markers[-1].header.stamp = rospy.Time.now()
        marker.markers[-1].type = Marker.TRIANGLE_LIST
        marker.markers[-1].scale.x = 1.0
        marker.markers[-1].scale.y = 1.0
        marker.markers[-1].scale.z = 1.0
        marker.markers[-1].pose.orientation.w = 1.0
        marker.markers[-1].color.a = 0.9
        marker.markers[-1].color.g = 1.0
        for triangle in mesh.triangles:
            for vertex in triangle:
                point = Point()
                point.x = vertex[0]
                point.y = vertex[1]
                point.z = vertex[2]
                marker.markers[-1].points.append(point)
            marker.markers[-1].colors.append(ColorRGBA(0.0, 1.0, 0.0, 0.5))

        # publish
        self.marker_publisher.publish(delete)
        self.marker_publisher.publish(marker)

    def visualize_spheres_rviz(self, joints, kinematics=None):
        state = self.parse_joint_state(joints)
        # visualize updates
        marker = MarkerArray()
        i = 900
        if kinematics is None:
            kinematics = self.motion_gen.kinematics
        spheres = kinematics.get_robot_as_spheres(state.position)
        # p=self.motion_gen.kinematics.get_state(state.position).ee_pose
        # test_sphere = Sphere(radius=0.03, pose=p.tolist(), name='test')
        for sphere in spheres[0]:
            marker.markers.append(Marker())
            marker.markers[-1].id = i
            i += 1
            marker.markers[-1].action = Marker.ADD
            marker.markers[-1].header.frame_id = "world"
            marker.markers[-1].header.stamp = rospy.Time.now()
            marker.markers[-1].type = Marker.SPHERE
            pose = sphere.pose
            r = sphere.radius * 2
            marker.markers[-1].scale.x = r
            marker.markers[-1].scale.y = r
            marker.markers[-1].scale.z = r
            marker.markers[-1].pose.position.x = pose[0]
            marker.markers[-1].pose.position.y = pose[1]
            marker.markers[-1].pose.position.z = pose[2]
            marker.markers[-1].pose.orientation.w = 1.0
            marker.markers[-1].color.a = 0.6
            marker.markers[-1].color.b = 1.0

        # ee_pose = self.motion_gen.kinematics.get_state(
        #     state.position
        # ).ee_position
        # ee_pose = np.asarray(ee_pose.squeeze().cpu())
        # spheres = self.motion_gen.robot_cfg.kinematics.kinematics_config.link_spheres
        # spheres = np.asarray(spheres.cpu())[-8:]
        # for sphere in spheres:
        #     marker.markers.append(Marker())
        #     marker.markers[-1].id = i
        #     i += 1
        #     marker.markers[-1].action = Marker.ADD
        #     marker.markers[-1].header.frame_id = "world"
        #     marker.markers[-1].header.stamp = rospy.Time.now()
        #     marker.markers[-1].type = Marker.SPHERE
        #     pose = sphere[:3] + ee_pose
        #     print(sphere, ee_pose, pose)
        #     r = sphere[3]
        #     marker.markers[-1].scale.x = r
        #     marker.markers[-1].scale.y = r
        #     marker.markers[-1].scale.z = r
        #     marker.markers[-1].pose.position.x = pose[0]
        #     marker.markers[-1].pose.position.y = pose[1]
        #     marker.markers[-1].pose.position.z = pose[2]
        #     marker.markers[-1].pose.orientation.w = 1.0
        #     marker.markers[-1].color.a = 1.0
        #     marker.markers[-1].color.r = 1.0

        self.marker_publisher.publish(marker)

    def parse_joint_state(self, js, names=None):
        velocities = None
        accelerations = None
        if type(js) is JointState_MSG:
            positions = js.position
            if len(js.velocity) > 0:
                velocities = js.velocity
            if len(js.effort) > 0:
                accelerations = js.effort
            joint_names = js.name
        elif hasattr(js, "__iter__") and len(js) > 0:
            if type(js) is dict:
                positions = list(js.values())
                joint_names = list(js.keys())
            else:
                positions = js
                joint_names = names if names else self.motion_gen.joint_names
        else:
            print("Invalid Joint State Spec:", js)
            return None
        joint_state = JointState.from_numpy(
            joint_names,
            positions,
            velocities,
            accelerations,
        ).unsqueeze(0)
        # print('joint_state', joint_state)
        joint_state = self.motion_gen.get_active_js(joint_state)
        return joint_state

    def franka_pose_joint_target(self, pose_world, qinit=None):
        if not self.is_franka_robot:
            raise RuntimeError("franka_pose_joint_target is only valid for Franka")
        if not hasattr(self, "_franka_lift_ik_solver"):
            self._franka_lift_ik_solver = TracIKSolver(
                self.urdf,
                self.torc_frames.get("base_link", "panda_link0"),
                self.torc_frames.get("ee_link", "panda_tcp"),
            )
            self._franka_lift_world_to_base = np.eye(4, dtype=np.float64)
            raw_base = os.environ.get("TORC_FRANKA_MUJOCO_BASE_POSE_WORLD", "0,0,0.86")
            values = [float(v) for v in raw_base.replace(";", ",").split(",") if v.strip()]
            if len(values) >= 3:
                base = np.eye(4, dtype=np.float64)
                base[:3, 3] = values[:3]
                self._franka_lift_world_to_base = np.linalg.inv(base)

        pose_matrix_world = (
            pose_to_matrix(pose_world)
            if hasattr(pose_world, "position")
            else np.asarray(pose_world, dtype=np.float64)
        )
        pose_base = self._franka_lift_world_to_base @ pose_matrix_world

        qseed = None if qinit is None else np.asarray(qinit, dtype=np.float64).reshape(-1)
        js = self._franka_lift_ik_solver.ik(pose_base, qinit=qseed)
        if js is None:
            js = self._franka_lift_ik_solver.ik(pose_base)
        if js is None:
            return None
        joint_state_tensor = JointState.from_position(
            torch.tensor(np.array(js)[np.newaxis, :], dtype=torch.float32).cuda(),
            joint_names=list(self._franka_lift_ik_solver.joint_names),
        )
        active = self.motion_gen.get_active_js(joint_state_tensor).position
        return active.detach().cpu().numpy()[0].tolist()

    def franka_lift_joint_target(self, contact_pose_world, lift_height, qinit=None):
        pose_world = (
            pose_to_matrix(contact_pose_world)
            if hasattr(contact_pose_world, "position")
            else np.asarray(contact_pose_world, dtype=np.float64)
        )
        lift_pose_world = np.array(pose_world, dtype=np.float64, copy=True)
        lift_pose_world[2, 3] += float(lift_height)
        return self.franka_pose_joint_target(lift_pose_world, qinit=qinit)

    def joint_motion_plan(
        self,
        start,
        goal,
        path_constraint=None,
        constraint_in_goal_frame=True,
        visualize=True,
        return_all=False,
    ):
        start_state = self.parse_joint_state(start)
        goal_state = self.parse_joint_state(goal)
        rays_updated_yet = False
        # while rays_updated_yet == False:
        #     rays_updated_yet, _ = self.trajopt_rospy_communication_custom(start_state, goal_state)

        if path_constraint is not None:
            motion_gen = self.motion_gen_short
            hold_pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.tensor_args.to_device(path_constraint),
                project_to_goal_frame=constraint_in_goal_frame,
            )
        else:
            motion_gen = self.motion_gen
            hold_pose_cost_metric = None

        result = motion_gen.plan_single_js(
            start_state,
            goal_state,
            MotionGenPlanConfig(
                # timeout=5,
                # need_graph_success=True,
                pose_cost_metric=hold_pose_cost_metric,
                enable_finetune_trajopt=False,  # bool(path_constraint),
                # time_dilation_factor=0.5,
                enable_graph=False,
            ),
        )

        success = bool(result.success.squeeze())
        # print('dt:', result.optimized_dt)
        # print('compute time:', result.total_time)
        if not success:
            print("Valid Query:", result.valid_query)
            print(result.status)
            if return_all:
                return None, result
            return None

        # plan = result.interpolated_plan
        # dt = result.interpolation_dt
        plan = result.optimized_plan
        dt = result.optimized_dt
        traj = self.joint_trajectory_from_curobo(plan, dt, start, visualize)

        if return_all:
            return traj, result
        return traj

    def parse_goal_pose(self, goal):
        def planner_position(xyz):
            xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
            if self.is_franka_robot:
                xyz_h = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
                xyz = (self.world_to_base @ xyz_h)[:3]
            return [float(xyz[0]), float(xyz[1]), float(xyz[2])]

        if hasattr(goal, "__iter__") and len(goal) > 0:
            if isinstance(goal, tuple):
                pos = goal[0]
                if self.is_franka_robot:
                    pos = self.tensor_args.to_device(planner_position(pos))
                return Pose(position=pos, quaternion=goal[1])
            if not hasattr(goal[0], "as_integer_ratio"):
                # TODO multiple goals still breaks right now
                # goal_list = []
                pos_list = []
                quat_list = []
                for i, g in enumerate(goal):
                    if type(g) is list or type(g) is np.ndarray:
                        pos_list = planner_position(g[0:3])
                        quat_list = g[3:7]
                        break
                    elif type(g) is Pose_MSG:
                        qx = g.orientation.x
                        qy = g.orientation.y
                        qz = g.orientation.z
                        qw = g.orientation.w
                        x = g.position.x
                        y = g.position.y
                        z = g.position.z
                        # pose_l = [x, y, z, qw, qx, qy, qz]
                        pos_l = planner_position([x, y, z])
                        quat_l = [qw, qx, qy, qz]
                    else:
                        print("Invalid goal item type:", g, type(g))
                        return None
                    # pose_t = Pose.from_list(pose_l)
                    # goal_list.append(pose_t)
                    # goal_list.append(pose_l)
                    pos_list.append(pos_l)
                    quat_list.append(quat_l)
                # pose_t = torch.tensor(goal_list, dtype=torch.float32).cuda()

                # create tensors of shape (1, n, 3) and (1, n, 4)
                pos_t = torch.tensor(
                    pos_list, dtype=torch.float32
                )[torch.newaxis, ...].cuda()
                quat_t = torch.tensor(
                    quat_list, dtype=torch.float32
                )[torch.newaxis, ...].cuda()

                # n_goalset is set automatically
                goal_pose = Pose(position=pos_t, quaternion=quat_t)

                # goal_pose = Pose(position=pose_t, n_goalset=len(goal_list))
                # goal_pose.quaternion = torch.zeros(1, 0).cuda()
                # goal_pose = Pose.cat(goal_list)
                # goal_pose = Pose.from_list(goal_list)
                # goal_pose.n_goalset = len(goal_list)
            elif len(goal) == 7:
                goal_l = list(goal)
                goal_l[:3] = planner_position(goal_l[:3])
                goal_pose = Pose.from_list(goal_l)
            else:
                print("Invalid goal:", goal)
                return None
        elif type(goal) is Pose_MSG:
            qx = goal.orientation.x
            qy = goal.orientation.y
            qz = goal.orientation.z
            qw = goal.orientation.w
            x = goal.position.x
            y = goal.position.y
            z = goal.position.z
            xyz = planner_position([x, y, z])
            goal_pose = Pose.from_list([*xyz, qw, qx, qy, qz])
        else:
            print("Invalid goal type:", type(goal))
            return None

        return goal_pose

    def pose_motion_plan(
        self,
        start,
        goal,
        path_constraint=None,
        constraint_in_goal_frame=True,
        offset=None,
        grasp_params=None,
        visualize=True,
        return_all=False,
        grasp=True,
        # custom_bias_state=None
    ):
        stage_probe(
            "pose_motion_plan begin",
            f"path_constraint={path_constraint} offset={offset is not None} grasp={grasp}",
        )
        start_state = self.parse_joint_state(start)
        goal_pose = self.parse_goal_pose(goal)
        # tb = rospy.Time.now().to_sec()
        # while self.set_planning_scene_time == 0 or tb - self.set_planning_scene_time < 3.0:
        #     tb = rospy.Time.now().to_sec()
        stage_probe("pose_motion_plan update_world begin")
        self.update_world_motion_gen()  # CRUCIAL
        stage_probe("pose_motion_plan update_world done")
        # rays_updated_yet = False
        # while rays_updated_yet == False:
        #     self.update_world_motion_gen()
        #     rays_updated_yet, _ = self.trajopt_rospy_communication_custom(start_state, goal_pose)
        if grasp == False:
            ik = TracIKSolver(
                self.urdf,
                self.torc_frames.get("base_link", "base_link"),
                self.torc_frames.get("ee_link", "motoman_right_ee"),
            )
            goal_config, ray_pose = find_ik_solutions(
                ik
            )  # We pass it the other stuff through rospy
            if goal_config is not None:
                print("using ray_pose")
                goal_pose = self.parse_goal_pose(ray_pose)
            else:
                print("ray pose failed")
                goal_pose = self.parse_goal_pose(goal)
        else:
            print("using grasp pose")
            goal_pose = self.parse_goal_pose(goal)

        # plan approach, grasp, and retraction
        if grasp_params is not None:
            r_offset = grasp_params.get("retract_offset", [0, 0, 0.05])
            retract = torch.tensor(r_offset, dtype=torch.float32)
            retract = Pose(retract.unsqueeze(0).cuda())

            pgjs = grasp_params.get("pre_grasp_state", None)
            if pgjs is None:
                ga_offset = grasp_params.get(
                    "grasp_approach_offset", [0, 0, -0.1]
                )
                pre_grasp = torch.tensor(ga_offset, dtype=torch.float32)
                pre_grasp = Pose(pre_grasp.unsqueeze(0).cuda())
                goal_pose = goal_pose.unsqueeze(0)
                result = self.motion_gen.plan_grasp(
                    start_state,
                    goal_pose,
                    MotionGenPlanConfig(
                        # timeout=5,
                        # enable_finetune_trajopt=False,
                        # time_dilation_factor=0.5,
                        enable_graph=False,
                    ),
                    disable_collision_links=self.disable_collision_links,
                    grasp_approach_offset=pre_grasp,
                    retract_constraint_in_goal_frame=False,
                    retract_offset=retract,
                )
            else:
                pre_grasp_state = JointState.from_position(
                    torch.tensor(pgjs, dtype=torch.float32).unsqueeze(0).cuda(),
                    self.motion_gen.joint_names,
                )
                result = self.plan_grasp_js(
                    start_state,
                    pre_grasp_state,
                    goal_pose,
                    MotionGenPlanConfig(
                        # timeout=5,
                        # enable_finetune_trajopt=False,
                        # time_dilation_factor=0.5,
                        enable_graph=False,
                    ),
                    disable_collision_links=self.disable_collision_links,
                    retract_constraint_in_goal_frame=False,
                    retract_offset=retract,
                )
            success = bool(result.success.squeeze())
            # print(result.goalset_result.status)
            # print(result.approach_result.status)
            # print(result.grasp_result.status)
            # print(result.retract_result.status)
            # print('grasp dt:', result.grasp_trajectory_dt)
            # print('retract dt:', result.retract_trajectory_dt)
            # print('compute time:', result.planning_time)
            if not success:
                print("Valid Query:", result.valid_query)
                print(result.status)
                if return_all:
                    return None, result
                return None

            g_plan = result.grasp_trajectory
            g_dt = result.grasp_trajectory_dt
            g_traj = self.joint_trajectory_from_curobo(g_plan, g_dt, start)

            r_plan = result.retract_trajectory
            r_dt = result.retract_trajectory_dt
            r_traj = self.joint_trajectory_from_curobo(r_plan, r_dt, start)

            if visualize:
                traj = JointTrajectory()
                traj.joint_names = g_traj.joint_names
                n_traj = copy.deepcopy(r_traj)
                last_time = g_traj.points[-1].time_from_start
                for i in range(len(n_traj.points)):
                    n_traj.points[i].time_from_start += last_time
                traj.points = g_traj.points + n_traj.points
                self.visualize_traj_rviz(traj)

            if return_all:
                return g_traj, r_traj, result
            return g_traj, r_traj

        hold_pose_cost_metric = None
        if path_constraint is not None:
            motion_gen = self.motion_gen_short
            if path_constraint != "short":
                hold_pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.tensor_args.to_device(path_constraint),
                    project_to_goal_frame=constraint_in_goal_frame,
                )
        else:
            motion_gen = self.motion_gen

        if offset is not None:
            offset = Pose.from_list(offset)
            if constraint_in_goal_frame:
                goal_pose = goal_pose.clone().multiply(offset)
            else:
                goal_pose = offset.clone().multiply(goal_pose.clone())

        # plan to pose
        # print("planning now")
        if goal_pose.n_goalset > 1:
            stage_probe("motion_gen.plan_goalset begin", f"n_goalset={goal_pose.n_goalset}")
            result = self.motion_gen.plan_goalset(
                start_state,
                goal_pose,
                MotionGenPlanConfig(
                    # timeout=1,
                    pose_cost_metric=hold_pose_cost_metric,
                    enable_finetune_trajopt=False,  # bool(path_constraint),
                    # partial_ik_opt=True,
                    # time_dilation_factor=0.5,
                    enable_graph=False,
                    # custom_bias_state=custom_bias_state
                ),
            )
            stage_probe("motion_gen.plan_goalset done")
        else:
            # self.set_cartesian_seed(start, goal)
            stage_probe("motion_gen.plan_single begin")
            result = self.motion_gen.plan_single(
                start_state,
                goal_pose,
                MotionGenPlanConfig(
                    # timeout=1,
                    pose_cost_metric=hold_pose_cost_metric,
                    enable_finetune_trajopt=False,  # bool(path_constraint),
                    # partial_ik_opt=True,
                    # time_dilation_factor=0.5,
                    enable_graph=False,
                ),
            )
            stage_probe("motion_gen.plan_single done")
        # print("planning over")

        success = bool(result.success.squeeze())
        # print('dt:', result.optimized_dt)
        # print('compute time:', result.total_time)
        if not success:
            print("Valid Query:", result.valid_query)
            print(result.status)
            stage_probe("pose_motion_plan failed", f"status={result.status}")
            if return_all:
                return None, result
            return None

        # plan = result.interpolated_plan
        # dt = result.interpolation_dt
        plan = result.optimized_plan
        dt = result.optimized_dt
        traj = self.joint_trajectory_from_curobo(
            plan, dt, start, visualize=True
        )
        stage_probe("pose_motion_plan done", "success=True")
        # for i in range(self.number_steps, self.motion_gen.trajopt_solver.action_horizon):
        #     traj.points[i].positions = traj.points[-1].positions.copy() #Turn the last horzion - self.number_steps to the last state
        #     traj.points[i].velocities = traj.points[-1].velocities.copy()
        #     traj.points[i].accelerations = traj.points[-1].accelerations.copy()
        # self.visualize_traj_rviz(traj)
        # Should be doing the optimization of the unfixed actions as if the next action is fixed so when replacing here it should be good.
        if return_all:
            return traj, result
        return traj

    def axis_angle(self, T, Td):
        e = np.empty(6)
        e[:3] = Td[:3, -1] - T[:3, -1]
        R = Td[:3, :3] @ T[:3, :3].T
        li = np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ]
        )
        if bool(np.linalg.norm(li) < 20 * np.finfo(np.float64).eps):
            # diagonal matrix case
            if np.trace(R) > 0:
                # (1,1,1) case
                a = np.zeros((3, ))
            else:
                a = np.pi / 2 * (np.diag(R) + 1)
        else:
            # non-diagonal matrix case
            ln = np.linalg.norm(li)
            a = math.atan2(ln, np.trace(R) - 1) * li / ln
        e[3:] = a
        return e

    def validate_trajectory(self, traj):
        # q should be shape [batch, horizon, dof]
        t0 = time.time()
        q = torch.zeros(1, len(traj.points), self.motion_gen.dof)
        for i, p in enumerate(traj.points):
            js_tensor = (
                torch.tensor(p.positions,
                             dtype=torch.float32).contiguous().unsqueeze(0)
            )  # .cuda()
            js_state = JointState.from_position(
                js_tensor,
                joint_names=traj.joint_names,
            )
            js_state = self.motion_gen.get_active_js(js_state)
            q[0, i, :] = js_state.position
        q = q.cuda()
        mask = self.robot_world.validate_trajectory(q)
        t1 = time.time()
        print("validate time:", t1 - t0)
        return mask.all().item()

    def set_cartesian_seed(self, start, goal):
        inds = list(map(start.name.index, self.motion_gen.joint_names))
        ss = copy.deepcopy(start)
        ss.name = self.pk_joint_names
        ss.position = np.array(ss.position)[inds].tolist()
        ss.velocity = []
        ss.effort = []
        traj, arrived = self.cartesian_motion(
            ss,
            goal,
            precise=False,
            visualize=True,
            return_all=True,
        )
        pos, vel, ts, jt = MI.retime_trajectroy(
            traj.points,
            0.15,
            20,
            500,
        )
        horizon = self.motion_gen.trajopt_solver.action_horizon
        times = np.linspace(
            0,
            jt.duration,
            horizon,
        )
        positions = jt(times)
        seed = torch.tensor(positions, dtype=torch.float32).unsqueeze(0).cuda()
        print("Seed shape:", seed.shape)
        self.motion_gen.trajopt_solver.custom_seed.copy_(seed)

    # Replace with pink
    def cartesian_motion(
        self,
        start,
        goal,
        offset=None,
        constraint_in_goal_frame=True,
        precise=True,
        visualize=True,
        return_all=False,
    ):
        """
        code inspired heavily from a few places:
        - robotics-toolbox's pservo function:
            https://github.com/petercorke/robotics-toolbox-python/blob/master/roboticstoolbox/tools/p_servo.py
        - pytorch_kinematics's compute_dq using SVD function:
            https://github.com/UM-ARM-Lab/pytorch_kinematics/blob/master/src/pytorch_kinematics/ik.py
        """
        start_state = self.parse_joint_state(start)
        goal_pose = self.parse_goal_pose(goal)
        if offset is not None:
            offset = Pose.from_list(offset)
            if constraint_in_goal_frame:
                goal_pose = goal_pose.clone().multiply(offset)
            else:
                goal_pose = offset.clone().multiply(goal_pose.clone())

        ord_state = start_state.get_ordered_joint_state(self.pk_joint_names)
        q_0 = ord_state.position.cpu().numpy()
        T = self.chain.forward_kinematics(q_0).get_matrix().numpy()[0]
        Td = goal_pose.get_numpy_matrix()[0]

        max_iters = 200
        js_dt = 0.0001  # extra fast because will be retimed later
        if precise:
            gain = 1
            dt = 0.2
            threshold = 0.001
        else:
            gain = 0.5
            dt = 0.05
            threshold = 0.1

        traj = JointTrajectory()
        traj.joint_names = start.name
        inds = list(map(start.name.index, self.pk_joint_names))
        i = 0
        q = q_0[0]
        qd = np.zeros_like(q)

        lower, upper = self.chain.get_joint_limits()

        arrived = False
        for i in range(max_iters):
            pos = np.array(start.position)
            vel = np.zeros_like(pos)
            pos[inds] = q
            vel[inds] = qd
            traj.points.append(JointTrajectoryPoint())
            traj.points[-1].positions = pos
            traj.points[-1].velocities = vel
            traj.points[-1].time_from_start = rospy.Duration(js_dt * i)
            if arrived:
                break

            # e = axis-angle difference
            aa = self.axis_angle(T, Td)
            if precise:
                e = aa
            else:
                e = aa / np.linalg.norm(aa)

            # get desired ee velocity
            if isinstance(gain, (int, np.integer, float, np.floating)):
                k = gain * np.eye(6)
            else:
                k = np.diag(gain)
            v = k @ e
            # did ee arrive?
            arrived = np.sum(np.abs(aa)) < threshold

            # update qd
            J = self.chain.jacobian(q)
            U, D, Vh = torch.linalg.svd(J)
            m = D.shape[1]
            # tmpA = U @ (D @ D.transpose(1, 2) + reg) @ U.transpose(1, 2)
            # singular_val = torch.diagonal(D)
            denom = D**2 + 1e-9
            prod = D / denom
            # J^T (JJ^T + lambda^2I)^-1 =
            #       = V @ (D @ D^T + lambda^2I)^-1 @ U^T
            #       = sum_i (d_i / (d_i^2 + lambda^2) v_i @ u_i^T)
            # should be equivalent to damped least squares
            inverted = torch.diag_embed(prod)
            # drop columns from V
            Vh = Vh[:, :m, :]
            total = Vh.transpose(1, 2) @ inverted @ U.transpose(1, 2)
            # dq = J^T (JJ^T + lambda^2I)^-1 dx
            qd = total @ torch.tensor(v, dtype=torch.float32).unsqueeze(1)
            qd = qd.squeeze().numpy()
            # print('e', np.max(e))
            # print('qd', np.max(np.abs(qd)))

            # update q and T
            q += qd * dt
            # q = np.clip(q, lower, upper)
            T = self.chain.forward_kinematics(q).get_matrix().numpy()[0]

        print("Iterations:", i, np.sum(np.abs(aa)), threshold)

        if visualize:
            self.visualize_traj_rviz(traj)

        if return_all:
            return traj, arrived
        return traj

    def pad_joint_values(
        self, all_joint_names, subset_joint_names, subset_values
    ):
        """
        Create a zero-padded full joint vector for all_joint_names,
        filling in known joint values by name.

        Args:
            all_joint_names (List[str]): full list of joint names (length N)
            subset_joint_names (List[str]): list of known joints (length M)
            subset_values (np.ndarray): shape (M,), joint values

        Returns:
            np.ndarray: shape (N,), full joint vector with known values filled in
        """
        full_q = np.zeros(len(all_joint_names), dtype=np.float64)
        name_to_index = {name: i for i, name in enumerate(all_joint_names)}

        for name, val in zip(subset_joint_names, subset_values):
            if name not in name_to_index:
                raise ValueError(
                    f"Joint name '{name}' not found in full joint list."
                )
            full_q[name_to_index[name]] = val

        return full_q

    def extract_joint_values(self, full_q, all_joint_names, subset_joint_names):
        """
        Extract values from a full joint vector by matching joint names.

        Args:
            full_q (np.ndarray): full joint vector, shape (N,)
            all_joint_names (List[str]): full joint name list, length N
            subset_joint_names (List[str]): subset of names to extract

        Returns:
            np.ndarray: joint values corresponding to subset_joint_names, shape (len(subset_joint_names),)
        """
        name_to_index = {name: i for i, name in enumerate(all_joint_names)}
        values = []

        for name in subset_joint_names:
            if name not in name_to_index:
                raise ValueError(
                    f"Joint name '{name}' not found in all_joint_names."
                )
            values.append(full_q[name_to_index[name]])

        return np.array(values, dtype=np.float64)

    def pink_cartesian_motion(
        self,
        start,
        goal,
        dt=0.01,
        threshold=1e-4,
        max_iters=3000,
        offset=None,
        constraint_in_goal_frame=True,
        visualize=True,
        return_all=False,
    ) -> np.ndarray:
        """
        Solve Cartesian IK using Pink.

        Args:
            start: Initial joint configuration (np.ndarray).
            goal: Desired end-effector pose (Pose.msg or Array).
            dt: Time step.
            threshold: Position/orientation error tolerance.
            max_iters: Maximum number of iterations.

        Returns:
            Final joint configuration as np.ndarray, or raises RuntimeError.
        """

        # parse start and goal
        start_state = self.parse_joint_state(start)
        goal_pose = self.parse_goal_pose(goal)
        if offset is not None:
            offset = Pose.from_list(offset)
            if constraint_in_goal_frame:
                goal_pose = goal_pose.clone().multiply(offset)
            else:
                goal_pose = offset.clone().multiply(goal_pose.clone())

        # init pin/pink structures
        start_crms = self.motion_gen.kinematics.get_state(start_state.position)
        Hs = start_crms.ee_pose.get_numpy_matrix()[0]
        # start_se3 = pin.SE3(Hs)
        Hg = goal_pose.get_numpy_matrix()[0]
        # goal_se3 = pin.SE3(Hg)
        se3_dist = np.linalg.norm(self.axis_angle(Hs, Hg))
        threshold2 = dt * 5.0
        N = int(se3_dist / threshold2) + 1
        se3_path = [trinterp(Hs, Hg, x) for x in np.linspace(0, 1, N + 1)][1:]
        se3_path.append(Hg)
        print(se3_path, N, len(se3_path), se3_dist)

        pin_names = list(self.pin_model.names[1:])
        pin2act = list(map(pin_names.index, start_state.joint_names))
        q_start = np.zeros(self.pin_model.nq)
        q_start[pin2act] = start_state.position.cpu()
        config = Configuration(self.pin_model, self.pin_data, q_start)

        # setup tasks
        ee_task = FrameTask(self.ee, position_cost=1.0, orientation_cost=1.0)
        ee_task.set_target(pin.SE3(se3_path[0]))
        # posture_task = PostureTask(cost=1.0)
        # posture_task.set_target(start)  # Stay near original config

        # IK loop
        traj = JointTrajectory()
        traj.joint_names = start.name
        in2act = list(map(start.name.index, start_state.joint_names))
        q = np.array(q_start)
        qd = np.zeros_like(q)
        js_dt = 0.0001  # extra fast because will be retimed later
        dur = 0
        arrived = False
        solvers = ['daqp']
        solvers += ['quadprog']
        # solvers = ['osqp']
        # solvers = ['proxqp']
        solvers += ['scs']
        # solvers = list(reversed(qpsolvers.available_solvers))
        # solvers = list(qpsolvers.available_solvers)
        final_solver = None
        while solvers:
            solver = solvers.pop(0)
            final_solver = solver
            print("Trying solver:", solver)
            for i in range(max_iters):
                pos = np.array(start.position)
                vel = np.zeros_like(pos)
                pos[in2act] = q[pin2act]
                vel[in2act] = qd[pin2act]
                traj.points.append(JointTrajectoryPoint())
                traj.points[-1].positions = pos
                traj.points[-1].velocities = vel
                traj.points[-1].time_from_start = rospy.Duration(dur)
                dur += js_dt
                if arrived:
                    print("arrived")
                    solvers = False
                    break

                try:
                    qd = solve_ik(
                        config,
                        [ee_task],
                        dt,
                        damping=1e-9,
                        solver=solver,
                    )
                    print(qd)
                    config.integrate_inplace(qd, dt)
                    config.check_limits()
                except Exception as e:
                    print("No solution found with solver:", solver)
                    ee_task.set_target(pin.SE3(se3_path[-1]))
                    error = np.sum(np.abs(ee_task.compute_error(config)))
                    print("Dist to goal:", error)
                    print(e)
                    if len(solvers) > 0:
                        # reset
                        traj.points.clear()
                        se3_path = [
                            trinterp(Hs, Hg, x)
                            for x in np.linspace(0, 1, N + 1)
                        ][1:]
                        se3_path.append(Hg)
                        ee_task.set_target(pin.SE3(se3_path[0]))
                        config = Configuration(
                            self.pin_model, self.pin_data, q_start
                        )
                        q = np.array(q_start)
                        qd = np.zeros_like(q)
                        dur = 0
                    break

                # Check convergence
                q = config.q
                current = config.get_transform_frame_to_world(self.ee)
                error = np.sum(np.abs(ee_task.compute_error(config)))
                if len(se3_path) > 1 and error < threshold2:
                    ee_task.set_target(pin.SE3(se3_path.pop(0)))
                    # print("Waypoints remaining:", len(se3_path))
                elif len(se3_path) == 1 and error < threshold:
                    arrived = True
                    solvers = False

        print("Iterations:", i, error, threshold)
        stage_probe(
            "pink_cartesian_motion done",
            f"arrived={bool(arrived)} error={float(error)} threshold={float(threshold)} "
            f"waypoints={int(N)} solver={final_solver}",
        )

        assert len(traj.points) > 0, "No trajectory found!"

        if visualize and traj.points:
            self.visualize_traj_rviz(traj)

        if return_all:
            return traj, arrived
        return traj

    def joint_trajectory_from_curobo(
        self,
        plan,
        dt,
        base_state,
        visualize=False,
    ):
        inds = list(map(base_state.name.index, plan.joint_names))
        traj = JointTrajectory()
        traj.joint_names = base_state.name
        positions = np.asarray(plan.position.cpu())
        velocities = np.asarray(plan.velocity.cpu())
        accelerations = np.asarray(plan.acceleration.cpu())
        i = 0
        for p, v, a in zip(positions, velocities, accelerations):
            traj.points.append(JointTrajectoryPoint())
            pos = np.array(base_state.position)
            pos[inds] = p
            if len(base_state.velocity) == 0:
                vel = np.zeros_like(pos)
            else:
                vel = np.array(base_state.velocity)
            vel[inds] = v
            acc = np.zeros_like(pos)
            acc[inds] = a
            traj.points[-1].positions = pos
            traj.points[-1].velocities = vel
            traj.points[-1].accelerations = acc
            traj.points[-1].time_from_start = rospy.Duration(dt * i)
            i += 1

        traj.points.pop(0)
        traj.points[0].time_from_start = rospy.Duration(0.0)
        if visualize:
            self.visualize_traj_rviz(traj)

        return traj

    def visualize_traj_rviz(self, traj):
        while len(traj.points) > 64:
            traj.points = traj.points[::2]
        display = DisplayTrajectory()
        display.trajectory_start.joint_state.name = traj.joint_names
        start_positions = traj.points[0].positions
        display.trajectory_start.joint_state.position = start_positions
        display.trajectory.append(RobotTrajectory())
        display.trajectory[0].joint_trajectory = traj
        self.traj_publisher.publish(display)

    ## based almost exactly on curobo's plan_grasp function
    def plan_grasp_js(
        self,
        start_state: JointState,
        pre_grasp_state: JointState,
        grasp_pose: Pose,
        plan_config: MotionGenPlanConfig,
        grasp_approach_path_constraint: Union[None, List[float]] = [
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.0,
        ],
        retract_offset: Pose = None,  # Pose.from_list([0, 0, -0.15, 1, 0, 0, 0]),
        retract_path_constraint: Union[None, List[float]] = [
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.0,
        ],
        disable_collision_links: List[str] = [],
        plan_approach_to_grasp: bool = True,
        plan_grasp_to_retract: bool = True,
        retract_constraint_in_goal_frame: bool = True,
    ) -> GraspPlanResult:
        """Plan a sequence of motions to grasp an object, given a grasp pose and pre-grasp state.

        This function plans three motions, first approaches the object to the pre grasp state, then
        moves with linear constraints to the grasp pose, and finally retracts the arm base to
        offset with linear constraints. During the linear constrained motions, collision between
        disable_collision_links and the world is disabled. This disabling is useful to enable
        contact between a robot's gripper links and the object.

        Args:
            start_state: Start joint state for planning.
            pre_grasp_state: Pre-grasp joint state.
            grasp_pose: Grasp pose.
            plan_config: Planning parameters for motion generation.
            grasp_approach_path_constraint: Path constraint for the approach to grasp pose and
                grasp to retract path. This is a list of 6 values, where each value is a weight
                for each Cartesian dimension. The first three are for orientation and the last
                three are for position. If None, no path constraint is applied.
            retract_offset: Retract offset pose from grasp pose. Reference frame is the grasp pose
                frame if retract_constraint_in_goal_frame is True, otherwise the reference frame is
                the robot base frame.
            retract_path_constraint: Path constraint for the retract path. This is a list of 6
                values, where each value is a weight for each Cartesian dimension. The first three
                are for orientation and the last three are for position. If None, no path
                constraint is applied.
            disable_collision_links: Name of links to disable collision with the world during
                the approach to grasp and grasp to retract path.
            plan_approach_to_grasp: If True, planning also includes moving from approach to
                grasp. If False, a plan to reach offset of the best grasp pose is returned.
            plan_grasp_to_retract: If True, planning also includes moving from grasp to retract.
                If False, only a plan to reach the best grasp pose is returned.
            retract_constraint_in_goal_frame: If True, the retract offset is in the grasp pose
                frame. If False, the retract offset is in the robot base frame. Also applies to
                retract_path_constraint.

        Returns:
            GraspPlanResult: Result of planning. Use :meth:`GraspPlanResult.grasp_trajectory` to
                get the trajectory to reach the grasp pose and
                :meth:`GraspPlanResult.retract_trajectory` to get the trajectory to retract from
                the grasp pose.
        """
        if plan_config.pose_cost_metric is not None:
            log_error("plan_config.pose_cost_metric should be None")
        result = GraspPlanResult()
        result.success = torch.Tensor([False])

        # plan to pre-grasp:
        reach_offset_mg_result = self.motion_gen.plan_single_js(
            start_state,
            pre_grasp_state,
            plan_config.clone(),
        )
        result.approach_result = reach_offset_mg_result
        if not reach_offset_mg_result.success.item():
            result.status = (
                f"Planning to Approach pose failed: {reach_offset_mg_result.status}"
            )
            return result

        if not plan_approach_to_grasp:
            result.grasp_trajectory = reach_offset_mg_result.optimized_plan
            result.grasp_trajectory_dt = reach_offset_mg_result.optimized_dt
            result.grasp_interpolated_trajectory = (
                reach_offset_mg_result.get_interpolated_plan()
            )
            result.grasp_interpolation_dt = reach_offset_mg_result.interpolation_dt
            return result
        # plan to final grasp
        if grasp_approach_path_constraint is not None:
            hold_pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.tensor_args
                .to_device(grasp_approach_path_constraint),
                project_to_goal_frame=True,
            )
            plan_config.pose_cost_metric = hold_pose_cost_metric

        offset_start_state = reach_offset_mg_result.optimized_plan[
            -1].unsqueeze(0)

        self.motion_gen.toggle_link_collision(disable_collision_links, False)

        reach_grasp_mg_result = self.motion_gen.plan_single(
            offset_start_state,
            grasp_pose,
            plan_config,
        )
        self.motion_gen.toggle_link_collision(disable_collision_links, True)
        result.grasp_result = reach_grasp_mg_result
        if not reach_grasp_mg_result.success.item():
            result.status = f"Planning from Approach to Grasp Failed: {reach_grasp_mg_result.status}"
            return result

        # Get stitched trajectory:

        offset_dt = reach_offset_mg_result.optimized_dt
        grasp_dt = reach_grasp_mg_result.optimized_dt
        if offset_dt > grasp_dt:
            # retime grasp trajectory to match offset trajectory:
            grasp_time_dilation = grasp_dt / offset_dt

            reach_grasp_mg_result.retime_trajectory(
                grasp_time_dilation,
                interpolate_trajectory=True,
            )
        else:
            offset_time_dilation = offset_dt / grasp_dt

            reach_offset_mg_result.retime_trajectory(
                offset_time_dilation,
                interpolate_trajectory=True,
            )

        if (reach_offset_mg_result.optimized_dt -
                reach_grasp_mg_result.optimized_dt).abs() > 0.01:
            reach_offset_mg_result.success[:] = False
            if reach_offset_mg_result.debug_info is None:
                reach_offset_mg_result.debug_info = {}
            reach_offset_mg_result.debug_info["plan_single_grasp_status"] = (
                "Stitching Trajectories Failed"
            )
            return reach_offset_mg_result, None

        result.grasp_trajectory = reach_offset_mg_result.optimized_plan.stack(
            reach_grasp_mg_result.optimized_plan
        ).clone()

        result.grasp_trajectory_dt = reach_offset_mg_result.optimized_dt

        result.grasp_interpolated_trajectory = (
            reach_offset_mg_result.get_interpolated_plan().stack(
                reach_grasp_mg_result.get_interpolated_plan()
            ).clone()
        )
        result.grasp_interpolation_dt = reach_offset_mg_result.interpolation_dt

        # update trajectories in results:
        result.planning_time = (
            reach_offset_mg_result.total_time +
            reach_grasp_mg_result.total_time +
            goalset_motion_gen_result.total_time
        )

        # check if retract path is required:
        result.success[:] = True
        if not plan_grasp_to_retract:
            return result

        result.success[:] = False
        self.motion_gen.toggle_link_collision(disable_collision_links, False)
        grasp_start_state = result.grasp_trajectory[-1].unsqueeze(0)

        # compute retract goal pose:
        if retract_constraint_in_goal_frame:
            retract_goal_pose = grasp_pose.clone().multiply(retract_offset)
        else:
            retract_goal_pose = retract_offset.clone().multiply(
                grasp_pose.clone()
            )

        # add path constraint for retract:
        plan_config.pose_cost_metric = None

        if retract_path_constraint is not None:
            hold_pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.tensor_args
                .to_device(retract_path_constraint),
                project_to_goal_frame=retract_constraint_in_goal_frame,
            )
            plan_config.pose_cost_metric = hold_pose_cost_metric

        # plan from grasp pose to retract:
        retract_grasp_mg_result = self.motion_gen.plan_single(
            grasp_start_state,
            retract_goal_pose,
            plan_config,
        )
        self.motion_gen.toggle_link_collision(disable_collision_links, True)
        result.planning_time += retract_grasp_mg_result.total_time
        if not retract_grasp_mg_result.success.item():
            result.status = (
                f"Retract from Grasp failed: {retract_grasp_mg_result.status}"
            )
            result.retract_result = retract_grasp_mg_result
            return result
        result.success[:] = True

        result.retract_trajectory = retract_grasp_mg_result.optimized_plan
        result.retract_trajectory_dt = retract_grasp_mg_result.optimized_dt
        result.retract_interpolated_trajectory = (
            retract_grasp_mg_result.get_interpolated_plan()
        )
        result.retract_interpolation_dt = retract_grasp_mg_result.interpolation_dt

        return result
