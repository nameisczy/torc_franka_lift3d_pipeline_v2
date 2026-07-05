import time
import copy
import torch
import numpy as np
import open3d as o3d
import trimesh as tm
from scipy.spatial import KDTree
from tracikpy import TracIKSolver

# from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.base import TensorDeviceType
from curobo.geom.types import WorldConfig, Mesh
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.geom.sdf.world import CollisionQueryBuffer

import transformations as tf

import rospy
import rosservice

OCCL_POINT_CLOUD_NAME = "occlusion"


class VisibilityPlanner:

    def __init__(self, curobo_config, world_config, urdf_file, resolution=0.005):
        # world_config = "/common/home/kc1317/Robotics/curobo/src/curobo/content/configs/world/collision_empty.yml"
        # curobo_config = "/data/local/kc1317/workspace/src/lab_vbnpm/robots/motoman/curobo/motoman_no_collision.yml"
        self.resolution = resolution
        self.curobo_config: RobotConfig = curobo_config
        self.world_config: WorldConfig = world_config

        self.tensor_args = TensorDeviceType()

        self.weight = torch.tensor([1.0], device=self.tensor_args.device)
        self.activation_distance = torch.tensor([0.0], device=self.tensor_args.device)
        # self.disable_ee_links = ignore_collision_ee_links

        robot_config = RobotWorldConfig.load_from_config(
            self.curobo_config,
            self.world_config,
            collision_activation_distance=0.0,
        )
        self.robot_config: RobotWorldConfig = robot_config

        base_link = robot_config.kinematics.base_link
        ee_link = robot_config.kinematics.ee_link
        # self.disable_ee_links.append(ee_link)

        # init global structures
        self.robot_world = RobotWorld(robot_config)
        self.kinematics_config = self.robot_world.kinematics.kinematics_config
        # input(self.robot_world.kinematics.robot_spheres[-2-max_s:])
        self.ik_solver = TracIKSolver(urdf_file, base_link, ee_link)

        # self.disable_link_collision()
        for obs_name in self.robot_world.world_model.get_obstacle_names():
            if obs_name is None:
                continue
            if "shelf" in obs_name:
                self.robot_world.world_model.enable_obstacle(obs_name, True)
                print(obs_name)

        self.buffer = CollisionQueryBuffer.initialize_from_shape(
            (0, 0, 0, 0),
            self.tensor_args,
            collision_types=self.robot_world.world_model.collision_types,
        )

        print("Obstacle names")
        print("CuRobo collision types:", self.robot_world.world_model.collision_types)
        # print(self.buffer.mesh_collision_buffer, self.buffer.primitive_collision_buffer)
        # self.robot_world.world_model.disable
        # raise KeyboardInterrupt()
        # self.robot_world.world_model.enable_obstacle()
        self.disable_all_collisions()

    def disable_link_collision(self):
        collision_link_names = [
            "torso_base_link",
            "torso_link_b1",
            "arm_left_link_1_s",
            "arm_right_link_1_s",
            "arm_left_link_2_l",
            "arm_right_link_2_l",
            "arm_left_link_3_e",
            "arm_right_link_3_e",
            "arm_left_link_4_u",
            "arm_right_link_4_u",
            "arm_left_link_5_r",
            "arm_right_link_5_r",
            "arm_left_link_6_b",
            "arm_right_link_6_b",
            "arm_left_link_7_t",
            "arm_right_link_7_t",
            "robotiq_arg2f_base_link",
            "left_outer_finger",
            "right_outer_finger",
            "right_inner_finger",
            "left_inner_finger",
            "motoman_right_ee",
        ]
        if len(collision_link_names) > 0:
            for k in collision_link_names:
                # self.kinematics_config.disable_link_spheres(k)
                self.robot_config.kinematics.kinematics_config.disable_link_spheres(k)

    def disable_all_collisions(self):
        self.disable_link_collision()

        for obs_name in self.robot_world.world_model.get_obstacle_names():
            self.robot_world.world_model.enable_obstacle(obs_name, False)

    def update_world(self, world_config):
        self.world_config = world_config
        self.robot_world.clear_world_cache()
        self.robot_world.update_world(world_config)
        # pass

    def set_collision_scene(self, points, filter_points=None):
        # init world config
        world_config = WorldConfig.from_dict(self.world_config)

        # filter collision points by proximity to filter points
        if filter_points is not None:
            # kdtree = KDTree(filter_points[:, :2])
            # dist, ind = kdtree.query(points[:, :2])
            surface, find = tm.sample.sample_surface_even(
                tm.convex.convex_hull(filter_points), 10000
            )
            kdtree = KDTree(surface)
            dist, ind = kdtree.query(points)
            points = points[dist > 0.05]

        # convert point cloud to mesh
        scene_mesh = Mesh.from_pointcloud(points, self.resolution, "world")

        # add to world config
        world_config.add_obstacle(scene_mesh)

        # update collision world
        self.robot_world.clear_world_cache()
        self.robot_world.update_world(world_config)

    def raytrace_single(self, x1, x2, length=1, spheres=64, r=0.01):
        """
        Use CuRobo sphere collision-checking to perform ray-object intersection.

        x1 is of shape (3,)
        x2 is of shape (3,)
        """

        # Note: we use None instead of torch.newaxis or np.newaxis
        # This is because, in raytrace_single, we do not care if x2 is a tensor or array
        return self.raytrace_batch(x1, x2[None, :], length=length, spheres=spheres, r=r)

    def raytrace_batch(self, x1, x2_batch, spheres=64, r=6.0, true_r=0.02):
        """
        Use CuRobo sphere collision-checking to perform ray-object intersection.

        x1 is of shape (3,)
        x2 is of shape (batch, 3)
        """
        r = 100.0
        # print("x1&2.shape", x1.shape, x2_batch.shape) #[3], [batch, 3]

        num_rays = x2_batch.shape[0]

        # Send x1, x2 to GPU
        x1 = torch.tensor(
            x1, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        x2_batch = torch.tensor(
            x2_batch, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )

        # Ensure shapes are correct for broadcasting
        # (batch, 3)
        if len(x1.shape) == 1:
            x1 = x1.unsqueeze(0)

        # Calculate the endpoint via broadcasting
        # endpoint is of shape (batch, 3)
        # normalizing_constants = length / torch.linalg.norm(x2_batch - x1, dim=1, keepdim=True)
        # #print("normalizing constants shape", normalizing_constants.shape) #[batch, 1]
        # endpoint = x1 + normalizing_constants * (x2_batch - x1) #The subtracting is the direction, the normalizing constant scales the direction then adds to x1
        endpoint = x2_batch

        # Create t_values tensor for linear interpolation
        # t_values is of shape (spheres,)
        t_values = torch.linspace(
            0, 1, spheres, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )

        # Ensure shapes are correct for broadcasting
        # (batch, spheres, 3)
        endpoint = endpoint[:, torch.newaxis, :]
        x1 = x1[:, torch.newaxis, :]
        t_values = t_values[torch.newaxis, :, torch.newaxis]

        # Calculate the centers of the spheres
        centers = x1 + t_values * (endpoint - x1)

        # Create radius column tensor
        radius_column = torch.full(
            (num_rays, spheres, 1),
            r,
            dtype=self.tensor_args.dtype,
            device=self.tensor_args.device,
        )

        # Concatenate sphere centers with sphere radius column for CuRobo query
        # Add a dimension so that shape is (batch, horizon=1, spheres, 4) for CuRobo query
        # Ensure contiguous memory (required for CuRobo collision query)
        query_spheres = torch.cat([centers, radius_column], dim=-1)[
            :, torch.newaxis, ...
        ].contiguous()

        # Update buffer size for CuRobo query
        # TODO: Confirm that the if-statement is the correct way to check buffer shape
        if self.buffer.shape != (num_rays, 1, spheres, 4):
            self.buffer.update_buffer_shape(
                (num_rays, 1, spheres, 4),
                self.tensor_args,
                self.robot_world.world_model.collision_types,
            )

        # Collision-checking query
        # dist is of shape (batch, 1, spheres)
        # dist = self.robot_world.world_model.get_sphere_collision(query_sphere=query_spheres,
        dist = self.robot_world.world_model.get_sphere_distance(
            query_sphere=query_spheres,
            collision_query_buffer=self.buffer,
            weight=self.weight,
            activation_distance=self.activation_distance,
            compute_esdf=True,
        )

        # They output 0's if it doesn't collide.
        # It outputs the distance from the sphere * -1. The lower value it outputs, the less in collision.
        # My guess is that it is higher abs value when farther away, so lower value is worse in collision
        # Since it doesn't change based on the radius, I think this is just distance from center of sphere to nearest collision

        # invalid_rays is of shape (batch,)
        # print("dist shape", dist.shape, dist[0], dist) #[rays, 1 ,spheres]
        invalid_rays = (dist > 0).any(dim=(1, 2))
        # print(invalid_rays.shape)
        # return ~invalid_rays

        # Distance to collision:
        dist = torch.abs(dist)
        dist_to_col = (
            dist  # r - dist #[rays, 1, spheres], needs to in the end be [rays]
        )
        # min distance meaning the closest to collision
        dist_to_col_min, _ = torch.min(
            dist_to_col, dim=-1
        )  # Get the worst one so the lowest dist
        dist_to_col_min = dist_to_col_min.squeeze()
        new_invalid_rays = (
            dist_to_col_min < true_r
        )  # [rays] when the distance to the collision is less than what is allowed
        # print(dist_to_col_min[2])

        # Largest cone:
        largest_cone_dist, largest_cone_ind = torch.max(
            dist_to_col_min, dim=0
        )  # Get the best out of the dists
        print("largest cone ind", largest_cone_ind, largest_cone_dist)
        largest_cone_ray = x2_batch[largest_cone_ind, :]  # [3]
        # print("largest cone ray", largest_cone_ray.shape)
        # print("largest cone dist", largest_cone_dist)

        return (
            ~new_invalid_rays,
            largest_cone_ray,
            dist_to_col_min,
            largest_cone_ind,
        )  # dist to col means cone size

    def batch_query_test(self, tgt_pts, num_rays=10, length=1, spheres=64, r=0.01):
        # centroid = np.mean(tgt_pts)
        centroid = np.array([1.05197, -0.219925, 1.03373])
        centroid[2] += 0.1

        valid_ray_poses = []
        invalid_ray_poses = []

        # Sample random ray directions
        ray_vertex_poses = []
        x2_batch = []
        unit_sphere_offsets = []

        for i in range(num_rays):

            random_orientation = sample_rotation_matrix()

            unit_sphere_offset = (random_orientation @ np.array([0, 0, length, 1]))[:3]
            random_orientation[:3, 3] = centroid

            ray_vertex_poses.append(random_orientation)
            x2_batch.append(unit_sphere_offset)
            unit_sphere_offsets.append(unit_sphere_offset)

        # Concatenate ray endpoints for CuRobo computations
        # x2_batch is of shape (num_rays, 3)
        x2_batch = np.array(x2_batch)

        # Perform batch raytracing
        # res should be of shape (num_rays, )
        start_time = time.time()
        res = self.raytrace_batch(
            centroid, centroid + x2_batch, length=length, spheres=spheres, r=r
        )
        print("[k_user] Raytrace batch query took", time.time() - start_time, "seconds")

        # NOTE: everything before this for loop works (as the iterative test works properly)
        for ray_result, ray_vertex_pose, unit_sphere_offset in zip(
            res, ray_vertex_poses, unit_sphere_offsets
        ):
            # sanity_check = self.raytrace_single(centroid, centroid+unit_sphere_offset,
            #                                     length=length,
            #                                     spheres=spheres,
            #                                     r=r)
            # if sanity_check != ray_result:
            #     pass
            #     print("[k_user] Sanity check failed for raytracing results")
            #     # raise ValueError("[k_user] Sanity check failed for raytracing results")
            if ray_result:
                valid_ray_poses.append(ray_vertex_pose)
            else:
                invalid_ray_poses.append(ray_vertex_pose)

        print("[k_user] RAYTRACING TEST")
        return valid_ray_poses, invalid_ray_poses


def sample_quaternion():
    """
    Implementation taken from "Effective Sampling and Distance Metrics for 3D Rigid Body Path Planning" (Kuffner 2004)
    """

    s = np.random.rand()

    sigma_1 = np.sqrt(1 - s)
    sigma_2 = np.sqrt(s)

    theta_1 = 2 * np.pi * np.random.rand()
    theta_2 = 2 * np.pi * np.random.rand()

    w = np.cos(theta_2) * sigma_2
    x = np.sin(theta_1) * sigma_1
    y = np.cos(theta_1) * sigma_1
    z = np.sin(theta_2) * sigma_2

    return np.array([w, x, y, z])


def sample_rotation_matrix():
    return tf.quaternion_matrix(sample_quaternion())
