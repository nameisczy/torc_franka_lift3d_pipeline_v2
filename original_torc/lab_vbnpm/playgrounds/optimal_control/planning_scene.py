"""
specify the planning scene for collision checking and Jacobian computation.
implemented by mujoco (for the robot only), hpp-fcl, open3d (visualization)
"""

from robot import Robot
import hppfcl
import numpy as np
import open3d as o3d
from typing import Union
import numpy.typing as npt
import copy

class PlanningScene:
    """
    describe the planning scene which includes the robot and a dynamic environment.
    the robot is loaded by mujoco model and hpp-fcl collision objects.
    the environment is described by point clouds. The collision object is through octree of hpp-fcl
    TODO: could also combine with geometric primitives such as boxes for the static scene.
    """
    def __init__(self, robot: Robot, scene_pcd: Union[npt.NDArray[np.float64], npt.NDArray[np.float32]]=None, octree_res: float=0.01):
        """
        initialize the planning scene
        :param robot: the robot object
        :param scene_pcd: the point cloud of the scene. When None, it is initialized as an empty point cloud.
        """
        self.robot = robot
        self.octree_res = octree_res
        if scene_pcd is None:
            scene_pcd = np.zeros((0,3))
        octree_geom = hppfcl.makeOctree(scene_pcd, octree_res)
        octree_obj = hppfcl.CollisionObject(octree_geom)
        self.octree_geom = octree_geom
        self.octree_obj = octree_obj
        self.scene_pcd = scene_pcd
        self.o3d_pcd = o3d.geometry.PointCloud()
        self.o3d_pcd.points = o3d.utility.Vector3dVector(scene_pcd)
        self.col_pair_num = self._compute_col_pair_num()

    def _compute_col_pair_num(self):
        """
        compute the total number of collision pairs, including self collision and collision with env
        """
        col_pair_num = self.robot.col_pair_num
        for link in self.robot.robot_link_names:
            col_pair_num += len(self.robot.robot_link_name_to_fcl_objs[link])
        return col_pair_num

    def update_scene_pcd(self, scene_pcd: Union[npt.NDArray[np.float64], npt.NDArray[np.float32]]):
        """
        update the scene point cloud
        :param scene_pcd: the new scene point cloud
        """
        self.octree_geom = hppfcl.makeOctree(scene_pcd, self.octree_res)
        # self.octree_obj.setCollisionGeometry(self.octree_geom, True)
        self.octree_obj = hppfcl.CollisionObject(self.octree_geom)
        self.scene_pcd = scene_pcd

    def compute_distance_total(self, dist_margin: float = 0.001, full: bool = False) -> list:
        # * self distance
        self_distance_results = self.robot.compute_distance_total(dist_margin)
        # * scene distance
        distance_results = []
        for link in self.robot.robot_link_names:
            for obj_i in range(len(self.robot.robot_link_name_to_fcl_objs[link])):
                dis_result = hppfcl.DistanceResult()
                self.robot.robot_link_to_distance_req[(link,obj_i)].enable_nearest_points = True
                distance = hppfcl.distance(self.robot.robot_link_name_to_fcl_objs[link][obj_i], self.octree_obj,
                                           self.robot.robot_link_to_distance_req[(link,obj_i)],
                                           dis_result)
                                        #    self.robot.robot_link_to_distance_res[(link,obj_i)])
                if distance < dist_margin:
                    distance_result = dis_result
                    # distance_result = self.robot.robot_link_to_distance_res[(link,obj_i)]
                    distance_results.append((link, "scene", obj_i, 0, distance_result))
                else:
                    if full:
                        distance_result = dis_result
                        # distance_result = self.robot.robot_link_to_distance_res[(link,obj_i)]
                        distance_results.append((link, "scene", obj_i, 0, distance_result))
        return self_distance_results + distance_results

    def compute_collision_total(self, dist_upper_bound: float = 0.001, security_margin: float = 0.001, full: bool = False) -> list:
        """
        compute the total collision in the planning scene, including self collision and collision with the scene
        the scene collision is named as "scene"
        :return: the total collision results
        """
        # * self collision
        self_collision_results = self.robot.compute_collision_total(security_margin, full=full)
        # * scene collision
        collision_results = []  # store the results of the collision, link1, link2, geom1_i, geom2_i, collision result
        for link in self.robot.robot_link_names:
            for obj_i in range(len(self.robot.robot_link_name_to_fcl_objs[link])):
                col_result = hppfcl.CollisionResult()
                self.robot.robot_link_to_collision_req[(link,obj_i)].distance_upper_bound = dist_upper_bound
                self.robot.robot_link_to_collision_req[(link,obj_i)].security_margin = security_margin
                collision = hppfcl.collide(self.robot.robot_link_name_to_fcl_objs[link][obj_i], self.octree_obj,
                                           self.robot.robot_link_to_collision_req[(link,obj_i)],
                                           col_result)
                                        #    self.robot.robot_link_to_collision_res[(link,obj_i)])
                # NOTE: collision with the environment (octree/octomap) is using robot.robot_link_to_collision_req
                #       self collision is using robot.collision_pair_to_collision_req
                # TODO: renaming robot.robot_link_to_colllision_res to be more understandable and consistent with self_collision or collision with env

                if collision:
                    collision_result = col_result#self.robot.robot_link_to_collision_res[(link,obj_i)]
                    collision_results.append((link, "scene", obj_i, 0, collision_result))
                else:
                    if full:
                        collision_result = col_result#self.robot.robot_link_to_collision_res[(link,obj_i)]
                        collision_results.append((link, "scene", obj_i, 0, collision_result))
                    
        return self_collision_results + collision_results


    def compute_collision_min_dist_total(self, dist_upper_bound: float = 0.001, security_margin: float = 0.001, full: bool = False) -> list:
        """
        compute the total collision in the planning scene, including self collision and collision with the scene
        the scene collision is named as "scene"
        :return: the total collision results
        """
        # * self collision
        self_distance_results = self.robot.compute_collision_min_dist_total(dist_upper_bound=dist_upper_bound, security_margin=security_margin, full=full)
        # * scene collision
        distance_results = []  # store the results of the collision, link1, link2, geom1_i, geom2_i, collision result
        for link in self.robot.robot_link_names:
            for obj_i in range(len(self.robot.robot_link_name_to_fcl_objs[link])):
                col_result = hppfcl.CollisionResult()
                self.robot.robot_link_to_collision_req[(link,obj_i)].distance_upper_bound = dist_upper_bound
                self.robot.robot_link_to_collision_req[(link,obj_i)].security_margin = dist_upper_bound
                collision = hppfcl.collide(self.robot.robot_link_name_to_fcl_objs[link][obj_i], self.octree_obj,
                                           self.robot.robot_link_to_collision_req[(link,obj_i)],
                                           col_result)
                                        #    self.robot.robot_link_to_collision_res[(link,obj_i)])
                # NOTE: collision with the environment (octree/octomap) is using robot.robot_link_to_collision_req
                #       self collision is using robot.collision_pair_to_collision_req
                # TODO: renaming robot.robot_link_to_colllision_res to be more understandable and consistent with self_collision or collision with env

                if collision:
                    # * compute the min distance
                    distance_result = hppfcl.DistanceResult()
                    self.robot.robot_link_to_distance_req[(link,obj_i)].enable_nearest_points = True
                    distance = hppfcl.distance(self.robot.robot_link_name_to_fcl_objs[link][obj_i], self.octree_obj,
                                                self.robot.robot_link_to_distance_req[(link,obj_i)],
                                                distance_result)
                    p1 = distance_result.getNearestPoint1()
                    p2 = distance_result.getNearestPoint2()
                    normal = p2 - p1
                    normal = normal / np.linalg.norm(normal)
                    ret_result = {'p1': p1, 'p2': p2, 'normal': normal, 'distance': distance}
                    # if distance < security margin, then it means in collision. In this case, we use the collision
                    # checker to get the distance results, since the nearest points do not provide enough information.
                    # the distance result does not return the center point. So we need to do it ourselves.
                    if distance < security_margin:
                        col_result = hppfcl.CollisionResult()
                        self.robot.robot_link_to_collision_req[(link,obj_i)].distance_upper_bound = dist_upper_bound
                        self.robot.robot_link_to_collision_req[(link,obj_i)].security_margin = security_margin
                        collision = hppfcl.collide(self.robot.robot_link_name_to_fcl_objs[link][obj_i], self.octree_obj,
                                                self.robot.robot_link_to_collision_req[(link,obj_i)],
                                                col_result)
                        contact = col_result.getContact(0)
                        normal = contact.normal
                        normal = normal / np.linalg.norm(normal)
                        ret_result['normal'] = normal
                        pos = contact.pos
                        p1 = pos - normal * 0.5 * (-contact.penetration_depth)
                        p2 = pos + normal * 0.5 * (-contact.penetration_depth)                            
                        ret_result['p1'] = p1
                        ret_result['p2'] = p2
                        ret_result['distance'] = -contact.penetration_depth
                    distance_results.append((link, "scene", obj_i, 0, ret_result))
                else:
                    if full:
                        distance_result = hppfcl.DistanceResult()#self.robot.robot_link_to_collision_res[(link,obj_i)]
                        ret_result = {'distance': dist_upper_bound + 1e-1}
                        distance_results.append((link, "scene", obj_i, 0, ret_result))
        return self_distance_results + distance_results



    def visualize(self, show=True) -> list:
        o3d_objs = self.robot.visualize(show=False)
        self.o3d_pcd.points = o3d.utility.Vector3dVector(self.scene_pcd)
        o3d_objs.append(self.o3d_pcd)
        if show:
            o3d.visualization.draw_geometries(o3d_objs)
        return copy.deepcopy(o3d_objs)