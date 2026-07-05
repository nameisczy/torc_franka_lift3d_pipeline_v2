import time
import copy
import trimesh
import numpy as np
import open3d as o3d
from math import log2
from urchin import URDF
from scipy.spatial import KDTree

import rospy
import rospkg
import rosservice

import tf2_ros

# import tf2_geometry_msgs
from std_msgs.msg import Header
import moveit_msgs.msg as mit
import bio_ik_msgs.msg as bik
from bio_ik_msgs.srv import GetIK
from sensor_msgs.msg import JointState
import sensor_msgs.point_cloud2 as pcl2
from trajectory_msgs.msg import JointTrajectoryPoint

from utils.print_color import *
from utils.conversions import pose_to_matrix, matrix_to_pose

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, PoseStamped
from gpd_docker.srv import (
    GetGrasps,
)  # , ReScoreGrasps #[j_user DEBUG] Temporarily this is not available for my code so I commented it out

# from gpg_ros.srv import GetGrasps
# from graspnet_ros.srv import GetGrasps


class GraspPlanner:

    def __init__(
        self,
        ee_scale=1.2,
        gripper_config_file="robotiq_gripper_params.cfg",
        gripper_group="arm_right_bioik",
        ee="motoman_right_ee",
    ):
        rospack = rospkg.RosPack()
        self.gripper_cfg = rospack.get_path("lab_vbnpm")
        self.gripper_cfg += "/scripts/grasp_planner/"
        self.gripper_cfg += gripper_config_file

        self.grasp_srvs = sorted(
            filter(lambda x: x.find("get_grasps") >= 0, rosservice.get_service_list())
        )

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(60))
        self.tf_listen = tf2_ros.TransformListener(self.tf_buffer)

        self.ik_req = bik.IKRequest()
        self.ik_req.timeout = rospy.Duration(0.02)
        self.ik_req.group_name = gripper_group
        self.ik_req.avoid_collisions = True
        self.ik_req.approximate = False
        self.ik_req.avoid_joint_limits_goals.append(bik.AvoidJointLimitsGoal())
        self.ik_req.avoid_joint_limits_goals[-1].weight = 1.0
        self.ik_req.avoid_joint_limits_goals[-1].primary = False
        self.ik_req.pose_goals.append(bik.PoseGoal())
        self.ik_req.pose_goals[-1].link_name = ee
        self.ik_req.pose_goals[-1].weight = 10
        self.ee_name = ee
        try:
            joint_state = rospy.wait_for_message(
                "/joint_states_all", JointState, timeout=1
            )
            self.ik_req.robot_state.joint_state = joint_state
        except rospy.ROSException as e:
            # print('Failed to get joint states:', e)
            pass

        # TODO read gripper_cfg for gripper params
        # distance to motoman_right_ee not to tip as in cfg
        self.hand_depth = 0.05  # 55 / 2

        # collision filtering init; TODO use parameters!
        urdf_path = rospack.get_path("lab_vbnpm")
        urdf_path += "/robots/robotiq_arg85_description.URDF"
        self.gripper = URDF.load(urdf_path)
        links = [
            "robotiq_85_base_link",
            "left_outer_knuckle",
            "left_outer_finger",
            "left_inner_knuckle",
            "left_inner_finger",
            "right_inner_knuckle",
            "right_inner_finger",
            "right_outer_knuckle",
            "right_outer_finger",
        ]
        # sort links
        self.g_links = list(self.gripper.link_fk(use_names=True, links=links).keys())
        self.geoms = self.gripper.collision_trimesh_fk(links=self.g_links)
        self.scene = trimesh.scene.Scene()
        for link, geometry_transform in zip(self.g_links, self.geoms.items()):
            geometry, transform = geometry_transform
            geometry.apply_transform(transform)
            self.scene.add_geometry(
                geometry,
                node_name=link,
                geom_name=link,
                # transform=transform,
            )
        ee_mesh = trimesh.util.concatenate(self.scene.geometry.values())
        # trimesh.scene.Scene([ee_mesh,ee_mesh.copy().apply_transform(ee_scale * np.eye(4))]).show()
        # ee_mesh.apply_transform(ee_scale * np.eye(4))
        ee_mesh.apply_transform(
            [
                [0.0, -1, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -0.135],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.ee_points = trimesh.sample.sample_surface_even(ee_mesh, 1000)[0]
        # self.cmanager, cobjects = trimesh.collision.scene_to_collision(scene)

    def gripper_in_collision(self, gripper_pose, voxels, visualize=False):
        points = copy.deepcopy(self.ee_points)
        points = np.matmul(points, gripper_pose.T[:3, :3]) + gripper_pose[:3, 3]
        points = o3d.utility.Vector3dVector(points)
        if visualize:
            pcl = o3d.geometry.PointCloud()
            pcl.points = points
            o3d.visualization.draw_geometries([voxels, pcl])
        mask_collision = np.array(voxels.check_if_included(points))
        # return mask_collision.any()
        # return number of points in collision
        return np.count_nonzero(mask_collision)

    def gripper_static_collision(self, gripper_pose, s_height, visualize=False):
        points = copy.deepcopy(self.ee_points)
        points = np.matmul(points, gripper_pose.T[:3, :3]) + gripper_pose[:3, 3]
        if visualize:
            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(points)
            box = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 0.01)
            x, y = np.mean(points[:, 0]), np.mean(points[:, 1])
            box.translate((x, y, s_height))
            print(s_height)
            self.ik_solve(matrix_to_pose(gripper_pose), visualize=True)
            o3d.visualization.draw_geometries([pcl, box])
        mask_collision = points[:, 2] < s_height
        return mask_collision.any()

    def gripper_collision(
        self, gripper_pose, static_only, s_height, voxels, point_cloud=None
    ):
        #         for a in range(10): printPurple("ORIGINAL GRIPPER COLLISION")
        if static_only:
            val = self.gripper_static_collision(gripper_pose, s_height)
            # self.gripper_in_collision(gripper_pose, voxels)
            return val
        return self.gripper_in_collision(gripper_pose, voxels) > 0

    # Empty method that is overriden in GraspPlannerHPPFCL when maintaining point cloud
    def update_point_cloud(self, point_cloud):
        pass

    def collision_check(self, gripper_pose, static_only, s_height, voxels, point_cloud):
        return self.gripper_collision(
            gripper_pose=gripper_pose,
            static_only=static_only,
            s_height=s_height,
            voxels=voxels,
            point_cloud=point_cloud,
        )

    def show_ik(self, sol):
        js = sol.joint_state
        d = mit.DisplayTrajectory()
        d.trajectory_start = sol
        d.trajectory.append(mit.RobotTrajectory())
        d.trajectory[0].joint_trajectory.joint_names = js.name
        d.trajectory[0].joint_trajectory.points.append(JointTrajectoryPoint())
        d.trajectory[0].joint_trajectory.points[-1].positions = js.position
        # d.trajectory[0].joint_trajectory.points[-1].time_from_start = rospy.Duration(0)
        display_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            mit.DisplayTrajectory,
            latch=True,
            queue_size=10,
        )
        display_publisher.publish(d)

    def ik_collision(self, gripper_pose, ee=None, visualize=False):
        self.ik_req.pose_goals[-1].pose = gripper_pose
        if ee is not None:
            self.ik_req.pose_goals[-1].link_name = ee
        else:
            self.ik_req.pose_goals[-1].link_name = self.ee_name
        rospy.wait_for_service("/bio_ik/get_bio_ik")
        get_bio_ik = rospy.ServiceProxy("/bio_ik/get_bio_ik", GetIK)
        response = get_bio_ik(self.ik_req).ik_response
        # response.solution
        if visualize:
            self.show_ik(response.solution)
        return response.error_code.val != 1

    def ik_solve(self, gripper_pose, ee=None, visualize=False):
        self.ik_req.pose_goals[-1].pose = gripper_pose
        if ee is not None:
            self.ik_req.pose_goals[-1].link_name = ee
        else:
            self.ik_req.pose_goals[-1].link_name = self.ee_name
        rospy.wait_for_service("/bio_ik/get_bio_ik")
        get_bio_ik = rospy.ServiceProxy("/bio_ik/get_bio_ik", GetIK)
        response = get_bio_ik(self.ik_req).ik_response
        # response.solution
        if visualize:
            self.show_ik(response.solution)
        return response.solution.joint_state, response.error_code.val != 1

    def get_transform(self, target_frame, source_frame):
        if target_frame == source_frame:
            return None
        try:
            in2out = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rospy.Time(),
                rospy.Duration(1.0),
            )
            return in2out
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            print(
                f"""
    Warning: Couldn't get transform from [{source_frame}] to [{target_frame}]!"
    Returning grasps in frame [{source_frame}]
                  """
            )
            return None

    def make_pre_grasp(self, pose_w, pre_grasp_dist):
        pose_pre = copy.deepcopy(pose_w)
        approach = pose_pre[:3, 2] / np.linalg.norm(pose_pre[:3, 2])
        displace = self.hand_depth - pre_grasp_dist
        pose_pre[:3, 3] += displace * approach
        return pose_pre
        # return matrix_to_pose(pose_pre)

    @staticmethod
    def normalize_score(scores):
        return 1 / (1 + np.exp(-np.divide(scores, 2000)))

    def get_grasp_poses(
        self,
        points,
        colors,
        ik_func=None,
        input_frame="world",
        output_frame="world",
        collision_voxel_tuple=None,
        pre_grasp_res=0.04,
        pre_grasp_range=[0.02, 0.1],
        filter_outliers=False,
        visualize=False,
        octomap_set=None,
        octomap_reset=None,
    ):
        if len(points) == 0:
            return [], [], [], []

        # ensure points and ppoints are correct formats
        if type(points[0]) is Point:
            ppoints = points
            points = np.array([[p.x, p.y, p.z] for p in points])
        else:
            ppoints = [Point(x, y, z) for x, y, z in points]

        # get transform from input frame to output frame
        in2out = self.get_transform(output_frame, input_frame)

        # target point cloud?
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        pcl.colors = o3d.utility.Vector3dVector(colors)
        if filter_outliers:
            pcl, _removed = pcl.remove_statistical_outlier(*filter_outliers)
        points = pcl.points
        colors = pcl.colors
        geom_to_draw = [pcl]

        if collision_voxel_tuple is not None:
            # vx_collision, vx_resolution, vx_pose, vx_frame_id = collision_voxel_tuple
            # tm_vx_collision = trimesh.voxel.VoxelGrid(vx_collision)
            vis_points, collision_points, vx_resolution = collision_voxel_tuple
            c_pcl = o3d.geometry.PointCloud()
            c_pcl.points = o3d.utility.Vector3dVector(collision_points)
            tm_vx_collision = o3d.geometry.VoxelGrid.create_from_point_cloud(
                c_pcl, vx_resolution
            )
            geom_to_draw.append(tm_vx_collision)
            if collision_points.shape[0] != 0:
                z_min = np.min(collision_points[:, 2])
            else:
                z_min = np.iinfo(np.int32).min

            self.update_point_cloud(collision_points)
        else:
            vis_points = points
            if collision_points.shape[0] != 0:
                z_min = np.min(np.array(points)[:, 2])
            else:
                z_min = np.iinfo(np.int32).min

        ## sample surface plane points of workspace
        lx, ly = np.min(vis_points[:, 0]), np.min(vis_points[:, 1])
        hx, hy = np.max(vis_points[:, 0]), np.max(vis_points[:, 1])
        surface_plane = np.random.uniform(
            [lx, ly, z_min], [hx, hy, z_min], size=(10000, 3)
        )
        vis_points = np.concatenate([vis_points, surface_plane], axis=0)

        if visualize:
            o3d.visualization.draw_geometries(geom_to_draw)
            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(vis_points)
            # pcl.colors = o3d.utility.Vector3dVector(np.concatenate())
            o3d.visualization.draw_geometries([pcl])

        ## init grasp service request
        ccolors = [ColorRGBA(r, g, b, 0) for r, g, b in colors]
        header = Header()
        header.frame_id = input_frame
        cloud = pcl2.create_cloud_xyz32(header, vis_points)
        if in2out is not None:
            camera_position = in2out.transform.translation
        else:
            # camera_position = Point(0.14, 0, 1.4)
            camera_position = Point(0, 0, 0)

        ## call service
        poses = []
        scores = []
        samples = []
        for srv_name in self.grasp_srvs:
            rospy.wait_for_service(srv_name, timeout=10)
            try:
                service = rospy.ServiceProxy(srv_name, GetGrasps)
                resp = service(ppoints, cloud, camera_position)  # gpd call
                # resp = service(ppoints, input_frame)  # gpg call
                # resp = service(ppoints, ccolors) # graspnet call
                poses.extend(resp.grasps.poses)
                norm_scores = self.normalize_score(resp.grasps.scores)
                # if srv_name.find('large') >= 0:
                #     norm_scores += 1
                scores.extend(norm_scores)
                samples.extend(resp.grasps.samples)
                # if len(poses) > 0:
                #     break
            except rospy.ServiceException as e:
                print("Service call failed:", e)
                # return [], [], [], []

        if len(poses) == 0:
            print("No grasps found!")
            return [], [], [], []
        # print(len(poses), 'grasps before filtering')

        def grasp_invalid(pose_mat, static_only=False):
            if ik_func(pose_mat) is None:
                return True
            if collision_voxel_tuple is None:
                return False
            return self.collision_check(
                pose_mat, static_only, z_min, tm_vx_collision, collision_points
            )
            if static_only:
                return self.gripper_static_collision(pose_mat, z_min)
            return self.gripper_in_collision(pose_mat, tm_vx_collision) > 0

        pose_list = []
        pre_grasp_list = []
        pose_list_scores = []
        sample_list = []
        n_ikc = 0.0001
        t_ikc = 0
        n_cc = 0.0001
        t_cc = 0
        n_pp = 0.0001
        t_pp = 0
        # score_threshold = np.percentile(normalized_scores, 90)
        # print('score_threshold:', score_threshold)
        for pose, score, sample in zip(poses, scores, samples):
            # transform pose to output frame
            if in2out is not None:
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = input_frame
                pose_stamped.pose = pose
                pose_transformed = tf2_geometry_msgs.do_transform_pose(
                    pose_stamped,
                    in2out,
                )
                pose_w = pose_to_matrix(pose_transformed.pose)
            else:
                pose_w = pose_to_matrix(pose)

            # directional filtering
            # approach_w = pose_w[:3, 2] / np.linalg.norm(pose_w[:3, 2])
            # # print(approach_w, approach_w.dot(np.array([0, 0, 1])))
            # if approach_w.dot(np.array([0, 0, 1])) > 0.1:
            #     continue
            # if approach_w.dot(np.array([1, 0, 0])) < -0.1:
            #     continue

            # adjusted grasp
            pose_t = copy.deepcopy(pose_w)
            approach_t = pose_t[:3, 2] / np.linalg.norm(pose_t[:3, 2])
            displace_t = self.hand_depth
            pose_t[:3, 3] = pose_t[:3, 3] + displace_t * approach_t
            pose_msg = matrix_to_pose(pose_t)

            ## exclude invalid IK and collision
            n_ikc += 1
            t0 = time.time()
            if ik_func is None:
                octomap_reset()
                invalid = self.ik_collision(pose_msg, visualize=True)
                # input(f'Score: {score}, Invalid: {invalid}, ...')
            else:
                invalid = grasp_invalid(pose_t, static_only=True)
            t1 = time.time()
            t_ikc += t1 - t0
            if invalid:
                continue

            ## find valid pre-grasp in range
            pose_pre = None
            l = -1
            h = (pre_grasp_range[1] - pre_grasp_range[0]) / pre_grasp_res
            h = round(h)
            c = h // 2
            while c != h:
                pre_grasp_dist = c * pre_grasp_res + pre_grasp_range[0]
                # print('bin search', c, l, h)
                # print('dist', pre_grasp_dist)
                pp = self.make_pre_grasp(pose_w, pre_grasp_dist)
                n_pp += 1
                t0 = time.time()
                if ik_func is None:
                    ppm = matrix_to_pose(pp)
                    octomap_set(collision_points)
                    invalid = self.ik_collision(ppm, visualize=True)
                    # input(f'Score: {score}, Invalid: {invalid}, ...')
                else:
                    invalid = grasp_invalid(pp)
                t1 = time.time()
                t_pp += t1 - t0
                if invalid:
                    l = c
                else:
                    pose_pre = matrix_to_pose(pp)
                    h = c
                c = (l + h + 1) // 2

            if pose_pre is None:
                continue

            adjusted_score = score
            # if collision_voxel_tuple is not None:
            #     n_cc += 1
            #     t0 = time.time()
            #     in_collision = self.gripper_in_collision(
            #         pose_t,
            #         tm_vx_collision,
            #         visualize=visualize,
            #     )
            #     t1 = time.time()
            #     t_cc += t1 - t0
            #     adjusted_score = score + 10 * ((1000 - in_collision) / 1000)

            pre_grasp_list.append(pose_pre)
            pose_list.append(pose_msg)
            pose_list_scores.append(adjusted_score)
            sample_list.append(sample)

            # for msg in pose_list:
            #     self.ik_solve(msg, visualize=True)
            #     input('next')

        # print('grasp check time:', t_ikc / n_ikc, t_ikc)
        # print('pre grasp check:', t_pp / n_pp, t_pp)
        # if collision_voxel_tuple is not None:
        #     print('cc_time:', t_cc / n_cc, t_cc)
        print(len(pose_list), "grasps after filtering.")
        if len(pose_list) == 0:
            # print('No grasps after filtering!', )
            return [], [], [], []
        return pose_list, pre_grasp_list, pose_list_scores, sample_list

    def rescore_grasps(
        self,
        grasp_cfgs,
        points,
        colors,
        ik_func=None,
        collision_voxel_tuple=None,
        pre_grasp_res=0.04,
        pre_grasp_range=[0.02, 0.1],
        filter_outliers=False,
        visualize=False,
        octomap_set=None,
        octomap_reset=None,
    ):
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        pcl.colors = o3d.utility.Vector3dVector(colors)
        if filter_outliers:
            pcl, _removed = pcl.remove_statistical_outlier(*filter_outliers)
        points = pcl.points
        colors = pcl.colors
        geom_to_draw = [pcl]

        if collision_voxel_tuple is not None:
            # vx_collision, vx_resolution, vx_pose, vx_frame_id = collision_voxel_tuple
            # tm_vx_collision = trimesh.voxel.VoxelGrid(vx_collision)
            vis_points, collision_points, vx_resolution = collision_voxel_tuple
            c_pcl = o3d.geometry.PointCloud()
            c_pcl.points = o3d.utility.Vector3dVector(collision_points)
            tm_vx_collision = o3d.geometry.VoxelGrid.create_from_point_cloud(
                c_pcl, vx_resolution
            )
            geom_to_draw.append(tm_vx_collision)
            z_min = np.min(collision_points[:, 2])
        else:
            vis_points = points
            z_min = np.min(np.array(points)[:, 2])

        if visualize:
            o3d.visualization.draw_geometries(geom_to_draw)

        ## init grasp service request
        ppoints = [Point(x, y, z) for x, y, z in points]
        ccolors = [ColorRGBA(r, g, b, 0) for r, g, b in colors]
        header = Header()
        header.frame_id = "world"
        cloud = pcl2.create_cloud_xyz32(header, vis_points)
        camera_position = Point(0.14, 0, 1.4)

        ## call service
        poses = []
        scores = []
        configs = []
        srv_name = "/rescore_grasps"
        rospy.wait_for_service(srv_name, timeout=10)
        try:
            service = rospy.ServiceProxy(srv_name, ReScoreGrasps)
            resp = service(ppoints, cloud, camera_position, grasp_cfgs)  # gpd
            # resp = service(ppoints, input_frame)  # gpg call
            # resp = service(ppoints, ccolors) # graspnet call
            poses.extend(resp.grasps.poses)
            norm_scores = self.normalize_score(resp.grasps.scores)
            if srv_name.find("large") >= 0:
                norm_scores += 1
            scores.extend(norm_scores)
            configs.extend(resp.grasps.configs)
            # if len(poses) > 0:
            #     break
        except rospy.ServiceException as e:
            print("Service call failed:", e)
            # return [], [], [], []

        if len(poses) == 0:
            print("No grasps found!")
            return [], [], [], []
        # print(len(poses), 'grasps before filtering')

        def grasp_invalid(pose_mat, static_only=False):
            if ik_func(pose_mat) is None:
                return True
            if collision_voxel_tuple is None:
                return False
            return self.collision_check(pose_mat, static_only, z_min, tm_vx_collision)
            if static_only:
                return self.gripper_static_collision(pose_mat, z_min)
            return self.gripper_in_collision(pose_mat, tm_vx_collision) > 0

        pose_list = []
        pre_grasp_list = []
        pose_list_scores = []
        config_list = []
        n_ikc = 0.0001
        t_ikc = 0
        n_cc = 0.0001
        t_cc = 0
        n_pp = 0.0001
        t_pp = 0
        # score_threshold = np.percentile(normalized_scores, 90)
        # print('score_threshold:', score_threshold)
        for pose, score, config in zip(poses, scores, configs):
            # adjusted grasp
            pose_w = pose_to_matrix(pose)
            pose_t = copy.deepcopy(pose_w)
            approach_t = pose_t[:3, 2] / np.linalg.norm(pose_t[:3, 2])
            displace_t = self.hand_depth
            pose_t[:3, 3] = pose_t[:3, 3] + displace_t * approach_t
            pose_msg = matrix_to_pose(pose_t)

            ## exclude invalid IK and collision
            n_ikc += 1
            t0 = time.time()
            if ik_func is None:
                octomap_reset()
                invalid = self.ik_collision(pose_msg, visualize=True)
                # input(f'Score: {score}, Invalid: {invalid}, ...')
            else:
                invalid = grasp_invalid(pose_t, static_only=True)
            t1 = time.time()
            t_ikc += t1 - t0
            if invalid:
                continue

            ## find valid pre-grasp in range
            pose_pre = None
            l = -1
            h = (pre_grasp_range[1] - pre_grasp_range[0]) / pre_grasp_res
            h = round(h)
            c = h // 2
            while c != h:
                pre_grasp_dist = c * pre_grasp_res + pre_grasp_range[0]
                # print('bin search', c, l, h)
                # print('dist', pre_grasp_dist)
                pp = self.make_pre_grasp(pose_w, pre_grasp_dist)
                n_pp += 1
                t0 = time.time()
                if ik_func is None:
                    ppm = matrix_to_pose(pp)
                    octomap_set(collision_points)
                    invalid = self.ik_collision(ppm, visualize=True)
                    # input(f'Score: {score}, Invalid: {invalid}, ...')
                else:
                    invalid = grasp_invalid(pp)
                t1 = time.time()
                t_pp += t1 - t0
                if invalid:
                    l = c
                else:
                    pose_pre = matrix_to_pose(pp)
                    h = c
                c = (l + h + 1) // 2

            if pose_pre is None:
                continue

            adjusted_score = score
            # if collision_voxel_tuple is not None:
            #     n_cc += 1
            #     t0 = time.time()
            #     in_collision = self.gripper_in_collision(
            #         pose_t,
            #         tm_vx_collision,
            #         visualize=visualize,
            #     )
            #     t1 = time.time()
            #     t_cc += t1 - t0
            #     adjusted_score = score + 10 * ((1000 - in_collision) / 1000)

            pre_grasp_list.append(pose_pre)
            pose_list.append(pose_msg)
            pose_list_scores.append(adjusted_score)
            config_list.append(config)

        # print('grasp check time:', t_ikc / n_ikc, t_ikc)
        # print('pre grasp check:', t_pp / n_pp, t_pp)
        # if collision_voxel_tuple is not None:
        #     print('cc_time:', t_cc / n_cc, t_cc)
        print(len(pose_list), "grasps after filtering.")
        if len(pose_list) == 0:
            # print('No grasps after filtering!', )
            return [], [], [], []
        return pose_list, pre_grasp_list, pose_list_scores, config_list
