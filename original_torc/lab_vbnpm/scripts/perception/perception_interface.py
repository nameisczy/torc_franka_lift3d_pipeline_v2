import sys
import time

import cv2
import numpy as np
import open3d as o3d
import trimesh as tm
import transformations as tf

import rospy

np.float = float
import ros_numpy as rnp
from cv_bridge import CvBridge
import message_filters
import tf2_ros
import tf2_geometry_msgs
from std_msgs.msg import String, Header
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import Image, CameraInfo, JointState, PointCloud2, PointField

from segment3d.srv import GetDeticResults, GetDeticResultsRequest, GetDeticResultsResponse
from lab_vbnpm.msg import SceneOcclusion, VoxelGridBool
from lab_vbnpm.srv import InitSceneOcclusion, InitSceneOcclusionRequest, InitSceneOcclusionResponse, \
        UpdateSceneOcclusion, UpdateSceneOcclusionRequest, UpdateSceneOcclusionResponse, \
        GetSceneOcclusion, GetSceneOcclusionRequest, GetSceneOcclusionResponse, \
        GetSceneOcclusionPCD, GetSceneOcclusionPCDRequest, GetSceneOcclusionPCDResponse

from utils.conversions import pose_to_matrix
from perception.putils import merge_rgba_fields, split_rgba_field


class PerceptionInterface(object):

    def __init__(
        self,
        camera_prefixes,
        pose_x,
        pose_y,
        pose_z,
        size_x,
        size_y,
        size_z,
        resolution=0.005
    ):
        self.bridge = CvBridge()
        # init occlusion scene
        rospy.wait_for_service('init_scene_occlusion')
        init_scene_occlusion = rospy.ServiceProxy(
            'init_scene_occlusion', InitSceneOcclusion
        )
        req = InitSceneOcclusionRequest()
        req.pose.position.x = pose_x
        req.pose.position.y = pose_y
        req.pose.position.z = pose_z
        req.pose.orientation.w = 1
        req.pose.orientation.x = 0
        req.pose.orientation.y = 0
        req.pose.orientation.z = 0
        req.xyz_size.x = size_x
        req.xyz_size.y = size_y
        req.xyz_size.z = size_z
        req.resols.x = resolution
        req.resols.y = resolution
        req.resols.z = resolution
        init_scene_occlusion(req)

        # init occlusion scene parameters
        self.occluded_pose = np.eye(4)
        self.occluded_pose[0, 3] = pose_x
        self.occluded_pose[1, 3] = pose_y
        self.occluded_pose[2, 3] = pose_z
        self.occluded_size = np.array([size_x, size_y, size_z])

        # init tf listener
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(60))
        self.tf_listen = tf2_ros.TransformListener(self.tf_buffer)

        # init camera result structures
        self.camera_prefixes = camera_prefixes
        self.cam_info = {}
        self.color_img = {}
        self.depth_img = {}
        self.cam_transform = {}
        self.joint_state = {}

        # init camera subscribers
        self.ts = {}
        self.info_sub = {}
        self.color_sub = {}
        self.depth_sub = {}
        for prefix in self.camera_prefixes:
            self.info_sub[prefix] = message_filters.Subscriber(
                prefix + '/color/camera_info', CameraInfo
            )
            self.color_sub[prefix] = message_filters.Subscriber(
                prefix + '/color/image_raw', Image
            )
            self.depth_sub[prefix] = message_filters.Subscriber(
                prefix + '/aligned_depth_to_color/image_raw', Image
            )
            self.ts[prefix] = message_filters.ApproximateTimeSynchronizer(
                [
                    self.info_sub[prefix],
                    self.color_sub[prefix],
                    self.depth_sub[prefix],
                ], 10, 0.5
            )
            self.ts[prefix].registerCallback(self.updateImage, prefix)

        # init point cloud filter publisher
        self.pcd_filter_pub = rospy.Publisher(
            '/velodyne_points', PointCloud2, queue_size=3
        )

        # init fused points and colors
        self.reset_fusion()

    def reset_fusion(self):
        self.fused_points = np.reshape([], (0, 3))
        self.fused_colors = np.reshape([], (0, 3))
        self.fused_target_mask = np.array([], dtype=bool)
        self.fuse_history = []

    def save_fusion(self, dir_path, target_name):
        now = time.asctime().replace(' ', '_').replace(':', '-')
        filename = f'{dir_path}/point_cloud_{now}.pcd'
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.fused_points)
        pcd.colors = o3d.utility.Vector3dVector(self.fused_colors)
        o3d.io.write_point_cloud(filename, pcd)
        maskfile = f'{dir_path}/target-#{target_name.replace(" ","_")}#-{now}.npy'
        np.save(maskfile, self.fused_target_mask)

    def updateImage(self, info, color, depth, prefix):
        try:
            camera2world = self.tf_buffer.lookup_transform(
                'world',
                prefix + '_color_optical_frame',
                rospy.Time(),
                rospy.Duration(0.1),
            )
            joint_state = rospy.wait_for_message(
                '/joint_states_all', JointState
            )
            self.cam_info[prefix] = info
            self.color_img[prefix] = color
            self.depth_img[prefix] = depth
            self.cam_transform[prefix] = camera2world.transform
            self.joint_state[prefix] = joint_state
        except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
        ):
            pass
            # print(
            #     "Error: Couldn't get transform of camera pose!",
            #     file=sys.stderr
            # )

    def create_cloud(self, points, colors, mask, frame_id, now):
        rgba_array = np.zeros(
            len(colors),
            dtype=[
                ('r', np.uint8),
                ('g', np.uint8),
                ('b', np.uint8),
                ('a', np.uint8),
            ]
        )
        rgba_array['r'] = colors[:, 0]
        rgba_array['g'] = colors[:, 1]
        rgba_array['b'] = colors[:, 2]
        rgba_array['a'] = mask
        f_array = merge_rgba_fields(rgba_array)

        pc_array = np.zeros(
            len(points),
            dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('rgba', np.float32),
            ]
        )
        pc_array['x'] = points[:, 0]
        pc_array['y'] = points[:, 1]
        pc_array['z'] = points[:, 2]
        pc_array['rgba'] = f_array

        msg = rnp.msgify(PointCloud2, pc_array, stamp=now, frame_id=frame_id)
        return msg

    def filter_cloud(self, points, colors, mask, time_msg):
        # filtering is not currently in sync with the sensor data
        cloud = self.create_cloud(
            points, colors * 255, mask * 255, 'world', time_msg
        )
        try:
            rospy.wait_for_message(
                '/velodyne_points_filtered', PointCloud2, 0.1
            )
        except:
            pass
        filtered_cloud = None
        while filtered_cloud is None:
            try:
                self.pcd_filter_pub.publish(cloud)
                filtered_cloud = rospy.wait_for_message(
                    '/velodyne_points_filtered', PointCloud2, 5.0
                )
            except:
                print('Trying to get filtered cloud again...', file=sys.stderr)
        filtered_cloud_array = rnp.numpify(filtered_cloud)
        filtered_points = rnp.point_cloud2.get_xyz_points(
            filtered_cloud_array, remove_nans=False
        )
        filtered_colors_array = split_rgba_field(filtered_cloud_array)
        filtered_colors = np.zeros((len(filtered_colors_array), 3))
        filtered_mask = np.zeros(len(filtered_colors_array), dtype=bool)
        filtered_colors[:, 0] = filtered_colors_array['r']
        filtered_colors[:, 1] = filtered_colors_array['g']
        filtered_colors[:, 2] = filtered_colors_array['b']
        filtered_colors /= 255
        filtered_mask = filtered_colors_array['a']

        return filtered_points, filtered_colors, filtered_mask

    def updated_fused_points(
        self,
        prefix,
        target,
        debug=False,
        resolution=0.005,
        filter_outliers=False,
    ):

        visible_tuple = self.get_visible_points(prefix, target, debug)
        points, colors, target_mask, bg_mask, image_mask_tuple = visible_tuple
        self.fuse_history.append(image_mask_tuple)

        filtered_points, filtered_colors, filtered_mask = self.filter_cloud(
            points, colors, target_mask, self.cam_info[prefix].header.stamp
        )
        scene_points = np.concatenate((filtered_points, self.fused_points))
        scene_colors = np.concatenate((filtered_colors, self.fused_colors))
        scene_mask = np.concatenate((filtered_mask, self.fused_target_mask))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene_points)
        pcd.colors = o3d.utility.Vector3dVector(scene_colors)
        mask = np.zeros((len(scene_mask), 3))
        mask[:, 0] = scene_mask
        pcd.normals = o3d.utility.Vector3dVector(mask)
        downpcd = pcd.voxel_down_sample(resolution)
        downpcd, trace, inds = pcd.voxel_down_sample_and_trace(
            resolution, pcd.get_min_bound(), pcd.get_max_bound(), False
        )
        downpcd = downpcd.crop(
            o3d.geometry.AxisAlignedBoundingBox(
                self.occluded_pose[:3, 3],
                self.occluded_pose[:3, 3] + self.occluded_size,
            )
        )
        if filter_outliers:
            # downpcd, ind = downpcd.remove_radius_outlier(*filter_outliers)
            downpcd, _removed = pcd.remove_statistical_outlier(*filter_outliers)
        self.fused_points = np.asarray(downpcd.points)
        self.fused_colors = np.asarray(downpcd.colors)
        fused_target_mask = np.asarray(downpcd.normals)
        self.fused_target_mask = fused_target_mask[:, 0].astype(bool)
        return

        self.fused_target_mask = np.zeros(len(self.fused_points), dtype=bool)
        for image_mask, cam_dist, cam_intr, cam_extr in self.fuse_history:
            target_image_mask = cv2.erode(image_mask, np.ones((20, 20)))
            # world_points = np.asarray(downpcd.points).transpose()
            # world_points = np.append(world_points, np.ones((1, world_points.shape[1])), axis=0)
            # points_in_ccs = np.matmul(np.linalg.inv(cam_extr), world_points)[:3]
            # points_in_ccs = points_in_ccs / points_in_ccs[2, :]
            # projected_points = np.matmul(cam_intr, points_in_ccs).T
            # projected_points = projected_points.astype(np.int16)
            # print(projected_points,projected_points.shape)

            # height = max([a.shape[0] for a, b, c, d in self.fuse_history])
            # width = max([a.shape[1] for a, b, c, d in self.fuse_history])
            height, width = target_image_mask.shape
            close_z = np.full((height, width), np.inf)
            close_i = np.full((height, width), -1, dtype=int)
            depths = {}
            for i, point in enumerate(downpcd.points):
                # points = np.matmul(points, mat.T[:3, :3]) + mat[:3, 3]
                point4d = np.append(point, 1)
                new_point4d = np.matmul(np.linalg.inv(cam_extr), point4d)
                point3d = new_point4d[:-1]
                zc = point3d[2]
                new_point3d = np.matmul(cam_intr, point3d)
                new_point3d = new_point3d / new_point3d[2]
                u = int(round(new_point3d[0]))
                v = int(round(new_point3d[1]))

                if u < 0 or u > width - 1 or v < 0 or v > height - 1:
                    continue
                if not target_image_mask[v, u]:
                    continue

                if zc < close_z[v, u]:
                    # dist = np.linalg.norm(point - cam_extr[:3, 3])
                    # if dist < close_z[v, u]:
                    close_z[v, u] = zc
                    depths[(v, u)] = depths.get((v, u), []) + [zc]
                    # close_z[v, u] = dist
                    close_i[v, u] = i

            print(depths)
            inds = close_i[(close_i >= 0)]  # & target_image_mask]
            self.fused_target_mask[inds] = True

    def get_fused_point_cloud(self):
        return self.fused_points, self.fused_colors

    def get_fused_bg_point_cloud(self):
        bg_points = self.fused_points[~self.fused_target_mask]
        bg_colors = self.fused_colors[~self.fused_target_mask]
        return bg_points, bg_colors

    def get_fused_target_point_cloud(self):
        target_points = self.fused_points[self.fused_target_mask]
        target_colors = self.fused_colors[self.fused_target_mask]
        return target_points, target_colors

    def get_visible_points(self, prefix, target, debug=False):
        while prefix not in self.cam_info:
            # print(prefix, self.cam_info)
            rospy.sleep(0.1)
        rospy.wait_for_service('detic_service', timeout=5)
        try:
            req = GetDeticResultsRequest()
            req.target_name = String(target)
            req.cam_info = self.cam_info[prefix]
            req.color_img = self.color_img[prefix]
            req.depth_img = self.depth_img[prefix]
            req.debug_mode = debug
            serv = rospy.ServiceProxy('detic_service', GetDeticResults)
            resp = serv(req)
            success = resp.success
        except Exception as e:
            success = False
            print("Service error:", e)
        if not success:
            print("Service failed!")
            return None, None, None, None, None

        points = np.array(resp.points.data, dtype=np.float32).reshape(-1, 3)
        colors = np.array(resp.colors.data, dtype=np.float32).reshape(-1, 3)
        target_mask = np.array(resp.target_mask)
        bg_mask = np.array(resp.background_mask)
        target_image_mask = self.bridge.imgmsg_to_cv2(
            resp.target_image_mask, "mono8"
        ).astype(bool)

        # transform points into world frame
        transform = self.cam_transform[prefix]
        qx = transform.rotation.x
        qy = transform.rotation.y
        qz = transform.rotation.z
        qw = transform.rotation.w
        x = transform.translation.x
        y = transform.translation.y
        z = transform.translation.z
        mat = tf.quaternion_matrix([qw, qx, qy, qz])
        mat[:3, 3] = [x, y, z]
        points = np.matmul(points, mat.T[:3, :3]) + mat[:3, 3]

        # joint_state_msg = self.joint_state[prefix]
        # now = joint_state_msg.header.stamp
        cam_intr = np.array(self.cam_info[prefix].K).reshape((3, 3))
        cam_dist = np.array(self.cam_info[prefix].D)

        image_mask_tuple = (target_image_mask, cam_dist, cam_intr, mat)
        return points, colors, target_mask, bg_mask, image_mask_tuple

    def update_occlusion(self, prefix):
        while prefix not in self.cam_info:
            rospy.sleep(0.1)
        rospy.wait_for_service("update_scene_occlusion")
        update_scene_occlusion = rospy.ServiceProxy(
            "update_scene_occlusion", UpdateSceneOcclusion
        )
        req = UpdateSceneOcclusionRequest()
        req.cam_info = self.cam_info[prefix]
        req.depth_img = self.depth_img[prefix]
        req.cam_transform = self.cam_transform[prefix]
        req.debug_mode = False
        update_scene_occlusion(req)

    def get_occlusion_voxels(self):
        rospy.wait_for_service("get_scene_occlusion")
        get_scene_occlusion = rospy.ServiceProxy(
            "get_scene_occlusion", GetSceneOcclusion
        )
        voxel = get_scene_occlusion().data.voxel
        frame_id = voxel.header.frame_id
        pose = pose_to_matrix(voxel.pose)
        shape = (voxel.size_x, voxel.size_y, voxel.size_z)
        occluded = np.reshape(voxel.data, shape).astype(bool)
        resolution = (voxel.resols.x, voxel.resols.y, voxel.resols.z)
        return occluded, resolution, pose, frame_id

    def get_occlusion_points(self):
        rospy.wait_for_service("get_scene_occlusion_pcd")
        get_scene_occlusion_pcd = rospy.ServiceProxy(
            "get_scene_occlusion_pcd", GetSceneOcclusionPCD
        )
        msg = get_scene_occlusion_pcd()
        pcd = np.array(msg.data).reshape((-1, 3))
        return pcd

    def get_largest_target_cluster(self):
        # get largest cluster in point cloud
        points, colors = self.get_fused_target_point_cloud()
        clusters = tm.grouping.clusters(points, 0.02)
        max_cluster_inds = max(clusters, key=len)
        c_points = points[max_cluster_inds]
        c_colors = colors[max_cluster_inds]
        return c_points, c_colors

    def get_symmetric_shape(self, points, colors, visualize=False):
        cloud = tm.points.PointCloud(points, colors)
        if visualize:
            cloud.show()

        # augmented oriented box candidates
        to_origin, extents = tm.bounds.oriented_bounds(cloud)
        a_extents = [min(extents[0] * 2, extents[1]), extents[1], extents[2]]
        transform0 = tm.transformations.concatenate_matrices(
            np.linalg.inv(to_origin),
            tm.transformations.translation_matrix((a_extents - extents) / 2),
        )
        transform1 = tm.transformations.concatenate_matrices(
            np.linalg.inv(to_origin),
            tm.transformations.translation_matrix((extents - a_extents) / 2),
        )
        a_box0 = tm.primitives.Box(a_extents, transform0)
        a_box1 = tm.primitives.Box(a_extents, transform1)

        # pick bounding volume with furthest centroid from point cloud
        shape = max(
            a_box0,
            a_box1,
            cloud.bounding_box,
            cloud.bounding_sphere,
            cloud.bounding_cylinder,
            key=lambda s: cloud.kdtree.query([s.center_mass], k=1)[0][0]
        )

        # rotate point cloud by centroid of bounding volume
        rot = tm.transformations.rotation_matrix(
            np.pi, [0, 0, 1], shape.center_mass
        )
        if visualize:
            centroid = tm.primitives.Sphere(
                radius=0.005, center=shape.center_mass
            )
            tm.scene.Scene([cloud, centroid, shape]).show()
        cloud.apply_transform(rot)

        # create symmetric point cloud from original and rotates points
        new_points = np.concatenate([points, cloud.vertices])
        new_colors = np.concatenate([colors, cloud.colors[:, :3] / 255.0])
        new_cloud = tm.points.PointCloud(new_points, new_colors)
        if visualize:
            tm.scene.Scene([new_cloud, new_cloud.bounding_primitive]).show()

        # return points and smallest volume bounding primitive of symmetric point cloud
        return new_cloud.bounding_primitive, new_cloud
