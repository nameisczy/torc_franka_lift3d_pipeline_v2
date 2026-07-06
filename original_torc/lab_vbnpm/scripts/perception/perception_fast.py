import sys
import json
import time
import itertools
import threading
from math import cos, sin

import cv2
import numpy as np
import open3d as o3d
import trimesh as tm
from numba import jit
import transformations as tf
from scipy.spatial import KDTree
from tracikpy import TracIKSolver

import rospy
from rospkg import RosPack

np.float = float
import ros_numpy as rnp
from cv_bridge import CvBridge
import message_filters
import tf2_ros

# import tf2_geometry_msgs
from std_msgs.msg import String, Header
import sensor_msgs.point_cloud2 as pcl2
from geometry_msgs.msg import TransformStamped, Transform, Point
from sensor_msgs.msg import Image, CameraInfo, JointState, PointCloud2, PointField

from segment3d.srv import (
    GetDeticResults,
    GetDeticResultsRequest,
    GetDeticResultsResponse,
    GetGSAMResults,
    GetGSAMResultsRequest,
    GetGSAMResultsResponse,
)

from utils.conversions import *
from fusion import fusion
from utils.visual_utils import decode_seg_img_rgb
from perception.putils import merge_rgba_fields, split_rgba_field
import mujoco
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker, MarkerArray
import subprocess

import torch
from omegaconf import OmegaConf
from foundation_stereo.core.utils.utils import InputPadder
from foundation_stereo.core.foundation_stereo import *


def set_seed(random_seed):
    import torch, random

    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        resolution=0.05,
        mode="gt",
        urdf_file=None,
    ):
        """
        valid modes are:
        - 'gt' for ground truth
        - 'fs' for foundation stereo
        - 'cam' for native camera (or simulated camera)
        """

        self.lock = threading.Lock()
        self.added_cam_poses = []
        self.bridge = CvBridge()
        self.mode = mode
        self.times_integrated = 0
        rp = RosPack()

        # init foundation stereo
        if self.mode == "fs":
            set_seed(0)
            # torch.autograd.set_grad_enabled(False)
            with torch.no_grad():
                ckpt_dir = rp.get_path("lab_vbnpm") + "/checkpoints"
                cfg = OmegaConf.load(f"{ckpt_dir}/cfg.yaml")
                ckpt_dir += "/model_best_bp2.pth"
                cfg["ckpt_dir"] = ckpt_dir
                if "vit_size" not in cfg:
                    cfg["vit_size"] = "vitl"
                args = OmegaConf.create(cfg)

                model = FoundationStereo(args)
                ckpt = torch.load(ckpt_dir, weights_only=False)
                print(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
                model.load_state_dict(ckpt["model"])
                model.cuda()
                model.eval()

            self.fs = model

            self.baseline = {  # The distance between IR cameras
                "d455": 0.095,
                "d435": 0.05,
            }

        # init tf listener
        param = rospy.get_param("/my_eob_calib_eye_on_base/transformation")
        if param:
            self.tf_buffer = tf2_ros.Buffer(rospy.Duration(60))
            self.tf_listen = tf2_ros.TransformListener(self.tf_buffer)
            self.tf_pub = {}
            self.tf_sub = {}
            self.ik = {}
            self.cam2link = {}
            for i, prefix in enumerate(camera_prefixes):
                while not self.tf_buffer.can_transform(
                    prefix + "_link",
                    prefix + "_color_optical_frame",
                    rospy.Time(),
                ):
                    print(
                        f"Waiting for transform {prefix}_link to {prefix}_color_optical_frame..."
                    )
                    rospy.sleep(0.1)
                cam2link = transform_to_matrix(
                    self.tf_buffer.lookup_transform(
                        prefix + "_link",
                        prefix + "_color_optical_frame",
                        rospy.Time(),
                    ).transform
                )
                if i == 0:
                    cam_link = rospy.get_param(
                        "/my_eob_calib_eye_on_base/robot_effector_frame"
                    )
                    self.cam2link[prefix] = dict_to_matrix(
                        rospy.get_param("/my_eob_calib_eye_on_base/transformation")
                    )
                else:
                    cam_link = rospy.get_param(
                        "/my_eob_calib_eye_on_hand/robot_effector_frame"
                    )
                    self.cam2link[prefix] = dict_to_matrix(
                        rospy.get_param("/my_eob_calib_eye_on_hand/transformation")
                    )
                self.ik[prefix] = TracIKSolver(urdf_file, "base_link", cam_link)
                self.cam2link[prefix] = self.cam2link[prefix] @ cam2link

                self.tf_pub[prefix] = rospy.Publisher(
                    "/tf_camera/" + prefix, TransformStamped, queue_size=3
                )
                self.tf_sub[prefix] = rospy.Subscriber(
                    "/joint_states_all", JointState, self.tf_callback, prefix
                )
        else:
            self.cam2link = {}
            # Maps link7 to camera
            self.cam2link[camera_prefixes[1]] = np.array(
                [
                    [1.0, 0, 0.0, 0.061],
                    [0, -1, 0.0, 0.000],
                    [0.0, 0, -1, 0.040],
                    [0.0, 0, 0.0, 1.000],
                ]
            )

            # self.cam2link[camera_prefixes[1]] = np.array([
            #     [1,     0,      0,      0],
            #     [0,     -1,      0,      -0.061],
            #     [0,     0,      -1,     0.04],
            #     [0,     0,      0,      1]
            # ])

            # self.cam2link[camera_prefixes[1]] = np.array([
            #     [0,     1,      0,      0],
            #     [1,     0,      0,      -0.061],
            #     [0,     0,      -1,     -0.112],
            #     [0,     0,      0,      1]
            # ])

            # for i, prefix in enumerate(camera_prefixes):
            #     if i == 0:
            #         pass
            #         # We don't need cam2link for torso in simulation (already have the exact value)

        self.ray_pub = rospy.Publisher("/ray_publish", Marker, queue_size=3)
        self.ray_pub_2 = rospy.Publisher("/ray_publish_2", Marker, queue_size=3)
        self.occl_cluster_pub = rospy.Publisher(
            "/occl_cluster_points", PointCloud2, queue_size=3
        )

        self.gt_state = None
        self.mesh_sub = rospy.Subscriber(
            "visualization_marker", MarkerArray, self.mesh_sub_callback, queue_size=1
        )

        self.marker = Marker()
        self.marker.header.frame_id = "world"
        self.marker.header.stamp = rospy.Time.now()
        self.marker.id = 0
        self.marker.type = Marker.LINE_LIST  # Using LINE_LIST for multiple rays
        self.marker.action = Marker.ADD
        self.marker.scale.x = 0.002
        self.marker.color.r = 0.0
        self.marker.color.g = 0.5
        self.marker.color.b = 0.5
        self.marker.color.a = 0.7  # alpha
        self.marker_cone = Marker()
        self.marker_cone.header.frame_id = "world"
        self.marker_cone.header.stamp = rospy.Time.now()
        self.marker_cone.id = 0
        self.marker_cone.type = Marker.LINE_LIST  # Using LINE_LIST for multiple rays
        self.marker_cone.action = Marker.ADD
        self.marker_cone.scale.x = 0.002
        self.marker_cone.color.r = 0.8
        self.marker_cone.color.g = 0.1
        self.marker_cone.color.b = 0.1
        self.marker_cone.color.a = 0.8  # alpha

        # init camera result structures
        self.cur_color = None
        self.cur_depth = None
        self.cur_mask = None
        self.cur_points = np.empty((0, 3), dtype=np.float32)
        self.cur_pmask = np.empty(0, dtype=np.uint32)
        self.camera_prefixes = camera_prefixes
        self.cam_info = {}
        self.color_img = {}
        self.depth_img = {}
        if self.mode == "gt":
            self.seg_img = {}
            self.seg_label_to_obj_id = {}
        elif self.mode == "fs":
            self.infra_info = {}
            self.left_img = {}
            self.right_img = {}
            self.t_rgb2infra = {}
        self.cam_transform = {}
        self.joint_state = {}

        # init point cloud filter publisher
        self.pcd_filter_pub = rospy.Publisher(
            "/velodyne_points", PointCloud2, queue_size=3
        )
        self.debug_pcd = rospy.Publisher(
            "/debug/realtime_pcd",
            PointCloud2,
            queue_size=3,
            latch=True,
        )

        # init fused, color, and occluded tsdfs
        pos_a = [pose_x, pose_y, pose_z]
        pos_b = [pose_x + size_x, pose_y + size_y, pose_z + size_z]
        vol_bnds = np.zeros((3, 2))
        vol_bnds[:, 0] = np.minimum(pos_a, pos_b)
        vol_bnds[:, 1] = np.maximum(pos_a, pos_b)
        vol_bnds[2, 0] += 0.005
        self.bounds = vol_bnds
        self.resolution = resolution
        self.tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=resolution)
        # self.tsdf_vis = fusion.TSDFVolume(vol_bnds, voxel_size=resolution)
        self.centroid = None
        self.camera_iter = 0

        # init camera subscribers
        self.ts = {}
        self.info_sub = {}
        self.color_sub = {}
        self.depth_sub = {}
        self.tf2_sub = {}
        if self.mode == "gt":
            self.seg_sub = {}
            self.target = "35"
        elif self.mode == "fs":
            self.infra_info_sub = {}
            self.infra_left_sub = {}
            self.infra_right_sub = {}
        for prefix in self.camera_prefixes:
            self.info_sub[prefix] = message_filters.Subscriber(
                prefix + "/color/camera_info", CameraInfo
            )
            print(prefix + "/color/camera_info")
            self.color_sub[prefix] = message_filters.Subscriber(
                prefix + "/color/image_raw", Image
            )
            self.depth_sub[prefix] = message_filters.Subscriber(
                prefix + "/aligned_depth_to_color/image_raw", Image
            )
            self.tf2_sub[prefix] = message_filters.Subscriber(
                "/tf_camera/" + prefix, TransformStamped
            )
            subs = [
                self.info_sub[prefix],
                self.color_sub[prefix],
                self.depth_sub[prefix],
                self.tf2_sub[prefix],
            ]
            if self.mode == "gt":
                self.seg_sub[prefix] = message_filters.Subscriber(
                    "/ground_truth/" + prefix + "/seg_image", Image
                )
                subs.append(self.seg_sub[prefix])
            elif self.mode == "fs":
                self.infra_info_sub[prefix] = message_filters.Subscriber(
                    prefix + "/infra1/camera_info", CameraInfo
                )
                self.infra_left_sub[prefix] = message_filters.Subscriber(
                    prefix + "/infra1/image_rect_raw", Image
                )
                self.infra_right_sub[prefix] = message_filters.Subscriber(
                    prefix + "/infra2/image_rect_raw", Image
                )
                subs.append(self.infra_info_sub[prefix])
                subs.append(self.infra_left_sub[prefix])
                subs.append(self.infra_right_sub[prefix])

                self.t_rgb2infra[prefix] = transform_to_matrix(
                    self.tf_buffer.lookup_transform(
                        prefix + "_color_optical_frame",
                        prefix + "_depth_optical_frame",
                        rospy.Time(),
                    ).transform
                )

            self.ts[prefix] = message_filters.ApproximateTimeSynchronizer(
                subs, 10, 0.01
            )
            if self.mode == "gt":
                self.ts[prefix].registerCallback(self.updateImage, None, None, prefix)
            else:
                if self.mode == "fs":
                    self.ts[prefix].registerCallback(self.updateImage, prefix)
                else:
                    self.ts[prefix].registerCallback(
                        self.updateImage, None, None, None, prefix
                    )
                self.img_pub = rospy.Publisher("/detic_topic", Image, queue_size=1)

    def mesh_sub_callback(self, msg):
        self.gt_state = msg

    def eval_pcd(self, pcd, obj_name="006_mustard_bottle"):
        # obj_name = "003_cracker_box" # "001_chips_can"
        # obj_name = "001_chips_can"
        print("Evaluating point cloud!")
        if not self.gt_state:
            # self.gt_state is None (have not received message from execution node yet)
            print(
                "self.gt_state is None (have not received message from execution node yet)"
            )
            return

        # Obtain ground-truth target point cloud from gt_state
        tgt_mesh_msg = None
        for body in self.gt_state.markers:
            if body.text == obj_name:
                tgt_mesh_msg = body
                break

        if tgt_mesh_msg is None:
            print(f"Target object {obj_name} mesh not found in gt_state")
            return

        tgt_mesh_path = tgt_mesh_msg.mesh_resource
        tgt_mesh_path = tgt_mesh_path.replace("file://", "")
        tgt_mesh = tm.load_mesh(tgt_mesh_path)
        tgt_mesh_pose = pose_to_matrix(tgt_mesh_msg.pose)

        # Convert mesh, pcd into point cloud format

        # tgt_mesh.vertices (N, 3)
        # x_transformed = rotation @ x_original + position
        transformed_tgt_mesh_pts = tgt_mesh_pose[:3, :3] @ tgt_mesh.vertices.T
        transformed_tgt_mesh_pts += tgt_mesh_pose[:3, 3:]
        transformed_tgt_mesh_pts = transformed_tgt_mesh_pts.T

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(transformed_tgt_mesh_pts)

        pcd_obs = o3d.geometry.PointCloud()
        pcd_obs.points = o3d.utility.Vector3dVector(pcd)

        ## EVALUATION

        print("[k_user] ============ POINT CLOUD EVALUATION STATISTICS ============")

        # Compute True/False positives

        # metric_1 = self.compute_rmse(pcd_gt, pcd_obs)
        percent_tp, num_tp = self.percent_matched(pcd_gt, pcd_obs)
        percent_fp, num_fp = self.percent_wrong_points(pcd_gt, pcd_obs)
        # In theory if we have a correct "True positive" metric we can implicitly compute False positives
        # We still compute False positives as a sanity check

        print("[k_user] True positive ratio/number", percent_tp, num_tp)
        print("[k_user] False positive ratio/number", percent_fp, num_fp)

        # Compute True/False negatives

        # metric_1 = self.compute_rmse(pcd_gt, pcd_obs)
        percent_tn, num_tn = self.percent_matched(pcd_obs, pcd_gt)
        percent_fn, num_fn = self.percent_wrong_points(pcd_obs, pcd_gt)
        # In theory if we have a correct "True positive" metric we can implicitly compute False positives
        # We still compute False positives as a sanity check

        print("[k_user] True negative ratio/number", percent_tn, num_tn)
        print("[k_user] False negative ratio/number", percent_fn, num_fn)

    def tf_callback(self, msg, prefix):
        ik = self.ik[prefix]
        cam2link = self.cam2link[prefix]
        joint_indices = list(map(msg.name.index, ik.joint_names))
        joint_values = np.array(msg.position)[joint_indices]
        pose = ik.fk(joint_values) @ cam2link
        tf_msg = TransformStamped()
        tf_msg.header = msg.header
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = prefix + "_color_optical_frame"
        tf_msg.transform = matrix_to_transform(pose)
        self.tf_pub[prefix].publish(tf_msg)

        # try:
        #     camera2world = self.tf_buffer.lookup_transform(
        #         'world',
        #         prefix + '_color_optical_frame',
        #         rospy.Time(),
        #         rospy.Duration(0.5),
        #     )
        #     self.tf_pub[prefix].publish(camera2world)
        #     print('tf_callback', camera2world.header.stamp)
        #     print('tf_callback cam2world:\n', camera2world.transform)
        #     print('tf_callback pose:\n', matrix_to_transform(pose))
        # except (
        #         tf2_ros.LookupException,
        #         tf2_ros.ConnectivityException,
        #         tf2_ros.ExtrapolationException,
        # ) as e:
        #     print('tf_error:', e)
        #     pass

    def save_fusion(self, dir_path, points, colors, name="target", number=None):
        if number is None:
            now = time.asctime().replace(" ", "_").replace(":", "-")
        else:
            now = f"{number:04d}"
        filename = f"{dir_path}/{name}_{now}.pcd"
        pcd = o3d.geometry.PointCloud()
        # fused_points, fused_colors = self.get_fused_point_cloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filename, pcd)
        # maskfile = f'{dir_path}/target-#{target_name.replace(" ","_")}#-{now}.npy'
        # fused_target_mask = self.get_fused_target_mask()
        # np.save(maskfile, fused_target_mask)

    def save_image(self, dir_path, cameras, number, segmentations):
        print("START")
        if number == 0:
            with self.lock:
                if self.mode == "fs":
                    left = self.left_img[cameras[0]]
                    right = self.right_img[cameras[0]]
                color = self.color_img[cameras[0]]
                depth = self.depth_img[cameras[0]]
                segim = segmentations[0]
                transform = self.cam_transform[cameras[0]]
            transform = transform_to_matrix(transform)
            # print(depth, flush=True)
            if self.mode == "fs":
                f_left = f"{dir_path}/{cameras[0]}_left.jpg"
                f_right = f"{dir_path}/{cameras[0]}_right.jpg"
                cv2.imwrite(f_left, self.bridge.imgmsg_to_cv2(left, "mono8"))
                cv2.imwrite(f_right, self.bridge.imgmsg_to_cv2(right, "mono8"))
            file_color = f"{dir_path}/{cameras[0]}_color.jpg"
            file_depth = f"{dir_path}/{cameras[0]}_depth.png"
            cv2.imwrite(file_color, self.bridge.imgmsg_to_cv2(color, "bgr8"))
            cv2.imwrite(file_depth, self.bridge.imgmsg_to_cv2(depth, "16UC1"))
            if segim is not None:
                file_segim = f"{dir_path}/{cameras[0]}_mask.npy"
                # cv2.imwrite(file_segim, segim)
                np.save(file_segim, segim)
            file_pose = f"{dir_path}/{cameras[0]}_pose.txt"
            np.savetxt(file_pose, np.array(transform))
            for cam in cameras:
                info = self.cam_info[cam]
                config_file = f"{dir_path}/{cam}_config.json"
                with open(config_file, "w") as f:
                    K = np.reshape(info.K, (3, 3)).tolist()
                    config = {
                        "intrinsic_matrix": K,
                        "height": info.height,
                        "width": info.width,
                        "depth_scale": 1.0,
                    }
                    json.dump(config, f, indent=2)
        with self.lock:
            if self.mode == "fs":
                left = self.left_img[cameras[-1]]
                right = self.right_img[cameras[-1]]
            color = self.color_img[cameras[-1]]
            depth = self.depth_img[cameras[-1]]
            segim = segmentations[-1]
            transform = self.cam_transform[cameras[-1]]
        transform = transform_to_matrix(transform)
        if self.mode == "fs":
            file_left = f"{dir_path}/depth/{number:04d}-left.jpg"
            file_right = f"{dir_path}/depth/{number:04d}-right.jpg"
            cv2.imwrite(file_left, self.bridge.imgmsg_to_cv2(left, "mono8"))
            cv2.imwrite(file_right, self.bridge.imgmsg_to_cv2(right, "mono8"))
        file_color = f"{dir_path}/color/{number:04d}-color.jpg"
        file_depth = f"{dir_path}/depth/{number:04d}-depth.png"
        cv2.imwrite(file_color, self.bridge.imgmsg_to_cv2(color, "bgr8"))
        cv2.imwrite(file_depth, self.bridge.imgmsg_to_cv2(depth, "16UC1"))
        if segim is not None:
            file_segim = f"{dir_path}/mask/{number:04d}-mask.npy"
            # cv2.imwrite(file_segim, segim)
            np.save(file_segim, segim)
        pose_file = f"{dir_path}/poses/{number:04d}-pose.txt"
        np.savetxt(pose_file, np.array(transform))
        print("FINISH")

    def updateImage(self, info, color, depth, tf, II_SEG, left, right, prefix):
        # print("UpdateImage Run")
        # print('Camera: ', info.header.stamp.to_sec())
        # print('Depth: ', depth.header.stamp.to_sec())
        # print('Transform: ', tf.header.stamp.to_sec())
        with self.lock:
            self.cam_info[prefix] = info
            self.color_img[prefix] = color
            self.depth_img[prefix] = depth
            self.cam_transform[prefix] = tf.transform
            if self.mode == "gt":
                self.seg_img[prefix] = II_SEG
            else:
                if self.mode == "fs":
                    self.infra_info[prefix] = II_SEG
                    self.left_img[prefix] = left
                    self.right_img[prefix] = right
                num_cams = len(self.camera_prefixes)
                if num_cams > 1 and prefix == self.camera_prefixes[1]:
                    # only pub hand cam
                    self.img_pub.publish(color)

        color_img = self.bridge.imgmsg_to_cv2(color, "rgb8")
        depth_img = self.bridge.imgmsg_to_cv2(depth, "32FC1")
        depth_img /= 1000.0
        # mask_img = np.zeros_like(depth_img)
        cam_intr = np.array(info.K).reshape((3, 3))
        cam_pose = transform_to_matrix(tf.transform)
        # self.tsdf_vol.integrate(
        #     color_img, depth_img, mask_img, cam_intr, cam_pose, obs_weight=1.
        # )

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.intrinsic_matrix = cam_intr
        depth_im_o3d = o3d.geometry.Image(depth_img)
        color_im_o3d = o3d.geometry.Image(color_img)
        rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(
            color_im_o3d, depth_im_o3d, depth_scale=1, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud().create_from_rgbd_image(
            rgbd, intrinsic_o3d, extrinsic=np.linalg.inv(cam_pose)
        )
        rgb_points = np.asarray(pcd.colors)
        r = rgb_points[:, 0]
        g = rgb_points[:, 1]
        b = rgb_points[:, 2]
        brg_points = np.array([b, r, g]).T * 255

        # debug #
        surface_msg = PerceptionInterface.create_cloud(
            np.asarray(pcd.points),
            brg_points,
            255,
            "world",
            rospy.Time.now(),
        )
        self.debug_pcd.publish(surface_msg)

    @staticmethod
    def create_cloud(points, colors, mask, frame_id, now):
        rgba_array = np.zeros(
            len(colors),
            dtype=[
                ("r", np.uint8),
                ("g", np.uint8),
                ("b", np.uint8),
                ("a", np.uint8),
            ],
        )
        rgba_array["r"] = colors[:, 0]
        rgba_array["g"] = colors[:, 1]
        rgba_array["b"] = colors[:, 2]
        rgba_array["a"] = mask
        f_array = merge_rgba_fields(rgba_array)

        pc_array = np.zeros(
            len(points),
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgba", np.float32),
            ],
        )
        pc_array["x"] = points[:, 0]
        pc_array["y"] = points[:, 1]
        pc_array["z"] = points[:, 2]
        pc_array["rgba"] = f_array

        msg = rnp.msgify(PointCloud2, pc_array, stamp=now, frame_id=frame_id)
        return msg

    def filter_cloud(self, points, colors, mask, time_msg):
        # filtering is not currently in sync with the sensor data
        cloud = self.create_cloud(points, colors * 255, mask * 255, "world", time_msg)
        try:
            old_cloud = rospy.wait_for_message(
                "/velodyne_points_filtered", PointCloud2, 0.05
            )
        except:
            old_cloud = None
        filtered_cloud = None
        while filtered_cloud == old_cloud:
            try:
                self.pcd_filter_pub.publish(cloud)
                filtered_cloud = rospy.wait_for_message(
                    "/velodyne_points_filtered", PointCloud2, 0.1
                )
            except:
                print("Trying to get filtered cloud again...", file=sys.stderr)
        filtered_cloud_array = rnp.numpify(filtered_cloud)
        filtered_points = rnp.point_cloud2.get_xyz_points(
            filtered_cloud_array, remove_nans=False
        )
        filtered_colors_array = split_rgba_field(filtered_cloud_array)
        filtered_colors = np.zeros((len(filtered_colors_array), 3))
        filtered_mask = np.zeros(len(filtered_colors_array), dtype=bool)
        filtered_colors[:, 0] = filtered_colors_array["r"]
        filtered_colors[:, 1] = filtered_colors_array["g"]
        filtered_colors[:, 2] = filtered_colors_array["b"]
        filtered_colors /= 255
        filtered_mask = filtered_colors_array["a"]

        return filtered_points, filtered_colors, filtered_mask

    # in 4x4 quat with position form
    @staticmethod
    def compare_poses(pose1, pose2):
        p1 = pose1[:3, 3]
        p2 = pose2[:3, 3]
        d_pos = np.linalg.norm(p1 - p2)
        R1 = pose1[:3, :3]
        R2 = pose2[:3, :3]
        q1 = R.from_matrix(R1).as_quat()
        q2 = R.from_matrix(R2).as_quat()
        dot_product = np.dot(q1, q2)
        dot_product = np.clip(
            dot_product, -1.0, 1.0
        )  # ----------------------------------Added by j_user
        d_rot = 2 * np.arccos(np.abs(dot_product))
        # print("pos", d_pos, "rot", d_rot, "added", d_pos + d_rot)
        return d_pos + d_rot

    def did_use_cam_pose_before(self, prefix):
        # [j_user] Checking similarity to past camera poses used:
        if len(self.added_cam_poses) == 0:
            return False
        while prefix not in self.cam_info:
            print(prefix, self.cam_info.keys())
            rospy.sleep(0.1)
        mat = transform_to_matrix(self.cam_transform[prefix])
        # points = np.matmul(points, mat.T[:3, :3]) + mat[:3, 3]
        # difference = min(self.added_cam_poses, key=lambda x: self.compare_poses(x, mat))
        difference = min(self.compare_poses(x, mat) for x in self.added_cam_poses)
        # If the lowest difference is this low, this means we have already seen this so return true for already used
        if difference < 0.01:
            return True
        else:
            return False

    # Gets the 2D segmentation target mask with get_seg_info then integrates that with the camera information into tsdf_vol
    def updated_fused_points(
        self,
        prefix,
        target,
        ground=True,
        debug=False,
        resolution=0.005,
    ):
        if self.mode == "gt" and target:
            self.target = target
        # If we have used this camera pose before
        # if self.did_use_cam_pose_before(prefix) == True:
        #     return
        # print("New pose found")
        # If vispoints has the service fail then it doesn't integrate the new points in to make a point cloud, meaning that if the very first image has the service have no target
        # points then the system will crash.
        t0 = time.time()
        result = self.get_seg_info(prefix, target, ground, debug)
        color, depth, mask, K, T, K_rgb, T_rgb = result
        self.cur_color, self.cur_depth, self.cur_mask = color, depth, mask

        # set current points and pmask so raw segmented points can be accessed
        self.cur_points = np.empty((0, 3), dtype=np.float32)
        self.cur_pmask = np.empty(0, dtype=np.uint32)
        n_depth = np.array(depth)
        n_depth[mask.astype(bool)] = 0
        pcd = self.create_pcd(n_depth, K, color)
        points = np.array(pcd.points)
        if len(points) > 0:
            points = points @ T[:3, :3].T + T[:3, 3]
            self.cur_points = np.vstack((self.cur_points, points))
            self.cur_pmask = np.hstack((self.cur_pmask, [0] * len(points)))
        for i in range(32):
            bin_mask = ((1 << i) & mask).astype(bool)
            n_depth = np.array(depth)
            n_depth[~bin_mask] = 0
            pcd = self.create_pcd(n_depth, K, color)
            points = np.array(pcd.points)
            if len(points) == 0:
                continue
            points = points @ T[:3, :3].T + T[:3, 3]
            self.cur_points = np.vstack((self.cur_points, points))
            self.cur_pmask = np.hstack((self.cur_pmask, [1 << i] * len(points)))

        self.added_cam_poses.append(T)
        t1 = time.time()
        print("[perception_fast] get_seg_info time:", t1 - t0)

        self.times_integrated += 1

        mask = mask.astype(np.uint32)  # convert to float for integration
        self.tsdf_vol.integrate(color, depth, mask, K, T, K_rgb, T_rgb)

        t2 = time.time()
        print("[perception_fast] integrate time:", t1 - t2)
        # print("[j_user DEBUG] fused target mask", self.get_fused_target_mask())
        return mask  # return mask for debug logging

    # Get the occluded cluster to use as the center, send rays out from the center, collision check them and remove rays that touch surface points
    def rays_from_occlusion(self):
        # occl_points = self.get_occlusion_cluster_near_tgt_points()
        occl_points = self.get_occlusion_points()
        msg = PerceptionInterface.create_cloud(
            occl_points,
            np.linspace([0, 0, 0], [255, 255, 255], len(occl_points)),
            255,
            "world",
            rospy.Time.now(),
        )
        self.occl_cluster_pub.publish(msg)

        t0 = time.time()
        num_y = 12
        num_z = 6
        num_rays = num_y * num_z
        center_points = self.make_origin()
        # center_point = center_points[-1]
        # center_point = occl_points[np.argmin(distances, axis=0)]
        # print("center point shape", center_point.shape)
        # origin_torch = torch.from_numpy(center_point).unsqueeze(0) #[1, 3]
        # origin_torch = origin_torch.repeat(num_rays, 1)
        # rays, rays_torch = self.fibonacci_sphere(center_point, num_rays)
        rays = []
        end_points = []
        for center_point in center_points:
            rays_ = self.grid_of_rays(center_point, num_y=num_y, num_z=num_z)
            end_points_ = self.end_points(center_point, rays_)
            rays.append(rays_.tolist())
            end_points.append(end_points_.tolist())
        # rays,_ = self.fibonacci_sphere(num_rays)

        # Setting parameters for the collision checking
        rospy.set_param("/rays_from_perception", rays)
        rospy.set_param("/rays_endpoints_from_perception", end_points)
        rospy.set_param("/rays_origin_from_perception", center_points)

        # Getting collision checked rays
        rays_collision_free = rospy.get_param("/rays_collision_free", [])
        rays_collision_free = np.asarray(rays_collision_free)
        print(
            "[perception_fast] Rays_collision_free.shape",
            rays_collision_free.shape,
            len(rays_collision_free),
        )
        ray_largest_cone = rospy.get_param("/ray_largest_cone", [])
        ray_largest_cone = np.asarray(ray_largest_cone)
        print("Ray largest cone", ray_largest_cone, len(ray_largest_cone))
        ray_it_worked = rospy.get_param("/ray_it_worked", 0)
        center_point = center_points[ray_it_worked]
        rays = rays[ray_it_worked]

        # surface_points,_ = self.get_fused_point_cloud()
        # surface_points = torch.from_numpy(surface_points)
        # rays = self.check_collisions(origin_torch, rays_torch, surface_points)
        # print("rays", rays.shape, time.time() - t0) #0.5 seconds around
        # rays = rays.numpy()
        # print("ray mask len", len(ray_mask))
        start = np.asarray(rospy.get_param("/start_ee_pos", []))
        cur_dir = np.asarray(rospy.get_param("/cur_dir", []))
        des_dir = np.asarray(rospy.get_param("/des_dir", []))
        visualize = True
        if visualize:
            self.marker.points = []
            self.marker_cone.points = []
            if len(rays) > 0:
                # ray_mask = np.asarray(ray_mask)
                # rays = rays[ray_mask]
                # print("rays.shape", rays.shape)

                # Do start and end for markers
                end_points = self.end_points(center_point, rays)
                i = 0
                for ray in rays:
                    self.marker.points.append(
                        Point(center_point[0], center_point[1], center_point[2])
                    )
                    # print("ray.shape", ray.shape)
                    end_point = end_points[
                        i
                    ]  # center_point + (0.9 * ray) #ray[0] is origin and ray[1] is direction
                    self.marker.points.append(
                        Point(end_point[0], end_point[1], end_point[2])
                    )
                    i += 1
                if len(cur_dir) > 0:
                    self.marker.points.append(Point(start[0], start[1], start[2]))
                    print("cur_dir", cur_dir)
                    end_point = start + (
                        0.9 * cur_dir
                    )  # ray[0] is origin and ray[1] is direction
                    self.marker.points.append(
                        Point(end_point[0], end_point[1], end_point[2])
                    )
                    self.marker.points.append(Point(start[0], start[1], start[2]))
                    print("des_dir", des_dir)
                    end_point = start + (
                        0.9 * des_dir
                    )  # ray[0] is origin and ray[1] is direction
                    self.marker.points.append(
                        Point(end_point[0], end_point[1], end_point[2])
                    )
                if len(ray_largest_cone) > 0 and len(rays_collision_free) > 0:
                    # print("Best ray visualized")
                    self.marker_cone.points.append(
                        Point(center_point[0], center_point[1], center_point[2])
                    )
                    end_point = center_point + (
                        0.9 * ray_largest_cone[0]
                    )  # ray[0] is origin and ray[1] is direction
                    self.marker_cone.points.append(
                        Point(end_point[0], end_point[1], end_point[2])
                    )
            self.ray_pub_2.publish(self.marker_cone)
            self.ray_pub.publish(self.marker)

    def make_origin(self, desired_radius=1.0):
        center_points = []
        target_points, _ = self.get_fused_target_point_cloud()
        # print("target points", type(target_points), target_points.shape)
        center = np.mean(target_points, axis=0)
        print("center shape", center.shape, center)
        occluded_points = self.get_occlusion_point_cloud()
        dist = np.linalg.norm(center - occluded_points, axis=-1)  # [pts, 3]
        mask = dist < desired_radius
        filtered_occl = occluded_points[mask]
        # Farthest from robot to use
        robot_base = np.array(
            [[0.0, 0.0, 1.0]]
        )  # It seems that 0.0,0.0,1.0 is the center of the robot, and the base is probably at the origin
        # The shelf is 0.5 to -0.5 on the y axis, the x remains the same at 0.85 and the z changes from 0.95 to 1.3
        dist_from_robot = np.linalg.norm(
            filtered_occl - robot_base, axis=-1
        )  # np.mean(occl_points, axis=0)
        # center_point = filtered_occl[np.argmax(dist_from_robot, axis=0)]

        # It should be occluded point set closer to the robot center than the point of interest.
        # Then point closest to the robot as the point of interest.
        # Then the occluded point from that list that is closest to the object

        # Percentile centerpoints
        percentiles = [75, 25, 100]
        for o in percentiles:
            use_dist = np.percentile(dist_from_robot, o)
            # print("percentile", use_dist, np.argmin(dist_from_robot - use_dist), (dist_from_robot - use_dist)[0], dist_from_robot[np.argmin(dist_from_robot - use_dist)])
            closest_center_point = filtered_occl[
                np.argmin(np.abs(dist_from_robot - use_dist))
            ]  # This means the index of distances that closest matches the percentile dist indexing into occluded
            # center_point_closest = filtered_occl[np.argmax(dist_from_robot, axis=0)]
            center_points.append(closest_center_point.tolist())

        # Forward point of the target object closest to the robot
        dist_tar_robot = np.linalg.norm(target_points - robot_base, axis=-1)
        closest_tar_point = target_points[np.argmin(dist_tar_robot)]
        shortest_dist_tar_to_robot = dist_tar_robot[np.argmin(dist_tar_robot)]
        # Occl points closer to robot base than closest tar point
        mask_closer_occl = dist_from_robot < shortest_dist_tar_to_robot
        if len(mask_closer_occl) > 0:
            closer_occl = filtered_occl[mask_closer_occl]
            # Occl point of closer to the robot that is closest to closest tar point
            dist_closer_occl_to_closest_tar_point = np.linalg.norm(
                closest_tar_point - closer_occl, axis=-1
            )
            right_in_front = closer_occl[
                np.argmin(dist_closer_occl_to_closest_tar_point)
            ]
            center_points.append(right_in_front.tolist())
        return center_points

    # surface_points is torch tensor [points, 3]
    def check_collisions(
        self, origin, rays_torch, surface_points, num_steps=50, threshold=0.007
    ):
        # interpolate down the line and check if its close enough
        # in the cost function if we are close enough to one, remove the cost for that for the rest of the trajectory
        surface_points = surface_points.to(torch.float32)
        t_values = (
            torch.linspace(0.0, 0.2, num_steps).unsqueeze(0).unsqueeze(2)
        )  # Shape: (1, num_steps, 1)
        origin_resized = origin.unsqueeze(1).to(
            torch.float32
        )  # shape is [num_rays, 1, 3]
        # rays_torch shape is this: [num_rays, 3]
        rays_torch_resized = rays_torch.unsqueeze(1)  # [num_rays, 1, 3]
        # print("rays_torch_resized.shape", rays_torch_resized.shape, (t_values * rays_torch_resized).shape, t_values.shape)
        propagated_points = origin_resized + t_values * rays_torch_resized
        # print("propagated points shape", propagated_points.shape)

        propagated_points_flat = propagated_points.view(-1, 3)
        print("check collisions2", propagated_points_flat.shape, surface_points.shape)
        # Compute pairwise distances between propagated points and surface points
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        propagated_points_flat = propagated_points_flat.to(device)
        surface_points = surface_points.to(device)
        distances = torch.cdist(
            propagated_points_flat.to(torch.float32), surface_points
        )  # Shape: (num_rays * num_steps, num_points)
        distances = distances.to(torch.device("cpu"))
        # Check if any distance is below the threshold
        within_threshold = (
            distances.min(dim=1).values <= threshold
        )  # Shape: (num_rays * num_steps,)
        within_threshold = within_threshold.view(
            propagated_points.shape[:2]
        )  # Shape: (num_rays, num_steps)
        # Identify rays that collide at any step
        colliding_rays = within_threshold.any(dim=1)  # Shape: (num_rays,)

        # Filter out colliding rays
        non_colliding_rays = rays_torch[~colliding_rays]  # Keep only non-colliding rays
        return non_colliding_rays  # (num_rays, 3)

    @staticmethod
    def ik_option_points_one_ray(origin, ray, num_samples=10):  # [3], [3]
        box_min_corner = np.array([0.84, -0.46, 0.99])
        box_max_corner = np.array([0.86, 0.46, 1.16])
        # ray = ray / np.linalg.norm(ray)  # Ensure the ray is a unit vector

        # How much to scale ray direction so that it goes from the origin to this min corner or max corner
        # rays * (box_min_corner - origin) / rays is equivalent to rays / rays  * box_min_corner - origin, so it just turns into the ray that goes from origin to the box corner
        t_min = (box_min_corner - origin) / ray
        t_max = (box_max_corner - origin) / ray

        epsilon = 1e-10
        # Avoid division by zero by adding a small epsilon value where rays are zero
        ray = np.where(ray == 0, epsilon, ray)
        # Get entry and exit t values
        # The least scaling needed to get to either corner of the both so whichever gets there first, then the highest of those as all dims need to enter
        minimum_t_xyz = np.minimum(
            t_min, t_max
        )  # scaling of [3] for xyz and finding the minimum scaling for each, minimum compares two and does smallest everywhere
        # Then maximum flat scaling as we need one value to scale them all
        t_entry = np.max(minimum_t_xyz)  # maximum value [rays]
        # The largest scaling out of getting to both corners so one of those scaling is the leaving one. So then the minimum as just one dim needs to leave
        maximum_t_xyz = np.maximum(t_min, t_max)
        t_exit = np.min(maximum_t_xyz)

        # Get the final entry and exit t values
        # t_start = np.max(t_entry)
        # t_end = np.min(t_exit)

        # Check if intersection is valid
        # if t_start > t_end or t_end < 0:
        #    return np.array([])  # No valid points

        # Sample t values between t_start and t_end
        t_values = np.linspace(t_entry, t_exit, num_samples)

        # Compute the corresponding points on the ray
        points = (
            origin + t_values[:, None] * ray
        )  # t_values[:, None] = (num_samples, 1) and ray is (3)

        return points

    @staticmethod
    def single_direction_to_quaternion(direction: torch.Tensor) -> torch.Tensor:
        """
        Convert a single unit direction vector (3,) into a quaternion (4,)
        by randomly assigning the missing rotational degree of freedom.

        :param direction: Tensor of shape (3,) representing a unit direction vector.
        :return: Tensor of shape (4,) representing a quaternion.
        """
        direction = (direction / torch.norm(direction)).float()
        random_angle = torch.rand(1) * 2 * torch.pi

        reference = torch.tensor([0.0, 0.0, 1.0], device=direction.device).float()
        if torch.abs(torch.dot(direction, reference)) > 0.99:
            reference = torch.tensor([1.0, 0.0, 0.0], device=direction.device)

        right = torch.cross(reference, direction)
        right = right / torch.norm(right)

        up = torch.cross(direction, right)

        cos_half_angle = torch.cos(random_angle / 2)
        sin_half_angle = torch.sin(random_angle / 2)

        x = right[0] * sin_half_angle + up[0] * sin_half_angle
        y = right[1] * sin_half_angle + up[1] * sin_half_angle
        z = right[2] * sin_half_angle + up[2] * sin_half_angle
        w = cos_half_angle

        return torch.tensor(
            [w.item(), x.item(), y.item(), z.item()], device=direction.device
        )

    def end_points(self, origin, rays):
        max_y = (
            self.bounds[1, 1] - 0.05
        )  # 0.45#0.35#0.47 #Cannot reach much further than 0.15
        min_y = self.bounds[1, 0] + 0.05  # -0.45#-0.35#-0.47
        min_z = self.bounds[2, 0] + 0.05  # 1.0#1.0#0.95
        max_z = self.bounds[2, 1] - 0.05  # 1.15#1.2#1.3
        box_min_corner = np.array([0.7, min_y, min_z])  # np.array([0.84, -0.46, 0.99])
        box_max_corner = np.array([0.8, max_y, max_z])

        # How much to scale ray direction so that it goes from the origin to this min corner or max corner
        # rays * (box_min_corner - origin) / rays is equivalent to rays / rays  * box_min_corner - origin, so it just turns into the ray that goes from origin to the box corner
        # Compute t values for intersections with the AABB
        epsilon = 1e-10
        # Avoid division by zero by adding a small epsilon value where rays are zero
        rays = np.where(rays == 0, epsilon, rays)
        t_min = (box_min_corner - origin) / rays
        t_max = (box_max_corner - origin) / rays
        # print("rays", rays, origin)
        # print("tmin", t_min)
        # print(t_max)

        # Get entry and exit t values
        minimum_t_xyz = np.minimum(
            t_min, t_max
        )  # scaling of [3] for xyz and finding the minimum scaling for each, minimum compares two and does smallest everywhere
        # print("minimum_t_xyz", minimum_t_xyz.shape)
        # Then maximum flat scaling as we need one value to scale them all
        t_entry = np.max(
            minimum_t_xyz, axis=1
        )  # maximum value on the dim 1 axis, resulting in [rays]

        # End point
        end_point = np.expand_dims(origin, 0) + (np.expand_dims(t_entry, -1) * rays)
        # print("endpoint info", end_point[0], end_point.shape, t_entry)

        return end_point

    def grid_of_rays(self, origin, num_y, num_z):
        # The shelf is 0.5 to -0.5 on the y axis, the x remains the same at 0.85 and the z changes from 0.95 to 1.3
        # num_each = int(np.sqrt(num_rays))
        print(self.bounds)
        max_y = (
            self.bounds[1, 1] - 0.15
        )  # 0.45#0.35#0.47 #Cannot reach much further than 0.15
        min_y = self.bounds[1, 0] + 0.15  # -0.45#-0.35#-0.47
        min_z = self.bounds[2, 0] + 0.1  # 1.0#1.0#0.95
        max_z = self.bounds[2, 1] - 0.1  # 1.15#1.2#1.3
        print("maxy miny", max_y, min_y)
        y = np.linspace(min_y, max_y, num=num_y)
        z = np.linspace(min_z, max_z, num=num_z)
        gridy, gridz = np.meshgrid(y, z)
        gridy = gridy.ravel()
        gridz = gridz.ravel()
        x = np.full_like(gridy, 0.85, dtype=float)
        result = np.column_stack((x, gridy, gridz))  # [num_rays, 3]
        rays = result - origin
        normalized_rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        return normalized_rays

    def fibonacci_sphere(self, num_rays):
        """
        Generate evenly spaced rays using the Fibonacci sphere method.

        Parameters:
            origin (numpy.ndarray): The origin point of the rays (3D).
            num_rays (int): The number of evenly spaced rays to generate.

        Returns:
            list of lists of arrays: List of lists as [origin, direction], where direction is normalized.
        """
        rays = []
        rays_torch = None
        phi_golden = (1 + np.sqrt(5)) / 2  # Golden ratio

        for i in range(num_rays):
            theta = np.arccos(1 - 2 * (i + 0.5) / num_rays)  # Latitude
            phi = 2 * np.pi * i / phi_golden  # Longitude

            # Convert spherical to Cartesian
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            direction = np.array([x, y, z])

            rays.append(direction)
            direction_torch = torch.from_numpy(direction).unsqueeze(0)
            if i == 0:
                rays_torch = direction_torch
            else:
                rays_torch = torch.cat([rays_torch, direction_torch], dim=0)

        return rays_torch.numpy(), rays_torch

    def get_occlusion_cluster_near_tgt_points(self):
        """
        Get the target point cluster and add those points to the pcd here. Then cluster everything. There will be an occlusion cluster attached to the target point cloud.
        Remember that cluster attached to the target point cloud as the target cluster. Now cluster again and select the cluster that intersects with points from the target cluster.
        """
        clust_dist = 0.005
        occl_points = self.get_occlusion_point_cloud()
        target_points, _ = (
            self.get_largest_target_cluster_simple()
        )  # self.get_fused_target_point_cloud()
        print("occl points", occl_points.shape, target_points.shape)
        joined_points = np.concatenate((occl_points, target_points), axis=0)
        if len(occl_points) == 0:
            return occl_points.copy()

        clusters = tm.grouping.clusters(
            joined_points, clust_dist
        )  # Creates clusters of points with radius 5cm, list of list, each list is a cluster of indices of what points they are
        occl_clusters = tm.grouping.clusters(occl_points, clust_dist)

        if len(clusters) == 0:
            return [], []

        # Creates clusters of points with radius 1cm, list of list, each list is a cluster of indices of what points they are
        for cluster in clusters:
            joined_cluster_pts = joined_points[cluster]
            # vol_on_tgt = np.intersect1d(target_points, joined_cluster_pts)
            batch1_hashed = set(map(tuple, joined_cluster_pts))
            batch2_hashed = set(map(tuple, target_points))

            # Find intersection of the sets
            shared_points = batch1_hashed.intersection(batch2_hashed)

            # Convert back to NumPy array
            vol_on_tgt = np.array(list(shared_points))
            if vol_on_tgt.size > 0:  # This cluster includes the target points
                print(
                    "joined_cluster_pts",
                    joined_cluster_pts.shape,
                    vol_on_tgt.shape,
                    target_points.shape,
                )
                return joined_cluster_pts
                # Now joined_cluster_pts will be points to look for intersection with as we will do clustering again just with occlusion
                for oc in occl_clusters:
                    occl_pts = occl_points[oc]  # occlusion actual points
                    matching = np.intersect1d(
                        joined_cluster_pts, occl_pts
                    )  # does it match any occlusion points that were clustered with target points?
                    if matching.size > 0:
                        # print("returned target point adjacent occluded cluster")
                        return occl_pts

        # If couldn't find cluster on target object
        cluster_ind = max(clusters, key=len)
        c_points = occl_points[cluster_ind]
        print("returned basic occluded cluster")
        return c_points

    def get_occlusion_point_cloud(self):
        pcd = self.tsdf_vol.get_point_cloud()
        fused_points = pcd[:, :3]
        return fused_points

    def get_fused_point_cloud(self):
        pcd = self.tsdf_vol.get_point_cloud()
        fused_points = pcd[:, :3]
        fused_colors = pcd[:, 3:6] / 255.0
        return fused_points, fused_colors

    def get_object_instance_mask(self):
        return self.tsdf_vol.get_point_cloud()[:, 6].astype(np.uint32)

    def get_fused_target_mask(self):
        mask = self.tsdf_vol.get_point_cloud()
        # print(
        #     "[j_user DEBUG] target points from the saved tsdf",
        #     np.array(mask).shape, "number of target points", tar_mask.sum()
        # )
        return mask[:, 6] != 0

    def get_fused_target_mask_new(
        self,
    ):
        from skimage import measure
        import cc3d

        tsdf_vol, occl_vol, color_vol, mask_vol = self.tsdf_vol.get_volume()

        # marching_cubes_mask is same dims as tsdf_vol
        marching_cubes_mask = np.logical_and(
            np.logical_and(tsdf_vol > -0.5, tsdf_vol < 0.5), mask_vol > 0
        )

        # Connected components

        # Marching cubes
        verts = measure.marching_cubes(tsdf_vol, mask=marching_cubes_mask, level=0)[0]
        # verts = measure.marching_cubes(tsdf_vol, level=0)[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts * self.tsdf_vol._voxel_size + self.tsdf_vol._vol_origin

        mask = mask_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]].reshape(
            (-1, 1)
        )

    def get_fused_real_mesh_mask(self, pcd_mesh):
        pcd, _ = self.get_fused_point_cloud()
        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(pcd)
        distances = np.asarray(
            pcd_all.compute_point_cloud_distance(pcd_mesh)
        )  # for each pcd_all, the closest point to it on pcd mesh
        pcd_output = np.asarray(
            [n for n, m in zip(pcd, distances) if m <= 0.01]
        )  # fill list with values from pcd, n, if that value zipped with distances has distance value, m, < 1cm from nearest neighbor
        return pcd_output

    def get_fused_bg_point_cloud(self):
        fused_points, fused_colors = self.get_fused_point_cloud()
        fused_target_mask = self.get_fused_target_mask()
        bg_points = fused_points[~fused_target_mask]
        bg_colors = fused_colors[~fused_target_mask]
        return bg_points, bg_colors

    def get_fused_target_point_cloud(self):
        fused_points, fused_colors = self.get_fused_point_cloud()
        fused_target_mask = self.get_fused_target_mask()
        target_points = fused_points[fused_target_mask]
        target_colors = fused_colors[fused_target_mask]
        return target_points, target_colors

    @staticmethod
    def create_pcd(depth_im, cam_intr, color_im=None, cam_extr=np.eye(4)):
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.intrinsic_matrix = cam_intr
        depth_im_o3d = o3d.geometry.Image(depth_im)
        if color_im is not None:
            color_im_o3d = o3d.geometry.Image(color_im)
            rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(
                color_im_o3d,
                depth_im_o3d,
                depth_scale=1,
                convert_rgb_to_intensity=False,
            )
            pcd = o3d.geometry.PointCloud().create_from_rgbd_image(
                rgbd, intrinsic_o3d, extrinsic=cam_extr
            )
        else:
            pcd = o3d.geometry.PointCloud().create_from_depth_image(
                depth_im_o3d, intrinsic_o3d, extrinsic=cam_extr, depth_scale=1
            )
        return pcd

    """
    import psutil
    import time
    def is_memory_available(self, required_memory_gb=2):

        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        print(f"Available memory: {available_memory_gb:.2f} GB")
        return available_memory_gb >= required_memory_gb

    def wait_for_memory(self, required_memory_gb=2, check_interval=5):

        while not self.is_memory_available(required_memory_gb):
            print(f"Waiting for sufficient memory ({required_memory_gb} GB)...")
            time.sleep(check_interval)
    """

    def get_gpu_memory(self):
        try:
            # Run the nvidia-smi command and capture the output
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,nounits,noheader",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            # Parse the free memory output
            free_memory = int(result.stdout.strip().split("\n")[0])
            print(f"Free GPU memory: {free_memory} MiB")
            return free_memory
        except Exception as e:
            print(f"Error checking GPU memory: {e}")
            return None

    def wait_for_gpu_memory(self, required_memory_mib=2000, check_interval=5):
        """
        Wait until the required amount of GPU memory (in MiB) is available.
        """
        while True:
            free_memory = self.get_gpu_memory()
            if free_memory is None:
                print("Failed to retrieve GPU memory information.")
                break

            if free_memory >= required_memory_mib:
                print(f"Sufficient GPU memory available: {free_memory} MiB")
                break
            else:
                print(
                    f"Waiting for sufficient GPU memory... Required: {required_memory_mib} MiB, Available: {free_memory} MiB"
                )
                time.sleep(check_interval)

    def get_seg_info(self, prefix, target, ground=True, debug=False):
        print(f"self.cam_info: {self.cam_info} {self.cam_info.keys()}")
        while prefix not in self.cam_info:
            print(f"    sleeping...")
            rospy.sleep(0.1)

        with self.lock:
            s_cam_info = self.cam_info[prefix]
            s_color_img = self.color_img[prefix]
            s_depth_img = self.depth_img[prefix]
            s_cam_transform = self.cam_transform[prefix]
            if self.mode == "gt":
                s_seg_img = self.seg_img[prefix]
            elif self.mode == "fs":
                s_infra_info = self.infra_info[prefix]
                s_left_img = self.left_img[prefix]
                s_right_img = self.right_img[prefix]

        # Camera intrinsics and extrinsics
        K_rgb = np.array(s_cam_info.K).reshape((3, 3))
        T_rgb = transform_to_matrix(s_cam_transform)

        # Get segmentation mask
        t0 = time.time()
        robot_exclusion_mask = None
        if self.mode == "gt":
            # print("GT")
            # color = self.bridge.imgmsg_to_cv2(s_color_img, 'rgb8')
            # depth = self.bridge.imgmsg_to_cv2(s_depth_img, '32FC1')
            seg_raw = self.bridge.imgmsg_to_cv2(s_seg_img, "rgb8")
            seg = decode_seg_img_rgb(seg_raw)
            robot_exclusion_mask = seg == -1

            # scene_pcd = self.create_pcd(depth, K_rgb, color)
            # points = np.asarray(scene_pcd.points)
            # colors = np.asarray(scene_pcd.colors)

            # mask = (seg == int(self.target))
            # mask = cv2.erode(mask.astype(float), np.ones((5, 5))).astype(bool)
            # masked_depth = depth * mask
            # target_pcd = self.create_pcd(masked_depth, K_rgb, color)

            # scene_kdtree = KDTree(points)
            # target_pts = np.asarray(target_pcd.points)
            # _d, target_indices = scene_kdtree.query(target_pts)
            # target_mask = np.zeros(points.shape[0], dtype=np.bool_)
            # target_mask[target_indices] = True  #
            # print("[j_user DEBUG] target mask inside of get visible points", mask)
            # bg_mask = None  #

            mask = np.zeros_like(seg).astype(np.uint32)
            i = 1
            for obj_id in sorted(set(seg.flatten())):
                # print('obj_id,seg', obj_id, set(seg.flatten()), np.any(seg == obj_id))
                if obj_id < 0:
                    continue
                elif self.target.isdigit() and obj_id == int(self.target):
                    mask[seg == obj_id] |= 1
                    self.seg_label_to_obj_id[0] = obj_id
                else:
                    mask[seg == obj_id] |= 1 << i
                    self.seg_label_to_obj_id[i] = obj_id
                    i += 1
            # print(set(mask.flatten()))
        else:
            # print("waiting for segment3d")
            # self.wait_for_gpu_memory()
            # print("finished waiting for gpu")
            # debug_depth = self.bridge.imgmsg_to_cv2(s_depth_img, '32FC1')
            # print(debug_depth)
            # print(np.histogram(debug_depth))
            rospy.wait_for_service("detic_service", timeout=5)
            try:
                req = GetGSAMResultsRequest()
                req.target_name = String(target)
                req.cam_info = s_cam_info
                req.color_img = s_color_img
                # req.depth_img = s_depth_img
                req.debug_mode = debug
                req.ground = ground
                serv = rospy.ServiceProxy("detic_service", GetGSAMResults)
                resp = serv(req)
                success = resp.success
            except Exception as e:
                success = False
                print("Service error:", e)
            if not success:
                print("Service failed!")
                return None, None, None, None, None, None, None

            # points = np.array(resp.points.data, dtype=np.float32).reshape(-1, 3)
            # colors = np.array(resp.colors.data, dtype=np.float32).reshape(-1, 3)
            # target_mask = np.array(resp.target_mask)
            # bg_mask = np.array(resp.background_mask)
            # print(resp.target_image_mask)
            # if len(resp.target_image_mask.data) == 0:
            #     mask = np.zeros((s_cam_info.height, s_cam_info.width))
            # else:
            #     mask = self.bridge.imgmsg_to_cv2(
            #         resp.target_image_mask, "mono8"
            #     ) #Keep as array instead of image
            #     # mask = cv2.erode(mask, np.ones((5, 5)))
            H, W = s_color_img.height, s_color_img.width
            mask = np.array(resp.encoded_masks, dtype=np.uint32).reshape(H, W)
            print("bits", np.binary_repr(mask[0, 0], 32), type(s_color_img))
        t1 = time.time()
        print("[get visible points] time", t1 - t0)

        # transform points into world frame
        # points = np.matmul(points, T_rgb.T[:3, :3]) + T_rgb[:3, 3]

        # Run foundation stereo for better depth on real system
        if self.mode == "fs":
            K = np.array(s_infra_info.K).reshape((3, 3))
            T = T_rgb @ self.t_rgb2infra[prefix]
            # H, W = s_left_img.height, s_left_img.width

            with torch.no_grad():
                img0 = self.bridge.imgmsg_to_cv2(s_left_img, "mono8")
                H, W = img0.shape
                # cv2.imshow("left", img0)
                img0 = torch.as_tensor(img0).cuda().float()[None]
                img0 = img0.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
                img1 = self.bridge.imgmsg_to_cv2(s_right_img, "mono8")
                # cv2.imshow("right", img1)
                img1 = torch.as_tensor(img1).cuda().float()[None]
                img1 = img1.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
                padr = InputPadder(img0.shape, divis_by=32, force_square=False)
                img0, img1 = padr.pad(img0, img1)
                # cv2.waitKey(0)

                with torch.cuda.amp.autocast(True):
                    disp = self.fs.forward(img0, img1, iters=5, test_mode=True)
                disp = padr.unpad(disp.float())
                disp = disp.data.cpu().numpy().reshape(H, W)

            # zero out non-overlapping observations between left and right images
            yy, xx = np.meshgrid(
                np.arange(disp.shape[0]),
                np.arange(disp.shape[1]),
                indexing="ij",
            )
            us_right = xx - disp
            invalid = us_right < 0
            disp[invalid] = 0

            depth = K[0, 0] * self.baseline[prefix] / disp
            # cv2.imshow("depth", depth)
            # cv2.waitKey(0)

            # filter out hand
            if prefix == "d435":
                ee_mask = (depth < 0.12).astype(float)
                ee_mask = cv2.dilate(ee_mask, np.ones((25, 25))).astype(bool)
                # cv2.imshow('before', depth)
                # depth[ee_mask] = 255
                # cv2.imshow('after', depth)
                # cv2.waitKey()
                depth[ee_mask] = 0
        else:
            K = K_rgb
            T = T_rgb
            depth = self.bridge.imgmsg_to_cv2(s_depth_img, "32FC1")
            depth /= 1000.0
            if robot_exclusion_mask is not None:
                depth = np.array(depth, copy=True)
                depth[robot_exclusion_mask] = 0
            # depth[depth == 0] = np.nan
        color = self.bridge.imgmsg_to_cv2(s_color_img, "bgr8")

        # masks = []
        # for i in range(32):
        #     imask = mask == (1 << i)
        #     if np.count_nonzero(imask) > 0:
        #         masks.append(imask)
        # fmasks = remove_thin_parts(masks)
        # mask = np.zeros_like(mask).astype(np.uint32)
        # for i, fmask in enumerate(fmasks):
        #     mask[fmask] |= 1 << i

        return color, depth, mask, K, T, K_rgb, T_rgb

    def update_occlusion(self, prefix=None):
        pass

    def get_occlusion_points(self):
        tsdf_vol, occl_vol, color_vol, mask_vol = self.tsdf_vol.get_volume()
        ix, iy, iz = ((-100 < occl_vol) & (occl_vol < 0)).nonzero()
        verts = np.array(list(zip(ix, iy, iz)))
        if len(verts) == 0:
            return np.empty((0, 3))
        verts = verts * self.tsdf_vol._voxel_size + self.tsdf_vol._vol_origin
        return verts

    def get_object_occlusion_mask(self):
        tsdf_vol, occl_vol, color_vol, mask_vol = self.tsdf_vol.get_volume()
        ix, iy, iz = ((-100 < occl_vol) & (occl_vol < 0)).nonzero()
        mask = mask_vol[ix, iy, iz].astype(np.uint32)
        return mask

    @staticmethod
    def look_at(eye, center, up):
        eye = np.asarray(eye)
        center = np.asarray(center)
        forward = center - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.asarray(up) / np.linalg.norm(up)
        up = np.cross(right, forward)
        m = np.eye(4, 4)
        m[:3, 0] = right
        m[:3, 1] = -up
        m[:3, 2] = forward
        # m[:3, 0] = -up
        # m[:3, 1] = forward
        # m[:3, 2] = right
        # print(eye)
        m[:3, 3] = eye  # [eye[0], -eye[1], -eye[2]]
        return matrix_to_pose(m)

    @staticmethod
    def sample_views(center, radii, angles):
        # Up/down
        thetas = np.deg2rad(angles)
        # Left/right
        phis = np.arange(0, 9, 1) * np.deg2rad(45)
        view_candidates = []
        for r, theta, phi in itertools.product(radii, thetas, phis):
            eye = (
                center
                + np.r_[
                    r * sin(theta) * cos(phi),
                    r * sin(theta) * sin(phi),
                    r * cos(theta),
                ]
            )
            up = np.r_[1.0, 0.0, 0.0]
            view = PerceptionInterface.look_at(eye, center, up)
            view_candidates.append(view)
        return view_candidates

    @staticmethod
    def project_point_cloud_to_image(
        points,
        colors,
        cam_intr,
        cam_pose,
        clear_low_z=0.01,
        visualize=False,
    ):
        if clear_low_z:
            inds = points[:, 2] > min(points[:, 2]) + clear_low_z
            points = points[inds]
            colors = colors[inds]

        x = cam_pose.position.x
        y = cam_pose.position.y
        z = cam_pose.position.z
        qw = cam_pose.orientation.w
        qx = cam_pose.orientation.x
        qy = cam_pose.orientation.y
        qz = cam_pose.orientation.z
        t0 = time.time()
        # easier to transform points rather than camera extrinsic
        mat = tf.quaternion_matrix([qw, qx, qy, qz])
        mat[:3, 3] = [x, y, z]
        mat = np.linalg.inv(mat)
        points = np.matmul(points, mat.T[:3, :3]) + mat[:3, 3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        t1 = time.time()
        # setup camera parameters
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        fx = cam_intr[0, 0]
        fy = cam_intr[1, 1]
        cx = cam_intr[0, 2]
        cy = cam_intr[1, 2]
        intrinsic.set_intrinsics(1920, 1080, fx, fy, cx, cy)
        extrinsic = np.eye(4)
        # extrinsic[:3, 3] = np.array([x, y, z])
        t2 = time.time()
        # set up visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=visualize)
        ctr = vis.get_view_control()
        t3 = time.time()
        # add points to scene
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 50.0
        vis.update_geometry(pcd)
        t4 = time.time()
        # set camera parameters
        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.intrinsic = intrinsic
        camera_params.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(camera_params, True)
        t5 = time.time()
        # render and capture image
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(do_render=True)
        t6 = time.time()
        # run 3D visualizer
        if visualize:
            vis.run()
        t7 = time.time()
        print(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6)
        return np.array(img)

    @staticmethod
    def np_int_to_rgba(a):
        arr = np.array(a, dtype="int")
        shape = (*arr.shape, 4)
        return np.frombuffer(arr.astype(">I").tobytes(), dtype=">B").reshape(shape)

    @staticmethod
    def np_rgb_to_int(a):
        arr = np.array(a.reshape(-1, 3), dtype="uint8")
        shape = arr.shape[:-1]
        arr = np.concatenate([np.zeros((len(arr), 1)), arr], axis=1)
        return np.frombuffer(arr.astype(">B").tobytes(), dtype=">I").reshape(shape)

    def compile(self, camera_name, downsample=20):

        cam_intr = np.array(self.cam_info[camera_name].K).reshape((3, 3))
        fx = cam_intr[0, 0]
        fy = cam_intr[1, 1]
        cx = cam_intr[0, 2]
        cy = cam_intr[1, 2]

        fx = fx / downsample
        fy = fy / downsample
        cx = cx / downsample
        cy = cy / downsample

        # Trigger the JIT compilation
        PerceptionInterface.raycast(
            1.0,
            np.zeros(3),
            np.zeros((225, 397, 206), dtype=np.float32),
            np.eye(3),
            np.zeros(3),
            fx,
            fy,
            cx,
            cy,
            # 0,
            # 1,
            # 0,
            # 1,
        )

    @staticmethod
    @jit(nopython=True)
    def raycast(
        voxel_size,
        voxel_origin,
        tsdf_grid,
        ori,
        pos,
        fx,
        fy,
        cx,
        cy,
        u_min=0,
        u_max=1920,
        v_min=0,
        v_max=1080,
    ):
        t_min = 0
        t_max = 0.5
        t_step = np.sqrt(3) * voxel_size
        voxel_indices = []
        # voxel_indices = set()
        for u in range(u_min, u_max):
            for v in range(v_min, v_max):
                direction = np.asarray([(u - cx) / fx, (v - cy) / fy, 1.0])
                direction = ori @ (direction / np.linalg.norm(direction))
                t = t_min
                while t < t_max:
                    p = pos + t * direction
                    t += t_step
                    # index = get_voxel_at(voxel_size, p, voxel_origin, lim=tsdf_grid.shape)
                    # Same as get_voxel_at. get_voxel_at was conflicting with just-in-time compilation
                    index = np.round((p - voxel_origin) / voxel_size)
                    index = index.astype(np.int64)
                    index = (
                        index
                        if np.all(index >= 0)
                        and np.all(index < np.array(tsdf_grid.shape))
                        else None
                    )
                    if index is not None:
                        i, j, k = index
                        if tsdf_grid[i, j, k] == 1:
                            break
                        voxel_indices.append(index)
                        # voxel_indices.add(index)
        return voxel_indices

    # @staticmethod
    # # @njit(parallel=True)
    # def cam2pix(cam_pts, intr):
    #     """Convert camera coordinates to pixel coordinates.
    #     """
    #     intr = intr.astype(np.float32)
    #     fx, fy = intr[0, 0], intr[1, 1]
    #     cx, cy = intr[0, 2], intr[1, 2]
    #     pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
    #     for i in range(cam_pts.shape[0]):
    #         pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
    #         pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
    #     return pix

    @staticmethod
    def cam2pix(cam_pts, intr):
        """Convert camera coordinates to pixel coordinates."""
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]

        # Extract camera points and apply the transformation
        x = cam_pts[:, 0]
        y = cam_pts[:, 1]
        z = cam_pts[:, 2]

        # Vectorized computation for pixel coordinates
        pix_x = np.round((x * fx / z) + cx).astype(np.int64)
        pix_y = np.round((y * fy / z) + cy).astype(np.int64)

        # Stack the results into a 2D array of pixel coordinates
        pix = np.column_stack((pix_x, pix_y))

        return pix

    def get_info_gain3(
        self, srf_pts, all_pts, cam_intr, camera_poses, cam_dims=(1920, 1080)
    ):
        """
        cam_intr is the camera intrinsic matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        camera_poses is a list of poses of the camera in the world frame"""

        if not isinstance(camera_poses[0], np.ndarray):
            c_valid_views = [pose_to_matrix(view) for view in c_valid_views]

        fx = cam_intr[0, 0]
        fy = cam_intr[1, 1]
        cx = cam_intr[0, 2]
        cy = cam_intr[1, 2]

        num_occl_pts = len(all_pts) - len(srf_pts)
        # color_mask = np.concatenate(
        #     [
        #         [255, 255, 255] * np.ones_like(srf_pts),
        #         PerceptionInterface.np_int_to_rgba(range(num_occl_pts))[:, 1:] / 255,
        #     ]
        # )
        id_mask = np.concatenate([[0] * len(srf_pts), np.arange(1, num_occl_pts + 1)])

        # Projecting all points from the world frame to the camera frame
        all_pts_camera = np.concatenate([all_pts, np.ones((len(all_pts), 1))], axis=1)
        # camera_poses is (poses, 4, 4)
        # all_pts_camera.T is (4, points)
        # print("[k_user] ================", np.stack(camera_poses).shape)
        # print("[k_user] ================", camera_poses)
        all_pts_camera = (np.stack(camera_poses) @ all_pts_camera.T).T

        # Projecting all points from the camera frame to the 2D image frame
        all_pts_pixel = (cam_intr @ all_pts_camera[:, :3].T).T
        all_pts_pixel = all_pts_pixel[:, :2] / all_pts_pixel[:, 2:]

        # print("CAM INTRINSICS", cam_intr)

        pixel_colors = np.zeros(cam_dims)
        for pose in camera_poses:
            pixel_colors[:, :] = 0

            pos = pose[:3, 3]

            # TODO: distance computation can be vectorized
            dists = np.linalg.norm(all_pts - pos, axis=1)
            sorted_indices = np.argsort(dists)[::-1]
            sorted_pts = all_pts[sorted_indices]

    def get_info_gain2(
        self,
        camera_name,
        camera_poses,
        bbox=None,
        u_min=0,
        u_max=1920,
        v_min=0,
        v_max=1080,
        downsample=1,
        ACTIVE_GRASP_VOXEL_SIZE=0.0075,
    ):

        tot_raycast_time = 0

        tA = time.time()
        cam_intr = np.array(self.cam_info[camera_name].K).reshape((3, 3))
        fx = cam_intr[0, 0]
        fy = cam_intr[1, 1]
        cx = cam_intr[0, 2]
        cy = cam_intr[1, 2]

        fx = fx / downsample
        fy = fy / downsample
        cx = cx / downsample
        cy = cy / downsample
        cam_intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        tB = time.time()
        print(
            "[k_user] Intrinsics retrieval and pixel-space downsampling took",
            tB - tA,
            " seconds",
        )

        tsdf_vol, occl_vol, color_vol, mask_vol = self.tsdf_vol.get_volume()

        tC = time.time()
        print("[k_user] Getting volumes took", tC - tB, " seconds")

        # print("[k_user]", tsdf_vol.shape)

        tsdf_grid = np.zeros_like(tsdf_vol)
        tsdf_grid[np.logical_and(tsdf_vol > -0.5, tsdf_vol < 0.5)] = 1
        tsdf_grid[tsdf_grid < 0] = -1

        tsdf_origin = self.tsdf_vol._vol_origin.copy()
        voxel_size = self.tsdf_vol._voxel_size
        if bbox is not None:

            bbox_min, bbox_max = bbox

            tsdf_min_idx = np.round((bbox_min - tsdf_origin) / voxel_size).astype(
                np.int64
            )
            tsdf_max_idx = np.round((bbox_max - tsdf_origin) / voxel_size).astype(
                np.int64
            )

            tsdf_min_idx = np.clip(tsdf_min_idx - 1, 0, np.array(tsdf_grid.shape) - 1)
            tsdf_max_idx = np.clip(tsdf_max_idx + 1, 0, np.array(tsdf_grid.shape) - 1)

            tsdf_origin = tsdf_origin + self.tsdf_vol._voxel_size * tsdf_min_idx
            # Crop to bounding box centered around reconstruction volume, with size 30cm x 30cm x 30cm
            tsdf_grid = tsdf_grid[
                tsdf_min_idx[0] : tsdf_max_idx[0] + 1,
                tsdf_min_idx[1] : tsdf_max_idx[1] + 1,
                tsdf_min_idx[2] : tsdf_max_idx[2] + 1,
            ]
            # Trim to ActiveGrasp size of 40x40x40
            # Take indices that are multiples of ACTIVE_GRASP_VOXEL_SIZE / voxel_size
            scaling_factor = ACTIVE_GRASP_VOXEL_SIZE / voxel_size
            assert np.isclose(scaling_factor, round(scaling_factor)), ValueError(
                "Voxel size should be a multiple of ACTIVE_GRASP_VOXEL_SIZE. (ACTIVE_GRASP_VOXEL_SIZE, voxel_size, quotient[])",
                ACTIVE_GRASP_VOXEL_SIZE,
                voxel_size,
                ACTIVE_GRASP_VOXEL_SIZE / voxel_size,
            )
            # print(tsdf_grid.shape, "BEFORE ====================================", int(ACTIVE_GRASP_VOXEL_SIZE / voxel_size))
            tsdf_grid = tsdf_grid[
                :: int(ACTIVE_GRASP_VOXEL_SIZE / voxel_size),
                :: int(ACTIVE_GRASP_VOXEL_SIZE / voxel_size),
                :: int(ACTIVE_GRASP_VOXEL_SIZE / voxel_size),
            ]
            voxel_size = ACTIVE_GRASP_VOXEL_SIZE
            # print(tsdf_grid.shape, "AFTER ASDFASDF ============================")

            # print("[k_user] tsdf min idx", tsdf_min_idx)
            # print("[k_user] tsdf max idx", tsdf_max_idx)

            # set new tsdf_origin
            # crop the tsdf_grid

        tD = time.time()
        print(
            "[k_user] Constructing tsdf grid took",
            tD - tC,
            " seconds. Cropped TSDF dimensions:",
            tsdf_grid.shape,
        )

        scores = []
        for cam_pose in camera_poses:
            print("[k_user] ====================================")
            t0 = time.time()
            if not isinstance(cam_pose, np.ndarray):
                homog = pose_to_matrix(cam_pose)
            else:
                homog = cam_pose

            # R = homog[:3, :3]
            # t = homog[:3, 3]
            R = np.ascontiguousarray(homog[:3, :3])
            t = np.ascontiguousarray(homog[:3, 3])

            print("[k_user] Contiguous R, t", time.time() - t0)

            u_min_i = u_min
            v_min_i = v_min
            u_max_i = u_max
            v_max_i = v_max

            if bbox is not None:
                bbox_min, bbox_max = bbox
                # (8, 3)
                bbox_world = np.stack(list(itertools.product(*zip(bbox_min, bbox_max))))

                mat = np.linalg.inv(homog)
                points = np.matmul(bbox_world, mat.T[:3, :3]) + mat[:3, 3]
                bbox_world_cam = points

                # TODO: check for consistency in pixel coordinates
                # cam_bbox = fusion.TSDFVolume.cam2pix(bbox_world_cam, cam_intr)
                cam_bbox = self.cam2pix(bbox_world_cam, cam_intr)

                # Bounding box minimums
                # Take max between default (0) and min u value obtained by projecting world bbox into pixels
                u_min_i = max(u_min, np.min(cam_bbox, axis=0)[0])
                v_min_i = max(v_min, np.min(cam_bbox, axis=0)[1])

                # Bounding box maximums
                # Take min between default and max value obtained by projecting world bbox into pixels
                u_max_i = min(u_max, np.max(cam_bbox, axis=0)[0])
                v_max_i = min(v_max, np.max(cam_bbox, axis=0)[1])

                # bbox_world_cam is (8, 3)
                # Checking that the bounding box is in front of the camera
                if np.all(bbox_world_cam[2, :] < 0):
                    print("[k_user] Bounding box is behind the camera")
                    scores.append(0)
                    continue

                if (
                    u_min_i > u_max
                    or u_max_i < u_min
                    or v_min_i > v_max
                    or v_max_i < v_min
                ):
                    print(
                        "[k_user] Bounding box is not in the camera view, based on pixel values"
                    )
                    scores.append(0)
                    continue

            print("[k_user]", time.time() - t0)

            # print('1')
            t00 = time.time()
            voxel_indices = self.raycast(
                # self.tsdf_vol._voxel_size,
                voxel_size,
                # self.tsdf_vol._vol_origin,
                tsdf_origin,
                tsdf_grid,
                R,
                t,
                fx,
                fy,
                cx,
                cy,
                u_min=u_min_i,
                u_max=u_max_i,
                v_min=v_min_i,
                v_max=v_max_i,
            )
            t01 = time.time()
            tot_raycast_time += t01 - t00
            print("[k_user] raycast took", t01 - t00, "seconds")

            # print('2', t1 - t0)

            # Count rear side voxels within the bounding box
            # N, 3 or (0,)
            indices = np.unique(voxel_indices, axis=0)
            if indices.shape[0] == 0:
                scores.append(0)
                continue

            print("[k_user] uniqueness check took", time.time() - t01, "seconds")
            t02 = time.time()
            i, j, k = indices.T
            tsdfs = tsdf_grid[i, j, k]
            ig = (tsdfs < 0).sum()
            scores.append(ig)

            print("[k_user] info gain took", time.time() - t02, "seconds")

            t1 = time.time()
            print(
                f"[k_user] active_grasp computed a view score ({ig}) in ",
                t1 - t0,
                "seconds",
            )

        print("[k_user] get_info_gain2:", time.time() - tA, "seconds")
        print("[k_user] get_info_gain2 raycast time:", tot_raycast_time, "seconds")

        # Count number of occluded voxels in the TSDF grid to get upper bound on information gain
        min_ig = 0
        max_ig = -np.sum(tsdf_grid[tsdf_grid == -1])
        return scores, (min_ig, max_ig)

    @staticmethod
    def get_info_gain(srf_pts, all_pts, cam_intr, camera_poses, tgt_bbox=None):
        # cam_intr = np.array(self.cam_info[camera_name].K).reshape((3, 3))
        print("[k_user] get_info_gain")

        num_occl_pts = len(all_pts) - len(srf_pts)
        color_mask = np.concatenate(
            [
                [255, 255, 255] * np.ones_like(srf_pts),
                PerceptionInterface.np_int_to_rgba(range(num_occl_pts))[:, 1:] / 255,
            ]
        )

        if tgt_bbox is not None:
            inside_bbox_mask = np.all(
                (all_pts >= tgt_bbox[0]) & (all_pts <= tgt_bbox[1]), axis=1
            )

            color_mask[~inside_bbox_mask] = [255, 255, 255]

        # color_mask = np.full((len(all_pts), 3), 255)

        # tgt_idx_mask = np.arange(len(all_pts)) > len(srf_pts)
        # tgt_mask = tgt_idx_mask & inside_bbox_mask
        # num_ones = np.sum(tgt_mask)
        # if num_ones > 0:
        #     color_mask[tgt_mask] = PerceptionInterface.np_int_to_rgba(range(num_ones))[:, 1:] / 255

        print("[k_user] color mask computed")
        # all_pts = np.concatenate([srf_pts, occ_pts])

        scores = []
        for cam_pose in camera_poses:
            t_start = time.time()
            img = PerceptionInterface.project_point_cloud_to_image(
                all_pts, color_mask, cam_intr, cam_pose
            )
            score = len(set(PerceptionInterface.np_rgb_to_int(img * 255)))
            scores.append(score)

            t_end = time.time()
            print(
                "[k_user] projection computed a view score in ",
                t_end - t_start,
                "seconds",
            )
        print("[k_user] view scores computed")

        return scores

    def get_largest_target_cluster(self):
        # get largest cluster in point cloud
        points, colors = self.get_fused_target_point_cloud()
        if len(points) == 0:
            return points.copy(), points.copy()
        clusters = tm.grouping.clusters(
            points, 0.01
        )  # Creates clusters of points with radius 1cm, list of list, each list is a cluster of indices of what points they are
        if self.centroid is None:  # Chooses largest cluster
            cluster_ind = max(
                clusters, key=len
            )  # which list has the most point indices in it
        else:
            cluster_ind = min(
                clusters,
                key=lambda x: np.linalg.norm(
                    np.mean(points[x], axis=0) - self.centroid
                ),
            )  # Chooses cluster with centroid nearest to previous centroid, getting the minimum distance one (this cluster's value x is a list, index points with the list and average them)
        c_points = points[cluster_ind]
        c_colors = colors[cluster_ind]
        self.centroid = np.mean(c_points, axis=0)
        return c_points, c_colors

    def get_UCB_values(self):
        mask = self.tsdf_vol.get_point_cloud()
        nonzero_mask_ind = mask[:, 6] > 0
        nonzero_mask = mask[nonzero_mask_ind]
        if len(nonzero_mask) == 0:
            return [], []
        UCB = (
            -1 * np.sqrt(float(self.times_integrated) / nonzero_mask[:, 7])
        ) + 1  # all values here must be seen before as these are labeled as correct object at least once
        voting = nonzero_mask[:, 6] / nonzero_mask[:, 7]
        scaling = 0.3
        values = (scaling * UCB) + voting
        # print("UCB and voting", UCB[0:10] * scaling, voting[0:10])
        # print("self.times_integrated", self.times_integrated)
        return values, nonzero_mask

    def get_bounded_points(self, range=-1, av=False):
        values, points = self.get_UCB_values()
        if len(values) == 0:
            return [], []
        threshold = None
        if range != -1:
            highest = max(values)
            threshold = highest - range
        elif av == True:
            threshold = np.mean(values)
        indices = values >= threshold
        out = points[indices]
        out_pts = out[:, 0:3]
        out_colors = out[:, 3:6] / 255.0
        return out_pts, out_colors

    def get_cluster_on_best_spot(self):
        # get largest cluster in point cloud
        """
        Maybe just cluster all values that have ever been seen as a target and choose the cluster that has the highest voting * confidence.
        The confidence will only scale the voting value, not add to it. Then with the cluster, can trim the some range of low percentile points.
        """
        # Then add any points that are at a lower threshold but not at or higher than the other threshold to not repeat any points and add them if they are close enough to any point there
        best_points, best_colors = self.get_bounded_points(0.02)
        if len(best_points) == 0:
            return np.array([]), np.array([])
        # print("best points", best_points.shape)
        clusters = tm.grouping.clusters(
            best_points, 0.05
        )  # Creates clusters of points with radius 5cm, list of list, each list is a cluster of indices of what points they are
        if len(clusters) == 0:
            best_points, best_colors = self.get_bounded_points(0.0)
            # best_points is always [pts, 3]
            # clusters should be [[cluster_pts], ...[cluster_pts]]
            clusters = [[0]]
        # for i in clusters:
        #     print("clusters shape", len(i))
        cluster_ind = max(
            clusters, key=len
        )  # List of points indices for the best cluster of points
        best_pt_cluster = best_points[cluster_ind]
        # print("best pt cluster shape", best_pt_cluster.shape)

        fill_in_points, fill_in_colors = self.get_bounded_points(-1, True)
        fill_in_clusters = tm.grouping.clusters(fill_in_points, 0.05)
        for clust in fill_in_clusters:
            pts = fill_in_points[clust]
            # print("pts shape", pts.shape)
            common_values = np.intersect1d(best_pt_cluster, pts)
            if common_values.size > 0:
                return pts, fill_in_colors[clust]

        return np.array([]), np.array([])
        # Get the value range to try
        # Get the points from the range
        # Get the largest cluster from the range or maybe we just should choose the highest value
        # Then find the average value for points
        # Then take values of this average value threshold as points and cluster them
        # Take the cluster that includes one of the values from the best point cluster

    def get_largest_target_cluster_simple(self):
        # get largest cluster in point cloud
        """
        Maybe just cluster all values that have ever been seen as a target and choose the cluster that has the highest voting * confidence.
        The confidence will only scale the voting value, not add to it. Then with the cluster, can trim the some range of low percentile points.
        """
        High_req = 0.8
        Low_req = 0.5
        points, colors = self.get_fused_target_point_cloud()
        if len(points) == 0:
            return points.copy(), points.copy()

        clusters = tm.grouping.clusters(
            points, 0.05
        )  # Creates clusters of points with radius 5cm, list of list, each list is a cluster of indices of what points they are

        if len(clusters) == 0:
            return [], []

        cluster_ind = max(clusters, key=len)
        c_points = points[cluster_ind]
        c_colors = colors[cluster_ind]
        return c_points, c_colors

    @staticmethod
    def get_symmetric_shape(points, colors=None):
        cloud = tm.points.PointCloud(points, colors)
        if colors is not None:
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
            * (tm.proximity.signed_distance(cloud.convex_hull, [s.center_mass])[0] > 0),
        )

        # rotate point cloud by centroid of bounding volume
        rot = tm.transformations.rotation_matrix(np.pi, [0, 0, 1], shape.center_mass)
        if colors is not None:
            centroid = tm.primitives.Sphere(radius=0.005, center=shape.center_mass)
            tm.scene.Scene([cloud, centroid, shape]).show()
        cloud.apply_transform(rot)

        # create symmetric point cloud from original and rotates points
        new_points = np.concatenate([points, cloud.vertices])
        new_colors = np.concatenate([colors, cloud.colors[:, :3] / 255.0])
        new_cloud = tm.points.PointCloud(new_points, new_colors)
        if colors is not None:
            tm.scene.Scene([new_cloud, new_cloud.bounding_primitive]).show()

        # return points and smallest volume bounding primitive of symmetric point cloud
        return new_cloud.bounding_primitive, new_cloud

    @staticmethod
    def get_shape_estimate(points, colors=None, dim=3, scale=1):
        if len(points) < 3:
            return None
        cloud = tm.points.PointCloud(points, colors)
        if len(points) < 4:
            return cloud.bounding_box_oriented
        return cloud.convex_hull

        # densly sample points
        samples = tm.sample.volume_rectangular(
            [2, 2, 1] * cloud.bounding_box.extents,
            10000,
            cloud.bounding_box.transform,
        )
        # TODO: change to sample from [2x,2x,distance to ground]

        # sample convex hull surface points
        conv_hull_points = tm.sample.sample_surface(cloud.convex_hull, 1000)[0]
        combined_points = np.concatenate([points, conv_hull_points])

        # get distance to nearest point in cloud for each convex hull point
        c_kdtree = KDTree(points[:, :dim])
        d, i = c_kdtree.query(combined_points[:, :dim], k=1)
        d *= scale

        # get points around convex hull points within respective distances d
        s_kdtree = KDTree(samples[:, :dim])
        ball_points = s_kdtree.query_ball_point(combined_points[:, :dim], d)
        inds = np.array(list(set(np.concatenate(ball_points).astype(int))))
        samples = np.concatenate([samples[inds], cloud.vertices])

        new_cloud = tm.points.PointCloud(samples)
        if colors is not None:
            to_show = []
            to_show.append(cloud)
            to_show.append(new_cloud.convex_hull)
            to_show.append(new_cloud.bounding_primitive)
            tm.scene.Scene(to_show).show()

        return new_cloud.convex_hull

    # [j_user] Evaluation of target point cloud function
    # The pipeline is to save the resulting position and rotation from adjust_object.py into an xml then load that xml, can just do it in the original xml and just change the values
    # Also remember to change the body name to the correct value
    def evaluate_target_pcd(self, mj_model, body_name, pcd_tar_raw):
        # Turn xml file into pcd vertices, get the rotation and position out too
        # Get pos and quat from xml
        # mj_model = mujoco.MjModel.from_xml_path(xml_file) #Current directory is in task_planner
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        body_pos = mj_model.body_pos[body_id]  # [x, y, z]
        body_quat = mj_model.body_quat[body_id]  # [w, x, y, z]
        rotation_matrix = R.from_quat(body_quat).as_matrix()

        # Get mesh vertices from xml
        geom_id = mj_model.body_geomadr[
            body_id
        ]  # This gets the first geom address but here I just assume it is one geom associated
        mesh_id = mj_model.geom_dataid[geom_id]
        start_vert = mj_model.mesh_vertadr[mesh_id]  # Start vertex index for this mesh
        vert_count = mj_model.mesh_vertnum[mesh_id]
        vertices = mj_model.mesh_vert[start_vert : start_vert + vert_count].reshape(
            -1, 3
        )  # so go from start vertex for that many vertices to get all the vertices, reshape [vertices, xyz]

        # transform the vertices with the position and rotation
        transformed_vertices = (rotation_matrix @ vertices.T).T + body_pos

        #####-----Now the meshes are given a rotation initially in euler angles so need to fix that
        # geom_quat = mj_model.geom_quat[geom_id]
        # rotation_matrix_from_euler = R.from_euler('xyz', [0.01, 0.02, 0.0]).as_matrix()
        # transformed_vertices = (rotation_matrix_from_euler @ transformed_vertices.T).T
        #####

        # Turn to pcd
        pcd_mesh = o3d.geometry.PointCloud()
        pcd_mesh.points = o3d.utility.Vector3dVector(
            transformed_vertices
        )  # The mesh pcd
        # print("transformed vertices", transformed_vertices)
        pcd_gt_list = self.get_fused_real_mesh_mask(
            pcd_mesh
        )  # The mesh pcd points that are close enough to the real pcd points
        pcd_gt = o3d.geometry.PointCloud()
        # print("pcd_gt_list", pcd_gt_list) #There are mesh pcd values but the is a zero pcd_gt_list which makes sense because the visualization we see is after it jumps
        pcd_gt.points = o3d.utility.Vector3dVector(
            pcd_gt_list
        )  # The parts that we would have perfectly seen of the mesh

        # get target pcd and turn it into the proper form for comparison
        # pcd_tar_raw, _ = self.get_fused_target_point_cloud() #This is not the clustering I am doing which was misleading
        pcd_tar = o3d.geometry.PointCloud()
        pcd_tar.points = o3d.utility.Vector3dVector(pcd_tar_raw)

        metric_1 = self.compute_rmse(pcd_gt, pcd_tar)
        percent_tp, num_tp = self.percent_matched(pcd_gt, pcd_tar)
        percent_fp, num_fp = self.percent_wrong_points(pcd_gt, pcd_tar)

        return metric_1, percent_tp, num_tp, percent_fp, num_fp, pcd_gt_list

    def compute_rmse(self, pcd_gt, pcd_tar):
        distances = np.asarray(
            pcd_gt.compute_point_cloud_distance(pcd_tar)
        )  # This does nearest neighbor to the other pcd for each point then keeps those distances
        rmse = np.sqrt(np.mean(distances**2))
        return rmse

    # Percent of the gt points that have tar points close enough to them
    def percent_matched(self, pcd_gt, pcd_tar):
        distances = np.asarray(
            pcd_gt.compute_point_cloud_distance(pcd_tar)
        )  # This does nearest neighbor to the other pcd for each point then keeps those distances
        hyperparameter = 0.01  # 1cm?
        close_enough = np.abs(distances) <= hyperparameter
        # print("matched distances", distances)
        return (
            (np.sum(close_enough) / len(distances)) * 100.0
            if len(distances) > 100.0
            else 0
        ), np.sum(close_enough)

    # Percent of the tar points that are too far from any gt point
    def percent_wrong_points(self, pcd_gt, pcd_tar):
        distances = np.asarray(
            pcd_tar.compute_point_cloud_distance(pcd_gt)
        )  # This does nearest neighbor to the other pcd for each point then keeps those distances
        hyperparameter = 0.01  # 1cm?
        too_far = np.abs(distances) > hyperparameter

        return (
            (np.sum(too_far) / len(distances)) * 100.0 if len(distances) > 100.0 else 0
        ), np.sum(too_far)


def get_voxel_at(voxel_size, p, voxel_origin, lim=(225, 397, 206)):
    index = np.round((p - voxel_origin) / voxel_size)
    index = index.astype(np.int64)
    print("asdfasdf", index)
    index = index if np.all(index >= 0) and np.all(index < np.array(lim)) else None
    return index
