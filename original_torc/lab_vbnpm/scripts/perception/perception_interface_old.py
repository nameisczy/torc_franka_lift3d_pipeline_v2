#!/usr/bin/env python
"""
a simplified interface for handling perception node as a separate server
This is to acheive real-time communication with the execution scene, instead of
doing discrete perception actions.
Provide services and topics to communicate with the task planner.
----------------------------------------------------------------------------
ROS parameters:
- scene resols
- obj resols
- publish workspace information through ROS param
ROS services:
- get scene occlusion
- get obj model, poses
Basic functions:
- on reading RGBD images, ask the perception node to update the occlusion
----------------------------------------------------------------------------
"""
import os
import gc
import sys
import cv2
import numpy as np

np.float = float
import ros_numpy as rnp
import open3d as o3d
import transformations as tf
from cv_bridge import CvBridge

import rospy
import rospkg
import message_filters
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, PointCloud2
from lab_vbnpm.msg import VoxelTSDF, ObjectTSDF, ObjectTSDFs
from lab_vbnpm.msg import VoxelGridBool, ObjectModel, ObjectModels
from lab_vbnpm.msg import ObjectOcclusion, ObjectOcclusions, SceneOcclusion
from lab_vbnpm.srv import GetObjectTSDFs, GetObjectTSDFsResponse
from lab_vbnpm.srv import GetObjectModels, GetObjectModelsResponse
from lab_vbnpm.srv import GetSceneOcclusion, GetSceneOcclusionResponse
from lab_vbnpm.srv import GetScenePointCloud, GetScenePointCloudResponse
from lab_vbnpm.srv import GetObjectOcclusions, GetObjectOcclusionsResponse

import perception.putils as putils
import utils.visual_utils as visual_utils
from utils.visual_utils import encode_seg_img_rgb, rgb_to_float
from perception.perception_system import PerceptionSystem


class PerceptionInterface():

    def __init__(self):
        self.load_ros_param()
        # construct inner representation
        self.system = PerceptionSystem(self.scene_resols, self.obj_resols)
        self.bridge = CvBridge()
        self.raw_points = []
        self.now = rospy.Time.now()

        ws_pose = self.system.ws_pose
        ws_size = self.system.ws_size
        # set the workspace parameters
        rospy.set_param('ws_pose', ws_pose.tolist())
        rospy.set_param(
            'ws_size',
            ws_size.tolist(),
        )  # real-valued workspace size

        # init ROS related ones
        rospy.Service(
            "get_object_tsdfs",
            GetObjectTSDFs,
            self.get_object_tsdfs,
        )
        rospy.Service(
            "get_object_models",
            GetObjectModels,
            self.get_object_voxels,
        )
        rospy.Service(
            "get_scene_occlusion",
            GetSceneOcclusion,
            self.get_scene_occlusion,
        )
        rospy.Service(
            "get_scene_point_cloud",
            GetScenePointCloud,
            self.get_scene_point_cloud,
        )
        rospy.Service(
            "get_object_occlusions",
            GetObjectOcclusions,
            self.get_object_occlusions,
        )

        self.rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        self.pcd_sub = message_filters.Subscriber(self.pcd_topic, PointCloud2)
        self.clk_sub = rospy.Subscriber('/clock', Clock, self.set_time)
        self.img_sub = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.pcd_sub],
            10,
            0.1,
        )
        self.img_sub.registerCallback(self.img_cb)

    def set_time(self, clk_msg):
        self.now = clk_msg.clock

    def load_ros_param(self):
        # load the ROS parameters (relative to the node's namespace)
        scene_resols = rospy.get_param('scene_resols', [0.01, 0.01, 0.01])
        obj_resols = rospy.get_param('obj_resols', [0.01, 0.01, 0.01])
        rgb_topic = rospy.get_param(
            'execution_interface/rgb_topic',
            '/camera/color/image_raw',
        )
        depth_topic = rospy.get_param(
            'execution_interface/depth_topic',
            '/camera/aligned_depth_to_color/image_raw',
        )
        pcd_topic = rospy.get_param(
            'perception/pcd_topic',
            '/point_cloud/points',
        )
        print(rgb_topic,depth_topic)
        scene_resols = np.array(scene_resols)
        obj_resols = np.array(obj_resols)

        self.scene_resols = scene_resols
        self.obj_resols = obj_resols
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.pcd_topic = pcd_topic

    def construct_voxel_msg(self, voxel, pose, resols, is_tsdf=False):
        if is_tsdf:
            msg = VoxelTSDF()
        else:
            msg = VoxelGridBool()
        qw, qx, qy, qz = tf.quaternion_from_matrix(pose)
        msg.pose.position.x = pose[0, 3]
        msg.pose.position.y = pose[1, 3]
        msg.pose.position.z = pose[2, 3]
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        msg.resols.x = resols[0]
        msg.resols.y = resols[1]
        msg.resols.z = resols[2]
        msg.size_x = voxel.shape[0]
        msg.size_y = voxel.shape[1]
        msg.size_z = voxel.shape[2]
        msg.data = voxel.flatten().tolist()
        return msg

    def get_scene_point_cloud(self, req):
        """
        get the current scene point cloud and publish
        """
        occluded = self.system.total_occluded
        if occluded is None:
            return GetScenePointCloudResponse(PointCloud2())  # blank msg

        resols = np.array(self.system.scene_occlusion.resols)
        pose = np.array(self.system.scene_occlusion.world_T_voxel)
        pose = tf.concatenate_matrices(
            np.linalg.inv(self.system.extrinsics), pose
        )
        ignore_occlusion_set = set(req.ignore_occlusion)
        scene_pcd = self.raw_points.copy()
        if False:
            max_id = max(self.system.objects.keys()) + 1
            for obj_id, obj in self.system.objects.items():
                ignore_occlusion = obj_id in ignore_occlusion_set
                if ignore_occlusion:
                    obj_pcd = obj.sample_optimistic_pcd()
                else:
                    obj_pcd = obj.sample_conservative_pcd()
                occupied, occluded = self.system.scene_occlusion.single_object_occlusion(
                    self.system.extrinsics,
                    self.system.intrinsics,
                    np.array(obj.world_T_voxel),
                    obj_pcd,
                )
                if ignore_occlusion:
                    voxels = occupied
                else:
                    voxels = occupied | occluded
                # print('occupied:',occupied.nonzero()[0])
                # print('occluded:',occluded.nonzero()[0])
                # print('voxels:',voxels.nonzero()[0])
                sample_pcd = self.system.scene_occlusion.sample_pcd(voxels)
                obj_occ_pcd = np.ones((len(sample_pcd), 4))
                obj_occ_pcd[:, :3] = sample_pcd
                obj_occ_pcd = np.matmul(obj_occ_pcd, pose.T)
                color = encode_seg_img_rgb(obj_id, offset=max_id)
                obj_occ_pcd[:, 3] = rgb_to_float(color)
                scene_pcd = np.concatenate((scene_pcd, obj_occ_pcd), axis=0)

        print(scene_pcd.shape)
        pc_array = np.zeros(
            len(scene_pcd),
            dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('rgb', np.float32),
            ]
        )
        pc_array['x'] = scene_pcd[:, 0]
        pc_array['y'] = scene_pcd[:, 1]
        pc_array['z'] = scene_pcd[:, 2]
        pc_array['rgb'] = scene_pcd[:, 3]

        msg = rnp.msgify(
            PointCloud2, pc_array, stamp=self.now, frame_id='camera_link'
        )
        return GetScenePointCloudResponse(msg)

    def get_scene_occlusion(self, req):
        """
        get the current scene occlusion and publish
        """
        occluded = self.system.total_occluded
        if occluded is None:
            return GetSceneOcclusionResponse(SceneOcclusion())  # blank msg
        resols = np.array(self.system.scene_occlusion.resols)
        pose = np.array(self.system.scene_occlusion.world_T_voxel)
        qw, qx, qy, qz = tf.quaternion_from_matrix(pose)
        voxel = self.construct_voxel_msg(occluded, pose, resols)
        print('visualizing scene occlusion...')
        v_voxel = visual_utils.visualize_voxel(
            self.system.scene_occlusion.voxel_x,
            self.system.scene_occlusion.voxel_y,
            self.system.scene_occlusion.voxel_z,
            occluded,
            [1, 0, 0],
        )
        o3d.visualization.draw_geometries([v_voxel])

        msg = SceneOcclusion()
        msg.voxel = voxel
        return GetSceneOcclusionResponse(msg)

    def get_object_occlusions(self, req):
        """
        get the object occlusions
        """
        obj_occlusions = []
        v_voxels = []
        for obj_id, obj in self.system.objects.items():
            obj_occlusion = ObjectOcclusion()
            obj_occlusion.obj_id = obj_id
            resols = np.array(self.system.scene_occlusion.resols)
            pose = np.array(self.system.scene_occlusion.world_T_voxel)
            # occluded = self.system.scene_occlusion.occlusion_from_pcd(
            occupied, occluded = self.system.scene_occlusion.single_object_occlusion(
                self.system.extrinsics,
                self.system.intrinsics,
                # self.system.img_shape,
                # {obj_id: np.array(obj.world_T_voxel)},
                # {obj_id: obj.sample_conservative_pcd()},
                np.array(obj.world_T_voxel),
                obj.sample_conservative_pcd(),
                # obj.sample_optimistic_pcd(),
            )
            voxel = self.construct_voxel_msg(occluded, pose, resols)
            obj_occlusion.voxel = voxel
            obj_occlusions.append(obj_occlusion)
            if req.visualize:
                print(
                    'visualizing object occlusion: ', obj_id,
                    self.system.tracker.name_dict[obj_id], occluded.any()
                )
                v_voxel = visual_utils.visualize_voxel(
                    self.system.scene_occlusion.voxel_x,
                    self.system.scene_occlusion.voxel_y,
                    self.system.scene_occlusion.voxel_z,
                    occluded,
                    # occupied,
                    [1, 0, 0],
                )
                v_voxels.append(v_voxel)
                pcd, color = obj.sample_conservative_pcd(color=True)
                # pcd, color = obj.sample_optimistic_pcd(color=True)
                color /= 255
                pcd = obj.world_T_voxel[:3, :3].dot(pcd.T).T
                pcd += obj.world_T_voxel[:3, 3]
                pcd = self.system.scene_occlusion.voxel_T_world[:3, :3].dot(
                    pcd.T
                ).T
                pcd += self.system.scene_occlusion.voxel_T_world[:3, 3]
                pcd /= resols
                v_pcd = visual_utils.visualize_pcd(pcd, color)
                v_voxels.append(v_pcd)

        if req.visualize:
            o3d.visualization.draw_geometries(v_voxels)
        msg = ObjectOcclusions()
        msg.obj_occlusions = obj_occlusions
        return GetObjectOcclusionsResponse(msg)

    def get_object_voxels(self, req):
        return self.get_object_models(is_tsdf=False, visualize=req.visualize)

    def get_object_tsdfs(self, req):
        return self.get_object_models(is_tsdf=True)

    def get_object_models(self, is_tsdf=False, visualize=False):
        """
        get the object models
        """
        obj_models = []
        for obj_id, obj in self.system.objects.items():
            if is_tsdf:
                obj_model = ObjectTSDF()
            else:
                obj_model = ObjectModel()
            obj_model.obj_id = obj_id
            obj_pose = np.array(obj.world_T_obj)
            qw, qx, qy, qz = tf.quaternion_from_matrix(obj_pose)
            obj_model.pose.position.x = obj_pose[0, 3]
            obj_model.pose.position.y = obj_pose[1, 3]
            obj_model.pose.position.z = obj_pose[2, 3]
            obj_model.pose.orientation.w = qw
            obj_model.pose.orientation.x = qx
            obj_model.pose.orientation.y = qy
            obj_model.pose.orientation.z = qz
            obj_T_voxel = np.array(obj.obj_T_voxel)
            if is_tsdf:
                obj_msg = self.construct_voxel_msg(
                    obj.tsdf,
                    obj_T_voxel,
                    obj.resols,
                    is_tsdf=True,
                )
                obj_model.tsdf = obj_msg
            else:
                con_voxel = obj.get_conservative_model()
                opt_voxel = obj.get_optimistic_model()
                con_msg = self.construct_voxel_msg(
                    con_voxel,
                    obj_T_voxel,
                    obj.resols,
                )
                opt_msg = self.construct_voxel_msg(
                    opt_voxel,
                    obj_T_voxel,
                    obj.resols,
                )
                obj_model.con_volume = con_msg
                obj_model.opt_volume = opt_msg
                if visualize:
                    print('visualizing object: ', obj_id)
                    obj.visualize_obj(con_voxel)
                    obj.visualize_obj(opt_voxel)

            obj_models.append(obj_model)

        if is_tsdf:
            msg = ObjectTSDFs()
            msg.obj_tsdfs = obj_models
            return GetObjectTSDFsResponse(msg)
        else:
            msg = ObjectModels()
            msg.obj_models = obj_models
            return GetObjectModelsResponse(msg)

    def img_cb(self, rgb_msg, depth_msg, pcd_msg):
        color_img = self.bridge.imgmsg_to_cv2(rgb_msg, 'passthrough')
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        raw_pcd = rnp.numpify(pcd_msg)
        self.raw_points = np.array(putils.cloud_array_to_point_list(raw_pcd))
        print("image callback")
        self.update_scene(color_img, depth_img)

    def update_scene(self, color_img, depth_img):
        """
        update the scene given perceived RGBD image and segmented image
        use the implemented function in the occlusion object for the inference of
        object models and scene model
        """
        print("update_scene")
        self.system.perceive(
            color_img, depth_img, visualize=rospy.get_param('~vis', False)
        )


def main():
    rospy.init_node("perception")
    interface = PerceptionInterface()
    rospy.spin()


if __name__ == "__main__":
    main()
