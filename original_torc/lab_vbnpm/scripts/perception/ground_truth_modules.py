"""
This code provides functions that deal with fake perceptions in simulations.
"""
import rospy
import cv_bridge
import numpy as np
import transformations as tf
from sensor_msgs.msg import Image
from lab_vbnpm.msg import ObjectPoses

from utils.visual_utils import decode_seg_img_rgb
from perception.inertial_tracker import InertialTracker


class GroundTruthSegmentation():

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()

    def segment_img(self, rgb_img, depth_img):
        seg_img_msg = rospy.wait_for_message('ground_truth/seg_image', Image)
        seg_img_raw = self.bridge.imgmsg_to_cv2(seg_img_msg, 'rgb8')
        # the seg_img gives a picture of geom_ids
        seg_img = decode_seg_img_rgb(seg_img_raw)
        return seg_img


class GroundTruthTracker():
    """
    record the relative transform of the mujoco pose and the object_model pose
    use this relative transform to then obtain the correct pose at each time
    Assuming the tracker obtains a tracked pose world_T_mjk. This later is used
    to update the object belief pose
    """

    def __init__(self, real_segmentation=False):
        # obj_id -> relative transform (NOTE: this is not delta transform)
        self.mjk_T_obj_dict = {}
        # this records the latest pose of objects
        self.world_T_mjk_dict = {}
        self.name_dict = {}
        self.real_segmentation = real_segmentation
        if self.real_segmentation:
            intertial_tracker = InertialTracker('/point_cloud/points')

    def track_poses(self, rgb_img, depth_img, obj_dict):
        obj_state = rospy.wait_for_message(
            '/ground_truth/object_poses',
            ObjectPoses,
        )
        if self.real_segmentation:
            obj_state_e = rospy.wait_for_message(
                '/object_tracker/object_poses',
                ObjectPoses,
            )
            zip_state = zip(obj_state_e.id, obj_state_e.name, obj_state_e.pose)
            real_positions = [
                (pose.position.x, pose.position.y, pose.position.z)
                for pose in obj_state.pose
            ]
            print(
                "Postions?", real_positions, [
                    (pose.position.x, pose.position.y, pose.position.z)
                    for pose in obj_state_e.pose
                ]
            )
        else:
            zip_state = zip(obj_state.id, obj_state.name, obj_state.pose)

        for oid, name, pose in zip_state:
            pos = [
                pose.position.x,
                pose.position.y,
                pose.position.z,
            ]

            if self.real_segmentation:
                dists = np.linalg.norm(np.subtract(pos, real_positions), axis=1)
                ind_min = dists.argmin()
                self.name_dict[oid] = obj_state.name[ind_min]
                real_pose = obj_state.pose[ind_min]
                print('MIN', oid, ind_min, self.name_dict[oid])
                # print(pos, real_positions, np.subtract(pos, real_positions))
                pos = [
                    real_pose.position.x,
                    real_pose.position.y,
                    real_pose.position.z,
                ]
                ori = [
                    real_pose.orientation.w,
                    real_pose.orientation.x,
                    real_pose.orientation.y,
                    real_pose.orientation.z,
                ]
            else:
                self.name_dict[oid] = name
                ori = [
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                ]

            mjk_pose = tf.quaternion_matrix(ori)
            mjk_pose[:3, 3] = np.array(pos)
            self.world_T_mjk_dict[oid] = mjk_pose

    def update_poses(self, obj_dict):
        for oid, world_T_mjk in self.world_T_mjk_dict.items():
            if oid in self.mjk_T_obj_dict:
                # update the pose based on the recorded value
                obj_pose = world_T_mjk.dot(self.mjk_T_obj_dict[oid])
                obj_dict[oid].update_pose(obj_pose)
            else:
                if oid not in obj_dict:
                    continue  # object belief model hasn't been created
                mjk_T_obj = np.linalg.inv(world_T_mjk).dot(
                    obj_dict[oid].world_T_obj
                )
                self.mjk_T_obj_dict[oid] = mjk_T_obj

    def add_new_obj(self, obj_id, obj):
        assert obj_id in self.world_T_mjk_dict  # the new obj should be sensed
        world_T_mjk = self.world_T_mjk_dict[obj_id]
        mjk_T_obj = np.linalg.inv(world_T_mjk).dot(obj.world_T_obj)
        self.mjk_T_obj_dict[id] = mjk_T_obj
