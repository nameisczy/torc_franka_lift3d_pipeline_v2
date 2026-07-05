"""
This code provides functions that deal with fake perceptions in simulations.
"""
import copy
import rospy
import numpy as np
import open3d as o3d
import transformations as tf

np.float = float
import ros_numpy as rnp

import tf2_ros
import message_filters
import tf2_geometry_msgs
from sensor_msgs.msg import PointCloud2
from lab_vbnpm.msg import ObjectPoses
from sensor_msgs import point_cloud2 as pc2
from geometry_msgs.msg import Pose, PoseStamped
from jsk_recognition_msgs.msg import LabelArray
from visualization_msgs.msg import Marker, MarkerArray

from utils.visual_utils import decode_seg_img_rgb


class InertialTracker():
    """
    Track position as the center of mass of segmented points of the pointcloud.
    Track orientation as intertial transform of "" "".
    """

    def __init__(self, subscribe=None, real_segmentation=False):
        # obj_id -> relative transform (NOTE: this is not delta transform)
        self.mjk_T_obj_dict = {}
        # this records the latest pose of objects
        self.world_T_mjk_dict = {}
        self.name_dict = {}
        print("Subscribe?", subscribe)
        if subscribe is not None:
            self.tf_buffer = tf2_ros.Buffer(rospy.Duration(60))
            self.tf_listen = tf2_ros.TransformListener(self.tf_buffer)
            self.obj_pose_pub = rospy.Publisher(
                '/object_tracker/object_poses', ObjectPoses, queue_size=5
            )
            self.marker_pub = rospy.Publisher(
                '/object_tracker/debug_bboxes', MarkerArray, queue_size=5
            )
            # rospy.Subscriber(subscribe, PointCloud2, self.publish_object_poses)
            cloud_sub = message_filters.Subscriber(subscribe, PointCloud2)
            label_sub = message_filters.Subscriber(
                '/docker/detic_segmentor/detected_classes', LabelArray
            )
            ts = message_filters.ApproximateTimeSynchronizer(
                [cloud_sub, label_sub], 5, 0.1
            )
            print("got to here")
            ts.registerCallback(self.publish_object_poses)
            print("got to here2")

    def track_poses(self, rgb_img, depth_img, obj_dict):
        obj_state = rospy.wait_for_message(
            '/object_tracker/object_poses',
            ObjectPoses,
        )
        zip_state = zip(obj_state.id, obj_state.name, obj_state.pose)
        for oid, name, pose in zip_state:
            self.name_dict[oid] = name
            pos = [
                pose.position.x,
                pose.position.y,
                pose.position.z,
            ]
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

    def publish_object_poses(self, pcd_msg, label_msg):
        # format point cloud and extract ids
        print("Recieved cloud!")
        cloud_array = rnp.point_cloud2.pointcloud2_to_array(pcd_msg)
        print("Size:", cloud_array.size)
        rgbs = rnp.point_cloud2.split_rgb_field(cloud_array)
        seg_img_rgb = np.zeros((*rgbs.shape, 3))
        seg_img_rgb[:, :, 0] = rgbs['r']
        seg_img_rgb[:, :, 1] = rgbs['g']
        seg_img_rgb[:, :, 2] = rgbs['b']
        oids = decode_seg_img_rgb(seg_img_rgb)
        xyzs = rnp.point_cloud2.get_xyz_points(cloud_array, remove_nans=False)
        ids = list(filter(lambda x: x >= 0, set(oids.flatten())))
        print("Ids:", ids)
        print("Ids:", set(oids.flatten()))

        # map ids to semantic labels
        id2name = {}
        for oid, label in enumerate(label_msg.labels):
            print('Obj: ', oid, label.name, label.id)
            assert (oid in ids)
            id2name[oid] = label.name

        try:
            camera2world = self.tf_buffer.lookup_transform(
                'world', 'camera_link', rospy.Time(), rospy.Duration(1.0)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            print("Error: Couldn't get transform to camera pose!")
            return

        # create msgs and publish
        markers = MarkerArray()
        object_poses = ObjectPoses()
        object_poses.id = ids
        object_poses.name = [id2name.get(oid, 'unknown') for oid in ids]
        for oid in ids:
            points = xyzs[oids == oid]
            print(xyzs.shape, points.shape)
            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(points)
            # o3d.visualization.draw_geometries([pcl])
            pcl, _ = pcl.remove_statistical_outlier(20, 1)
            # o3d.visualization.draw_geometries([pcl])
            box = pcl.get_oriented_bounding_box()
            # o3d.visualization.draw_geometries(box)
            extents = box.extent
            transform = np.eye(4)
            transform[:3, :3] = box.R
            transform[:3, 3] = box.get_center()
            print(transform)

            pos = transform[:3, 3]
            quat = tf.quaternion_from_matrix(transform)
            print(extents, pos, quat)

            pose = Pose()
            pose.position.x = pos[0]
            pose.position.y = pos[1]
            pose.position.z = pos[2]
            pose.orientation.w = quat[0]
            pose.orientation.x = quat[1]
            pose.orientation.y = quat[2]
            pose.orientation.z = quat[3]

            pose_s = PoseStamped()
            pose_s.header.frame_id = 'camera_link'
            pose_s.pose = pose
            pose_w = tf2_geometry_msgs.do_transform_pose(pose_s, camera2world)
            object_poses.pose.append(pose_w.pose)

            marker = Marker()
            marker.header.frame_id = 'camera_link'
            marker.header.stamp = rospy.Time.now()
            marker.ns = 'bboxes'
            marker.id = oid
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = pose
            marker.scale.x = extents[0]
            marker.scale.y = extents[1]
            marker.scale.z = extents[2]
            marker.color.a = 0.7
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.lifetime = rospy.Duration(0)
            marker.frame_locked = False
            markers.markers.append(marker)

        self.obj_pose_pub.publish(object_poses)
        self.marker_pub.publish(markers)


if __name__ == '__main__':
    rospy.init_node('intertial_object_tracker')
    rospy.sleep(1.0)
    intertial_tracker = InertialTracker('/point_cloud/points')
    print("Running...")
    rospy.spin()
