"""
provide utility functions for various conversions
"""

import copy
import numpy as np
import transformations as tf

import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import Transform, Vector3, Quaternion, Pose, PoseStamped


def joint_state_to_dict(joint_state: JointState):
    names = joint_state.name
    position = joint_state.position
    velocity = joint_state.velocity
    position = {names[i]: position[i] for i in range(len(names))}
    velocity = {names[i]: velocity[i] for i in range(len(names))}
    return position, velocity


def pose_to_list(pose_msg):
    translation = [
        pose_msg.position.x,
        pose_msg.position.y,
        pose_msg.position.z,
    ]
    quaternion = [
        pose_msg.orientation.w,
        pose_msg.orientation.x,
        pose_msg.orientation.y,
        pose_msg.orientation.z,
    ]
    return translation + quaternion


def list_to_matrix(pose_list):
    transform = tf.quaternion_matrix(pose_list[3:])
    transform[:3, 3] = pose_list[:3]
    return transform


def list_to_pose(pose_list):
    pose_msg = Pose()
    pose_msg.position.x = pose_list[0]
    pose_msg.position.y = pose_list[1]
    pose_msg.position.z = pose_list[2]
    pose_msg.orientation.w = pose_list[3]
    pose_msg.orientation.x = pose_list[4]
    pose_msg.orientation.y = pose_list[5]
    pose_msg.orientation.z = pose_list[6]
    return pose_msg


def pose_to_matrix(pose_msg):
    translation = [
        pose_msg.position.x,
        pose_msg.position.y,
        pose_msg.position.z,
    ]
    quaternion = [
        pose_msg.orientation.w,
        pose_msg.orientation.x,
        pose_msg.orientation.y,
        pose_msg.orientation.z,
    ]
    transform = tf.quaternion_matrix(quaternion)
    transform[:3, 3] = translation
    return transform


def matrix_to_pose(matrix):
    translation = tf.translation_from_matrix(matrix)
    quaternion = tf.quaternion_from_matrix(matrix)
    pose_msg = Pose()
    pose_msg.position.x = translation[0]
    pose_msg.position.y = translation[1]
    pose_msg.position.z = translation[2]
    pose_msg.orientation.w = quaternion[0]
    pose_msg.orientation.x = quaternion[1]
    pose_msg.orientation.y = quaternion[2]
    pose_msg.orientation.z = quaternion[3]
    return pose_msg


def matrix_to_pose_stamped(matrix, target_frame, stamp=None):
    translation = tf.translation_from_matrix(matrix)
    quaternion = tf.quaternion_from_matrix(matrix)
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = target_frame
    pose_msg.header.stamp = stamp if stamp else rospy.Time.now()
    pose_msg.position.x = translation[0]
    pose_msg.position.y = translation[1]
    pose_msg.position.z = translation[2]
    pose_msg.orientation.w = quaternion[0]
    pose_msg.orientation.x = quaternion[1]
    pose_msg.orientation.y = quaternion[2]
    pose_msg.orientation.z = quaternion[3]
    return pose_msg


def transform_to_matrix(transform):
    qx = transform.rotation.x
    qy = transform.rotation.y
    qz = transform.rotation.z
    qw = transform.rotation.w
    x = transform.translation.x
    y = transform.translation.y
    z = transform.translation.z
    matrix = tf.quaternion_matrix([qw, qx, qy, qz])
    matrix[:3, 3] = [x, y, z]
    return matrix


def matrix_to_transform(matrix):
    translation = list(tf.translation_from_matrix(matrix))
    quaternion = list(tf.quaternion_from_matrix(matrix))
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]
    x = translation[0]
    y = translation[1]
    z = translation[2]
    return Transform(
        translation=Vector3(x=x, y=y, z=z),
        rotation=Quaternion(x=qx, y=qy, z=qz, w=qw)
    )


def dict_to_matrix(transform):
    qx = transform["qx"]
    qy = transform["qy"]
    qz = transform["qz"]
    qw = transform["qw"]
    x = transform["x"]
    y = transform["y"]
    z = transform["z"]
    matrix = tf.quaternion_matrix([qw, qx, qy, qz])
    matrix[:3, 3] = [x, y, z]
    return matrix


def float_to_ros_duration(ft, decimals=9):
    seconds = int(ft)
    nseconds = int(round((ft - seconds), decimals) * 1e9)
    return rospy.Duration(seconds, nseconds)
