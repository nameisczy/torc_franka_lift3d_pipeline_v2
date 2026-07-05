"""
previous node publishes tf to color_optical_frame. Here we change that
to publish to camera_link
"""

import rospy

import tf2_ros
import tf2_geometry_msgs


def get_transform(source_frame, target_frame, tf_buffer):
    in2out = tf_buffer.lookup_transform(
        source_frame,
        target_frame,
        rospy.Time(),
        rospy.Duration(1.0),
    )
    print('in2out: ')
    print(in2out)
    return in2out
if __name__ == "__main__":
    rospy.init_node("get_transform")
    rospy.sleep(1.0)
    tf_buffer = tf2_ros.Buffer(rospy.Duration(60))
    tf_listen = tf2_ros.TransformListener(tf_buffer)
    get_transform('torso_link_b1', 'd455_link', tf_buffer)
