import rospy

import tf2_ros
import tf2_geometry_msgs
import transformations as tf

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

    frame = {
        "qw": 0.7120613112859896,
        "qx": -0.7020610309718125,
        "qy": -0.0009169361072556281,
        "qz": 0.008840644130837929,
        "x": -0.0375896555091571,
        "y": -0.011982238262219176,
        "z": 0.10619504056964525
      }


    torso_to_d455_color = tf.quaternion_matrix([frame["qw"],
                                                frame["qx"],
                                                frame["qy"],
                                                frame["qz"]])
    torso_to_d455_color[0,3] = frame['x']
    torso_to_d455_color[1,3] = frame['y']
    torso_to_d455_color[2,3] = frame['z']

    color_T_link_msg = get_transform('d435_color_optical_frame', 'd435_link', tf_buffer)
    color_T_link = tf.quaternion_matrix([color_T_link_msg.transform.rotation.w,
                                         color_T_link_msg.transform.rotation.x,
                                         color_T_link_msg.transform.rotation.y,
                                         color_T_link_msg.transform.rotation.z])
    color_T_link[0,3] = color_T_link_msg.transform.translation.x
    color_T_link[1,3] = color_T_link_msg.transform.translation.y
    color_T_link[2,3] = color_T_link_msg.transform.translation.z

    torso_to_link = torso_to_d455_color.dot(color_T_link)

    qw,qx,qy,qz = tf.quaternion_from_matrix(torso_to_link)
    print('qx: ', qx)
    print('qy: ', qy)
    print('qz: ', qz)
    print('qw: ', qw)

    print('x: ', torso_to_link[0,3])
    print('y: ', torso_to_link[1,3])
    print('z: ', torso_to_link[2,3])
