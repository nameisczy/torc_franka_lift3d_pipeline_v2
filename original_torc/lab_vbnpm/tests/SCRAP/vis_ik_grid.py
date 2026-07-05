import numpy as np
import transformations as tf

import rospy
from rospkg import RosPack
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, MarkerArray

from tracikpy import TracIKSolver
from utils.conversions import matrix_to_pose


def make_mark(p, i, num_suc, num_rot):
    marker = Marker()
    marker.id = i
    marker.action = Marker.ADD
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.type = Marker.ARROW
    marker.scale.x = 0.02
    marker.scale.y = 0.01
    marker.scale.z = 0.01
    if p.shape == (4, 4):
        marker.pose = matrix_to_pose(p)
    elif p.shape == (2, 3):
        marker.points = p
    if num_suc:
        marker.color.a = 0.25
    else:
        marker.color.a = 0.25
    marker.color.g = num_suc / num_rot
    marker.color.r = 1.0 - num_suc / num_rot
    return marker


if __name__ == '__main__':
    rospy.init_node('vis_ik_grid')
    rp = RosPack()
    urdf = rp.get_path('lab_vbnpm')
    urdf += '/robots/motoman/curobo/motoman.urdf'
    ik_solver = TracIKSolver(urdf, "base_link", "motoman_right_ee")

    pub = rospy.Publisher(
        '/visualization_marker', MarkerArray, latch=True, queue_size=10
    )

    # Example of usage
    spacing = 0.1
    num_rot = 4

    markers = MarkerArray()
    # markers.markers.append(make_mark(np.eye(4), 1, True))
    # pub.publish(markers)
    # rospy.spin()

    # Generate the grid of translations (x, y)
    num = 0
    pose = np.eye(4)
    for i in range(10):
        for j in range(-10, 11):
            for k in range(14):
                x = i * spacing + 0.3
                y = j * spacing
                z = k * spacing + 0.5
                t = np.array([x, y, z])
                pose[:3, 3] = t
                for l in range(num_rot + 1):
                    theta = np.pi * (l / num_rot - 0.5)  # From -pi/2 to pi/2
                    R = tf.rotation_matrix(theta, [0, 0, 1])
                    pose[:3, :3] = R[:3, :3]

                    # markers.markers[-1] =make_mark(pose, num, 1)
                    # pub.publish(markers)

                    num_to_rot = num_rot
                    clear = np.allclose(-np.pi / 2, theta)
                    clear = clear or np.allclose(np.pi / 2, theta)
                    if clear:
                        num_to_rot = 1
                    for m in range(num_to_rot):
                        phi = 2 * np.pi * m / num_rot  # From 0 to 2*pi
                        R2 = tf.rotation_matrix(phi, [0, 1, 0]) @ R
                        pose[:3, :3] = R2[:3, :3]

                        # markers.markers[-1] =make_mark(pose, num, 1)
                        # pub.publish(markers)
                        num_suc = 0
                        for n in range(num_rot + 1):
                            psi = 2 * np.pi * n / num_rot
                            R3 = R2 @ tf.rotation_matrix(psi, [1, 0, 0])
                            pose[:3, :3] = R3[:3, :3]

                            # markers.markers[-1] =make_mark(pose, num, 1)
                            # pub.publish(markers)
                            success = ik_solver.ik(pose) is not None
                            num_suc += success
                        markers.markers.append(
                            make_mark(pose, num, num_suc, num_rot)
                        )
                        num += 1
            print(num)
            pub.publish(markers)

    rospy.spin()
