from pprint import pp

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from lab_vbnpm.msg import ObjectPoses

rospy.init_node('print_object_poses')


def pose_callback(msg):
    names = msg.name
    oids = msg.id
    poses = msg.pose
    min_name = ''
    min_pos = None
    min_z = 1000
    for name, pose in zip(names, poses):
        if pose.position.z < min_z:
            min_name = name
            min_pos = pose.position
            min_z = pose.position.z
    print(f'{min_name} at {min_pos}')



sub = rospy.Subscriber('/ground_truth/object_poses', ObjectPoses, pose_callback)

rospy.spin()
