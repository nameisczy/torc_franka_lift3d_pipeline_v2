import sys
import os
import rospy
from rospkg import RosPack


class MotomanSDA10F:
    def __init__(self):
        self.ignore_collision_ee_links = [
            "motoman_right_ee",
            "left_outer_knuckle",
            "left_outer_finger",
            "left_inner_finger",
            "left_inner_finger_pad",
            "left_inner_knuckle",
            "right_outer_knuckle",
            "right_outer_finger",
            "right_inner_finger",
            "right_inner_finger_pad",
            "right_inner_knuckle",
            "robotiq_arg2f_extra_link",
            "robotiq_arg2f_base_link",
            "arm_right_link_7_t",
            "arm_right_link_6_b",
        ]
        rp = RosPack()
        self.robot_urdf = rp.get_path(
            'motoman_sda10f_moveit_config'
        ) + '/config/gazebo_motoman_sda10f.urdf'
        package_dirs = []
        for package in ['motoman_sda10f_support', 'robotiq_2f_85_gripper_visualization']:
            path = os.path.join(rp.get_path(package), '..')
            path = os.path.abspath(path)
            package_dirs.append(path)
            # package_dirs.append(rp.get_path(package))
        self.package_dirs = package_dirs