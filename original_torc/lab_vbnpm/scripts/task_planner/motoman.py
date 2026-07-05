import numpy as np
from urchin import URDF

import rospy
from rospkg import RosPack

from curobo.types.robot import RobotConfig
from curobo.types.file_path import ContentPath
from curobo.cuda_robot_model.util import load_robot_yaml

# from motion_planner.moveit_planner import MoveitPlanner
# from motion_planner.bio_ik_planner import BioIkPlanner
from motion_planner.motion_planner import MotionPlanner as DummyPlanner
from motion_planner.curobo_planner import CuroboPlanner
from perception.perception_fast import PerceptionInterface


class MotomanSDA10F:

    def __init__(self, is_sim, ground_truth=False):

        self.set_sim(is_sim)
        self.ground_truth = ground_truth
        self.gripper_link = "motoman_right_ee"
        self.gripper_group = "arm_right"
        self.camera_link = "camera_arm_link"
        self.camera_axis = (0, 1, 0)
        # self.camera_link = 'motoman_right_ee'
        # self.camera_axis = (0, 0, 1)

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
        self.urdf = rp.get_path("lab_vbnpm")
        self.urdf += "/robots/motoman/curobo/motoman.urdf"
        self.curobo_root = rp.get_path("lab_vbnpm")
        self.curobo_root += "/robots/motoman/curobo/"
        content_path = ContentPath(
            robot_config_absolute_path=self.curobo_root + "motoman.yml",
            robot_urdf_absolute_path=self.curobo_root + "motoman.urdf",
            robot_usd_absolute_path=self.curobo_root + "motoman.usd",
            robot_asset_absolute_path=self.curobo_root,
        )
        robot_dict = load_robot_yaml(content_path)
        self.curobo_config = robot_dict["robot_cfg"]

        # * load the point cloud representation of the robot. In the form of
        # link_name -> pcd (at identity pose, so that given link pose, we can transform the pcd)
        robot = URDF.load(self.urdf)
        """
        to check collision of robot with voxel, it might be more efficient to check pcd vs. voxel
        """
        pcd_link_dict = {}
        for link in robot.links:
            # print('link name: ', link.name)
            collisions = link.collisions
            if len(collisions) == 0:
                continue
            link_pcd = []  # one link may have several collision objects
            for collision in collisions:
                origin = (
                    collision.origin
                )  # this is the relative transform to get the pose of the geometry (mesh)
                # pose of the trimesh:
                # pose of link * origin * scale * trimesh_obj
                geometry = collision.geometry.geometry
                # print('geometry: ', geometry)
                # print('tag: ', geometry._TAG)
                # geometry.scale: give us the scale for the mesh

                meshes = geometry.meshes
                for mesh in meshes:
                    # print('mesh vertices: ')
                    # print(mesh.vertices)
                    # mesh.sample()
                    pcd = mesh.sample(len(mesh.vertices) * 5)
                    if collision.geometry.mesh is not None:
                        if collision.geometry.mesh.scale is not None:
                            pcd = pcd * collision.geometry.mesh.scale
                    pcd = origin[:3, :3].dot(pcd.T).T + origin[:3, 3]
                    link_pcd.append(pcd)
            link_pcd = np.concatenate(link_pcd, axis=0)
            # print('link_pcd shape: ', link_pcd.shape)
            pcd_link_dict[link.name] = link_pcd
        self.pcd_link_dict = pcd_link_dict
        self.urchin_robot = robot

    def set_sim(self, is_sim):
        self.is_sim = is_sim
        if self.is_sim:
            self.camera = ["camera0", "camera1"]
        else:
            self.camera = ["zedm"]
            # self.camera = ["d455"]

    def init_motion_planner(self, planner="curobo", warmup=True):

        if planner == "curobo":
            self.ignore_collision_ee_links = [
                # "arm_right_link_7_t",
                "robotiq_arg2f_base_link",
                "left_outer_finger",
                "right_outer_finger",
                "right_inner_finger",
                "left_inner_finger",
            ]
            planner = CuroboPlanner(
                self.urdf,
                ["motoman_left_ee", "motoman_right_ee"],  # , "camera_arm_link"],
                self.curobo_config,
                self.ignore_collision_ee_links,
                is_sim=self.is_sim,
                warmup=warmup,
            )
        else:
            planner = DummyPlanner(
                self.urdf,
                ["motoman_left_ee", "motoman_right_ee", "camera_arm_link"],
            )
        return planner

    def init_perception_interface(self):
        if self.ground_truth:
            mode = "gt"
        else:
            if self.is_sim:
                mode = "cam"
            else:
                mode = "fs"
        if self.is_sim and not rospy.has_param("/workspace/pose"):
            # init simulated perception interface
            perception = PerceptionInterface(
                self.camera,
                pose_x=0.9 - 0.29,
                pose_y=0 - 0.5,
                pose_z=0.90,
                size_x=0.58,
                size_y=1.0,
                size_z=0.5,
                resolution=0.0025,
                mode=mode,
                urdf_file=self.urdf,
            )
        else:
            # init real perception interface
            pose_x, pose_y, pose_z = rospy.get_param(
                "/workspace/pose", [0.55, 0.63, 1.05]
            )
            size_x, size_y, size_z = rospy.get_param(
                "/workspace/size", [0.4, 1.26, 0.52]
            )
            padding = 0.01
            perception = PerceptionInterface(
                self.camera,
                pose_x - padding,
                pose_y - size_y + padding,
                pose_z + padding / 10,
                size_x,
                size_y - 2 * padding,
                size_z - 2 * padding,
                resolution=0.002,
                # resolution=0.012,  # 0.01
                mode=mode,
                urdf_file=self.urdf,
            )
        return perception

    def get_pcd_at_joints(self, joints: dict, links: list):
        """
        given the joint dictionary (name -> value), return the concatenation of point cloud
        of selected links (names) at the given joint values.
        """
        total_pcd = []
        link_pose_dict = self.urchin_robot.link_fk(
            cfg=joints, links=links, use_names=True
        )
        for link in links:
            pcd = self.pcd_link_dict[link]
            # transform the pcd
            pose = link_pose_dict[link]
            pcd = pose[:3, :3].dot(pcd.T).T + pose[:3, 3]
            total_pcd.append(pcd)
        total_pcd = np.concatenate(total_pcd, axis=0)
        return total_pcd
