import time
import trimesh
import numpy as np
# from urchin import URDF
# from urdfpy import URDF
from yourdfpy import URDF
import transformations as tf

t0 = time.time()
moto = URDF.load("motoman.urdf")
moto.show()
links = [
    'torso_base_link',
    'torso_link_b1',
    'arm_left_link_1_s',
    'arm_left_link_2_l',
    'arm_left_link_3_e',
    'arm_left_link_4_u',
    'arm_left_link_5_r',
    'arm_left_link_6_b',
    'arm_left_link_7_t',
    'arm_right_link_1_s',
    'arm_right_link_2_l',
    'arm_right_link_3_e',
    'arm_right_link_4_u',
    'arm_right_link_5_r',
    'arm_right_link_6_b',
    'arm_right_link_7_t',
    'robotiq_arg2f_base_link',
    'left_outer_finger',
    'right_outer_finger',
    'right_inner_finger',
    'left_inner_finger',
    'left_outer_knuckle',
    'left_inner_finger_pad',
    'left_inner_knuckle',
    'right_outer_knuckle',
    'right_inner_finger_pad',
    'right_inner_knuckle',
    'robotiq_arg2f_extra_link',
]
joints = {
    "torso_joint_b1": 0,
    "arm_left_joint_1_s": 1.75,
    "arm_left_joint_2_l": 0.8,
    "arm_left_joint_3_e": 0,
    "arm_left_joint_4_u": -0.66,
    "arm_left_joint_5_r": 0,
    "arm_left_joint_6_b": 0,
    "arm_left_joint_7_t": 0,
    "arm_right_joint_1_s": 0.2,
    "arm_right_joint_2_l": -0.7,
    "arm_right_joint_3_e": 0.0,
    "arm_right_joint_4_u": -1.7,
    "arm_right_joint_5_r": 0,
    "arm_right_joint_6_b": -1.3,
    "arm_right_joint_7_t": 0.0,
}

geoms = moto.visual_trimesh_fk(joints, links)
scene = trimesh.scene.Scene()
for link, geometry_transform in zip(links, geoms.items()):
    geometry, transform = geometry_transform
    print(link, )
    geometry.show()
    scene.add_geometry(geometry, node_name=link, transform=transform)

mesh = scene.to_mesh()
mesh.show()
