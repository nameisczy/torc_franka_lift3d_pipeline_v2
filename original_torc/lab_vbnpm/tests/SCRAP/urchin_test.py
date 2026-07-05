import sys
import numpy as np
import open3d as o3d
import trimesh
from urchin import URDF

urdf_path = sys.argv[1]
gripper = URDF.load(urdf_path)
links = [
    'robotiq_85_base_link',
    'left_outer_knuckle',
    'left_outer_finger',
    'left_inner_knuckle',
    'left_inner_finger',
    'right_inner_knuckle',
    'right_inner_finger',
    'right_outer_knuckle',
    'right_outer_finger',
]
# sort links
g_links = list(gripper.link_fk(use_names=True, links=links).keys())
geoms = gripper.collision_trimesh_fk(links=g_links)
scene = trimesh.scene.Scene()
for link, geometry_transform in zip(g_links, geoms.items()):
    geometry, transform = geometry_transform
    geometry.apply_transform(transform)
    scene.add_geometry(
        geometry,
        node_name=link,
        geom_name=link,
        # transform=transform,
    )
ee_mesh = trimesh.util.concatenate(scene.geometry.values())
# trimesh.scene.Scene([ee_mesh,ee_mesh.copy().apply_transform(ee_scale * np.eye(4))]).show()
# ee_mesh.apply_transform(ee_scale * np.eye(4))
ee_mesh.show()
ee_mesh.apply_transform(
    [
        [0., -1, 0., 0.],
        [1., 0., 0., 0.],
        [0., 0., 1., -.135],
        [0., 0., 0., 1.],
    ]
)
ee_mesh.show()
