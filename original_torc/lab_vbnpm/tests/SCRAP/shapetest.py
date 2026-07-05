import sys
import time
import copy
import glob
import numpy as np
import trimesh as tm
import open3d as o3d
from urchin import URDF

import rospy
import rospkg

import utils.visual_utils as vutils

date = sys.argv[1]
# date = 'Wed_Mar__6_17-10-09_2024'
p_name = glob.glob(f'./pc_data/*{date}*.pcd')[0]
a_name = glob.glob(f'./pc_data/*{date}*.npy')[0]
a = np.load(a_name)
p = o3d.io.read_point_cloud(p_name)

# show original points
points = np.array(p.points)[a]
colors = np.array(p.colors)[a]
cloud = tm.points.PointCloud(points, colors)
cloud.show()

# show largest cluster points
clusters = tm.grouping.clusters(points, 0.01)
max_cluster_inds = max(clusters, key=len)
points = points[max_cluster_inds]
colors = colors[max_cluster_inds]
cloud = tm.points.PointCloud(points, colors)
cloud.show()

to_origin, extents = tm.bounds.oriented_bounds(points)
center = to_origin[:3, 3]
transform = np.linalg.inv(to_origin)
box = tm.primitives.Box(extents=extents, transform=transform)
tm.scene.Scene([cloud, box]).show()

rotate_axis = (np.dot(to_origin[:3, :3], [0, 0, 1])).round(0).astype(int)
print((np.dot(to_origin[:3, :3], [0, 0, 1])).round(0))
print(np.dot(to_origin[:3, :3], [1, 0, 0]))
rot = tm.transformations.rotation_matrix(np.pi, rotate_axis)
# rot = tm.transformations.rotation_matrix(np.pi, extents - (extents / 2))
flip = tm.transformations.reflection_matrix([0, 0, 0], [1, 0, 0])
flip2 = tm.transformations.reflection_matrix([0, 0, 0], [0, 1, 0])
flip3 = tm.transformations.reflection_matrix([0, 0, 0], [0, 0, 1])
move = tm.transformations.translation_matrix(
    [
        np.sign(np.dot(to_origin[:3, 0], extents)) * extents[0],
        # np.sign(np.dot(to_origin[:3, 1], extents)) * extents[1]/2,
        0,
        0,
    ]
    # [-extents[0], 0, 0]
)

print(to_origin, extents)
cloud.apply_transform(np.linalg.inv(transform))
print(cloud.centroid, np.linalg.norm(cloud.centroid))
cloud.apply_transform(rot)
cloud.apply_transform(move)
# cloud.apply_transform(flip)
# cloud.apply_transform(flip2)
# cloud.apply_transform(flip3)
cloud.apply_transform(transform)
tm.scene.Scene([cloud, box]).show()

# rotate  and add points
if False:
    axis = tm.points.major_axis(points)
    to_origin, extents = tm.bounds.oriented_bounds(cloud)
    print(to_origin[:3, 3], axis)
    offset = (
        -1 * axis + 0. * (to_origin[:3, 3] / np.linalg.norm(to_origin[:3, 3]))
    )
    offset = extents * offset / np.linalg.norm(offset)
    offset[2] = 0
    center = to_origin[:3, 3]  #cloud.centroid
    # center = offset
    rot = tm.transformations.rotation_matrix(np.pi, axis, center)
    flip = tm.transformations.reflection_matrix(center, axis)
    flip2 = tm.transformations.reflection_matrix(center, [0, 0, 1])
    move = tm.transformations.translation_matrix(offset)
    cloud.apply_transform(rot)
    cloud.apply_transform(flip)
    cloud.apply_transform(flip2)
    cloud.apply_transform(move)
new_points = np.concatenate([points, cloud.vertices])
new_colors = np.concatenate([colors, cloud.colors[:, :3] / 255.0])
new_cloud = tm.points.PointCloud(new_points, new_colors)
new_cloud.show()

if False:
    N = 100
    vox_size = 0.2

    pts = np.zeros((N, 3))
    pts[:, 0] = np.linspace(0, 1, N)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    normals = np.zeros((N, 3))
    normals[:, 0] = (np.array(list(range(N))) / N).round().astype(bool)
    # normals[:,0] = np.random.randint(0, 2, N).astype(bool)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd_ds, trace, trace2 = pcd.voxel_down_sample_and_trace(
        vox_size, pcd.get_min_bound(), pcd.get_max_bound(), False
    )
    pcd_ds2 = pcd.voxel_down_sample(vox_size)
    # Offset the new points a little in y so we can see them in the visualization
    # pcd_ds.points = o3d.utility.Vector3dVector(np.asarray(pcd_ds.points) + [0.0, 0.1, 0.0])
    print("N = %d" % (len(pcd.points)))
    print("M = %d" % (len(pcd_ds.points)))
    print("Trace Shape = ", trace.shape)
    print(trace)
    og_class = np.zeros(len(pcd_ds.points))
    for i, t in enumerate(trace2):
        print(f"Trace2{i} Shape = ", np.array(t).shape)
        print(t)
        print(np.array(pcd.normals)[np.array(t), 0])
        og_class[i] = np.array(pcd.normals)[np.array(t), 0].any()

    o3d.visualization.draw_geometries([pcd, pcd_ds])
    print(np.array(pcd.normals)[:, 0])
    print(np.array(pcd_ds.normals)[:, 0])
    print(np.array(pcd_ds.normals)[:, 0].round())
    print(og_class)
    print(np.array(pcd_ds2.normals)[:, 0])
    print(np.array(pcd_ds2.normals)[:, 0].round())
    print(np.array(pcd_ds2.normals)[:, 0].round().astype(bool))

if False:
    rospack = rospkg.RosPack()
    urdf_path = rospack.get_path('lab_vbnpm')
    urdf_path += '/robots/robotiq_arg85_description.URDF'
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
    mesh0 = trimesh.util.concatenate(scene.geometry.values())

    def in_collision(gripper_transform, voxels, visualize=False):
        mesh = copy.deepcopy(mesh0)
        mesh.apply_transform(gripper_transform)
        points = trimesh.sample.sample_surface(mesh, 10000)[0] * 100
        # pcl = o3d.geometry.PointCloud()
        # pcl.points = o3d.utility.Vector3dVector(points * 100)
        if visualize:
            pcd = trimesh.points.PointCloud(points)
            trimesh.scene.Scene([voxels.as_boxes(), pcd]).show()
            # voxel_x, voxel_y, voxel_z = np.indices(occluded.shape).astype(float)
            # vvoxel = vutils.visualize_voxel(
            #     voxel_x, voxel_y, voxel_z, occluded, [1, 0, 0]
            # )
            # o3d.visualization.draw_geometries([vvoxel, pcl])
        return voxels.is_filled(points).any()

    shape_x, shape_y, shape_z = 10, 20, 30
    occluded = np.random.randint(0, 2, (shape_x, shape_y, shape_z), bool)
    occluded = trimesh.voxel.VoxelGrid(occluded)

    print(
        in_collision(
            np.array(
                [
                    [1., 0., 0., 0],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.1],
                    [0., 0., 0., 1.],
                ]
            ),
            occluded,
            visualize=True
        )
    )
    print(
        in_collision(
            np.array(
                [
                    [1., 0., 0., 0],
                    [0., 1., 0., 0.],
                    [0., 0., 1., -0.1],
                    [0., 0., 0., 1.],
                ]
            ),
            occluded,
            visualize=True
        )
    )
