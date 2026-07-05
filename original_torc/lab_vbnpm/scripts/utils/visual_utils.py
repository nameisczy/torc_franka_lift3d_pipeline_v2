import struct
import numpy as np
import open3d as o3d
import trimesh as tm
import seaborn as sns
import colorcet as cc
from psutil import Process


def vis_normals(points, normals, scale=0.1):
    pts = np.array(points)
    norms = np.array(normals)
    lines = []
    for i, point in enumerate(pts):
        line_start = point
        line_end = point + scale * norms[i]
        lines.append([line_start, line_end])
    lines_mesh = tm.load_path(np.array(lines))
    return lines_mesh


def from_color_map(num, size):
    if np.max(np.abs(num)) <= size:
        size = np.max(np.abs(num)) + 1
    return np.array(sns.color_palette(cc.glasbey_light, n_colors=size))[num]


def int16_to_rgb(num_f):
    return struct.unpack('>BBB', struct.pack('>xh', num_f))


def rgb_to_int16(rgba):
    return struct.unpack('>xh', struct.pack('>BBB', *rgba))[0]


def rgba_to_float(rgba):
    return struct.unpack('>f', struct.pack('>BBBB', *rgba))[0]


def rgb_to_float(rgb):
    return struct.unpack('>f', struct.pack('>xBBB', *rgb))[0]


def float_to_int16(flt):
    return struct.unpack('>xhx', struct.pack('>f', flt))[0]


def np_rgba_to_int16(a):
    arr = np.array(a, dtype='uint8')
    shape = arr.shape[:-1]
    return np.frombuffer(arr.astype('>B').tobytes(), dtype='>h').reshape(shape)


def np_int16_to_rgba(a):
    arr = np.array(a, dtype='int16')
    shape = (*arr.shape, 2)
    return np.frombuffer(arr.astype('>h').tobytes(), dtype='>B').reshape(shape)


def encode_seg_img_rgb(img, offset=2):
    dim = len(np.array(img).shape)
    id_encoded = np_int16_to_rgba(img)
    num = np.max(img) - np.min(img) + offset
    vis_value = from_color_map(img, num)[..., 0:1] * 256
    id_encoded = id_encoded.astype('uint8')
    vis_value = vis_value.astype('uint8')
    return np.concatenate([id_encoded, vis_value], axis=dim)


def decode_seg_img_rgb(img):
    return np_rgba_to_int16(img[..., :2])


def visualize_pcd(pcd, color):
    pcd_pcd = o3d.geometry.PointCloud()
    pcd_pcd.points = o3d.utility.Vector3dVector(pcd)
    colors = np.zeros(pcd.shape)
    color = np.array(color)
    if color.ndim == 1:
        colors[:, 0] = color[0]
        colors[:, 1] = color[1]
        colors[:, 2] = color[2]
    else:
        colors[:, 0] = color[:, 0]
        colors[:, 1] = color[:, 1]
        colors[:, 2] = color[:, 2]
    pcd_pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd_pcd


def visualize_voxel(voxel_x, voxel_y, voxel_z, filter, color):
    pcd = o3d.geometry.PointCloud()
    voxel_x = voxel_x[filter].reshape(-1, 1)
    voxel_y = voxel_y[filter].reshape(-1, 1)
    voxel_z = voxel_z[filter].reshape(-1, 1)
    pcd_points = np.concatenate(
        [voxel_x + 0.5, voxel_y + 0.5, voxel_z + 0.5], axis=1
    )
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    colors = np.zeros(pcd_points.shape)
    colors[:, 0] = color[0]
    colors[:, 1] = color[1]
    colors[:, 2] = color[2]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    min_bound = [voxel_x.min(), voxel_y.min(), voxel_z.min()]
    min_bound = np.array(min_bound)
    max_bound = [voxel_x.max(), voxel_y.max(), voxel_z.max()]
    max_bound = np.array(max_bound) + 1.0
    voxel = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd, 1., min_bound, max_bound
    )

    # bbox = voxel.get_axis_aligned
    return voxel


def visualize_voxel_highlight(
    voxel_x, voxel_y, voxel_z, filter, highlight_filter, color, highlight_color
):
    filter_voxel_x = voxel_x[filter & (~highlight_filter)].reshape(-1, 1)
    filter_voxel_y = voxel_y[filter & (~highlight_filter)].reshape(-1, 1)
    filter_voxel_z = voxel_z[filter & (~highlight_filter)].reshape(-1, 1)
    pcd_points = np.concatenate(
        [filter_voxel_x + 0.5, filter_voxel_y + 0.5, filter_voxel_z + 0.5],
        axis=1
    )
    colors = np.zeros(pcd_points.shape)
    colors[:, 0] = color[0]
    colors[:, 1] = color[1]
    colors[:, 2] = color[2]

    highlight_voxel_x = voxel_x[highlight_filter].reshape(-1, 1)
    highlight_voxel_y = voxel_y[highlight_filter].reshape(-1, 1)
    highlight_voxel_z = voxel_z[highlight_filter].reshape(-1, 1)

    highlight_pcd_points = np.concatenate(
        [
            highlight_voxel_x + 0.5,
            highlight_voxel_y + 0.5,
            highlight_voxel_z + 0.5,
        ],
        axis=1
    )
    highlight_colors = np.zeros(highlight_pcd_points.shape)
    highlight_colors[:, 0] = highlight_color[0]
    highlight_colors[:, 1] = highlight_color[1]
    highlight_colors[:, 2] = highlight_color[2]

    pcd_points = np.concatenate([pcd_points, highlight_pcd_points], axis=0)
    pcd_colors = np.concatenate([colors, highlight_colors], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    min_bound = [voxel_x.min(), voxel_y.min(), voxel_z.min()]
    min_bound = np.array(min_bound)
    max_bound = [voxel_x.max(), voxel_y.max(), voxel_z.max()]
    max_bound = np.array(max_bound) + 1.0
    voxel = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd, 1., min_bound, max_bound
    )

    # bbox = voxel.get_axis_aligned
    return voxel


def visualize_bbox(voxel_x, voxel_y, voxel_z, color=[1, 1, 1]):
    min_bound = [voxel_x.min(), voxel_y.min(), voxel_z.min()]
    min_bound = np.array(min_bound)
    max_bound = [voxel_x.max(), voxel_y.max(), voxel_z.max()]
    max_bound = np.array(max_bound) + 1.0

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    bbox.color = color
    return bbox


def visualize_coordinate_frame_centered(size=1.0, transform=np.eye(4)):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size).transform(
        transform
    )
    return frame


def visualize_arrow(
    scale=1.0,
    translation=np.zeros(3),
    direction=np.array([0, 0, 1.0]),
    color=[1, 0, 0]
):
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=1 * scale,
        cone_radius=1.5 * scale,
        cylinder_height=5 * scale,
        cone_height=4 * scale
    )

    z_axis = direction
    x_axis = np.array([-z_axis[2], 0, z_axis[0]])
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.array([x_axis, y_axis, z_axis]).T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    arrow.transform(transform)
    arrow.paint_uniform_color(color)
    return arrow


def visualize_mesh(vertices, triangles, color=[1, 0, 0]):
    vert = o3d.utility.Vector3dVector(vertices)
    tria = o3d.utility.Vector3iVector(triangles)
    mesh = o3d.geometry.TriangleMesh(vert, tria)
    mesh.paint_uniform_color(color)
    return mesh


def get_color_picks():
    color_pick = np.zeros((8, 3))
    color_pick[0] = np.array([1., 0., 0.])
    color_pick[1] = np.array([0., 1.0, 0.])
    color_pick[2] = np.array([0., 0., 1.])
    color_pick[3] = np.array([252 / 255, 169 / 255, 3 / 255])
    color_pick[4] = np.array([252 / 255, 3 / 255, 252 / 255])
    color_pick[5] = np.array([20 / 255, 73 / 255, 82 / 255])
    color_pick[6] = np.array([22 / 255, 20 / 255, 82 / 255])
    color_pick[7] = np.array([60 / 255, 73 / 255, 10 / 255])
    return color_pick


def setup_render(center, eye, up):
    # Create a renderer with the desired image size
    img_width = 640
    img_height = 480
    render = o3d.visualization.rendering.OffscreenRenderer(
        img_width, img_height
    )

    # Pick a background colour (default is light gray)
    # render.scene.set_background([0.1, 0.2, 0.3, 1.0])  # RGBA

    # Since the arrow material is unlit, it is not necessary to change the scene lighting.
    #render.scene.scene.enable_sun_light(False)
    #render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
    # Optionally set the camera field of view (to zoom in a bit)
    vertical_field_of_view = 50.0  # between 5 and 90 degrees
    aspect_ratio = img_width / img_height  # azimuth over elevation
    near_plane = 0.1
    far_plane = 100.0
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    render.scene.camera.set_projection(
        vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type
    )

    # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
    render.scene.camera.look_at(center, eye, up)
    return render


def create_material(rgba, shader):
    mtl = o3d.visualization.rendering.Material()
    mtl.base_color = rgba
    mtl.shader = shader
    return mtl
