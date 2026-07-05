import cv2
import numpy as np

np.float = float
import ros_numpy as rnp


def merge_rgba_fields(cloud_arr):
    '''Takes an array with named np.uint8 fields 'r', 'g', 'b', and 'a', and returns an array in
    which they have been merged into a single np.float32 'rgba' field. The first byte of this
    field is the 'r' uint8, the second is the 'g', uint8, the third is the 'b' uint8, and
    the fourth is the 'a' uint8.

    This is the way that pcl likes to handle RGBA colors for some reason.
    '''
    r = np.asarray(cloud_arr['r'], dtype=np.uint32)
    g = np.asarray(cloud_arr['g'], dtype=np.uint32)
    b = np.asarray(cloud_arr['b'], dtype=np.uint32)
    a = np.asarray(cloud_arr['a'], dtype=np.uint32)
    rgba_arr = np.array(
        (r << 24) | (g << 16) | (b << 8) | (a << 0), dtype=np.uint32
    )

    # not sure if there is a better way to do this. i'm changing the type of the array
    # from uint32 to float32, but i don't want any conversion to take place -jdb
    rgba_arr.dtype = np.float32

    # create a new array, without r, g, b, and a, but with rgba float32 field
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if field_name not in ('r', 'g', 'b', 'a'):
            new_dtype.append((field_name, field_type))
    new_dtype.append(('rgba', np.float32))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        if field_name == 'rgba':
            new_cloud_arr[field_name] = rgba_arr
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]

    return new_cloud_arr


def split_rgba_field(cloud_arr):
    '''Takes an array with a named 'rgba' float32 field, and returns an array in which
    this has been split into 4 uint 8 fields: 'r', 'g', 'b', and 'a'.

    (pcl stores rgba in packed 32 bit floats)
    '''
    rgba_arr = cloud_arr['rgba'].copy()
    rgba_arr.dtype = np.uint32
    r = np.asarray((rgba_arr >> 24) & 255, dtype=np.uint8)
    g = np.asarray((rgba_arr >> 16) & 255, dtype=np.uint8)
    b = np.asarray((rgba_arr >> 8) & 255, dtype=np.uint8)
    a = np.asarray((rgba_arr >> 0) & 255, dtype=np.uint8)

    # create a new array, without rgba, but with r, g, b, and a fields
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if not field_name == 'rgba':
            new_dtype.append((field_name, field_type))
    new_dtype.append(('r', np.uint8))
    new_dtype.append(('g', np.uint8))
    new_dtype.append(('b', np.uint8))
    new_dtype.append(('a', np.uint8))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        if field_name == 'r':
            new_cloud_arr[field_name] = r
        elif field_name == 'g':
            new_cloud_arr[field_name] = g
        elif field_name == 'b':
            new_cloud_arr[field_name] = b
        elif field_name == 'a':
            new_cloud_arr[field_name] = a
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]
    return new_cloud_arr


def cloud_array_to_point_list(cloud_array):
    # cloud_array = rnp.point_cloud2.pointcloud2_to_array(point_cloud)
    points = np.zeros((cloud_array.size, 4), dtype=np.float32)
    points[..., 0] = cloud_array['x'].flatten()
    points[..., 1] = cloud_array['y'].flatten()
    points[..., 2] = cloud_array['z'].flatten()
    points[..., 3] = cloud_array['rgb'].flatten()
    return points.tolist()


def depth_color_to_pcd(depth, color, intrinsics):
    """
    assuming the camera has its forward-vec in the reverse direction of the lookat
    since the depth value is positive (inverted), we need to get the reverse of it
    to find the 
    """
    fx = intrinsics[0, 0]
    ppx = intrinsics[0, 2]
    fy = intrinsics[1, 1]
    ppy = intrinsics[1, 2]

    print('intrinsics: ')
    print(intrinsics)

    # if self.mode == 'real':
    #     depth = np.array(depth_numpy_image) / 1000
    # else:
    depth = np.array(depth)
    i, j = np.indices(depth.shape)
    x = (j - ppx) / fx * depth
    y = (i - ppy) / fy * depth
    x = x.reshape(-1)
    y = y.reshape(-1)
    depth = depth.reshape(-1)
    pcd = np.array([x, y, depth]).T
    color = color.reshape((-1, 3))
    mask = np.nonzero(pcd[:, 2])
    pcd = pcd[mask]
    color = color[mask]
    return pcd, color


def pcd_to_depth(pcd, intrinsics, height, width):
    """
    given the pcd in the camera frame, use the intrinsics to generate
    a depth image of the pcd.
    """
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]
    projected_pcd = np.zeros((len(pcd), 2))
    projected_pcd[:, 0] = pcd[:, 0] / pcd[:, 2] * fx + cx
    projected_pcd[:, 1] = pcd[:, 1] / pcd[:, 2] * fy + cy
    depth = pcd[:, 2]
    projected_pcd = np.floor(projected_pcd).astype(int)
    # mask out outside pts
    valid_mask = (projected_pcd[:, 1] >= 0)
    valid_mask &= (projected_pcd[:, 1] < height)
    valid_mask &= (projected_pcd[:, 0] >= 0)
    valid_mask &= (projected_pcd[:, 0] < width)
    projected_pcd = projected_pcd[valid_mask]
    depth_img = np.zeros((height, width)).astype(float)
    depth_img[projected_pcd[:, 1], projected_pcd[:, 0]] = depth
    depth_img = cv2.boxFilter(np.float32(depth_img), -1, (5, 5))
    return depth_img
