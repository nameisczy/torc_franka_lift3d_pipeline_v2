import os
import sys
import time
import json
import glob

import cv2
import numpy as np
import open3d as o3d


class TSDFVolume(object):
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size

        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def integrate(self, depth_img, mask_img, intrinsic, extrinsic):
        """
        Args:
            depth_img: The depth image.
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
        """
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            mask_img,
            depth_img,
            depth_scale=1000,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False,
        )

        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def get_grid(self):
        shape = (1, self.resolution, self.resolution, self.resolution)
        tsdf_grid = np.zeros(shape, dtype=np.float32)
        voxel_cloud = self._volume.extract_voxel_point_cloud()
        for point, color in zip(voxel_cloud.points, voxel_cloud.colors):
            i = int((point[0] - self.voxel_size / 2) / self.voxel_size)
            j = int((point[1] - self.voxel_size / 2) / self.voxel_size)
            k = int((point[2] - self.voxel_size / 2) / self.voxel_size)
            tsdf_grid[0, i, j, k] = color[0]
        return tsdf_grid

    def get_cloud(self):
        return self._volume.extract_point_cloud()


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '../recordings/recording_02/'

    instrinics = []
    for i in range(2):
        filename = f'{path}/camera{i}_config.json'
        with open(filename) as f:
            config = json.load(f)
            K = np.array(config['intrinsic_matrix'])
            width = config['width']
            height = config['height']
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=width,
                height=height,
                fx=K[0, 0],
                fy=K[1, 1],
                cx=K[0, 2],
                cy=K[1, 2],
            )
            instrinics.append(intrinsic)

    depth_file = f'{path}/camera0_depth.png'
    mask_file = f'{path}/camera0_mask.png'
    T_c0_w = np.loadtxt(f'{path}/camera0_pose.txt')
    depth = o3d.io.read_image(depth_file)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color[mask > 0] = [255, 0, 0]
    color = o3d.geometry.Image(color)

    tsdf = TSDFVolume(size=2.0, resolution=256)
    T_0 = np.eye(4)
    T_0[:3, 3] = [-1, -1, 0]
    t = time.time()
    tsdf.integrate(depth, color, instrinics[0], T_0)
    print('Integrate time:', time.time() - t)
    cloud = tsdf.get_cloud()
    o3d.visualization.draw_geometries([cloud])
    t = time.time()
    tsdf.get_grid()
    print('Get grid time:', time.time() - t)

    for depth_file in glob.glob(f'{path}/depth/*depth.png'):
        num = os.path.basename(depth_file)[:4]
        T_c1_w = np.loadtxt(f'{path}/poses/{num}-pose.txt')
        T_c1_c0 = np.linalg.inv(T_c1_w) @ T_c0_w
        T_c1_0 = T_c1_c0 @ T_0
        depth = o3d.io.read_image(depth_file)
        mask_file = f'{path}/mask/{num}-mask.png'
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color[mask > 0] = [255, 0, 0]
        color = o3d.geometry.Image(color)
        t = time.time()
        tsdf.integrate(depth, color, instrinics[1], T_c1_0)
        print('Integrate time:', time.time() - t)
        t = time.time()
        grid = tsdf.get_grid()
        print('Get grid time:', time.time() - t)

    cloud = tsdf.get_cloud()
    print(np.count_nonzero(grid))
    o3d.visualization.draw_geometries([cloud])
