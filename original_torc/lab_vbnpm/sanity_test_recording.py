import sys
import cv2
import json
import glob
import numpy as np
import open3d as o3d

folder = sys.argv[1]

with open(f'{folder}/d455_config.json', 'r') as f:
    d435_config = json.load(f)
    cam_intr = d435_config['intrinsic_matrix']
    print(cam_intr)

depth_img = cv2.imread(f'{folder}/d455_depth.png', cv2.IMREAD_ANYDEPTH)
color_img = cv2.imread(f'{folder}/d455_color.jpg', cv2.IMREAD_UNCHANGED)
cam_pose = np.loadtxt(f'{folder}/d455_pose.txt')
intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
intrinsic_o3d.intrinsic_matrix = cam_intr
depth_im_o3d = o3d.geometry.Image(depth_img)
color_im_o3d = o3d.geometry.Image(color_img)
rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(
    color_im_o3d,
    depth_im_o3d,
    depth_scale=1000,
    convert_rgb_to_intensity=False
)
total_pcd = o3d.geometry.PointCloud().create_from_rgbd_image(
    rgbd, intrinsic_o3d, extrinsic=np.linalg.inv(cam_pose)
)
o3d.visualization.draw_geometries([total_pcd])

with open(f'{folder}/d435_config.json', 'r') as f:
    d435_config = json.load(f)
    cam_intr = d435_config['intrinsic_matrix']
    print(cam_intr)

for i in range(100):
    print(i)
    if len(glob.glob(f'./{folder}/*/{i:04d}-*')) < 3:
        print('Done')
        break
    depth_img = cv2.imread(
        f'{folder}/depth/{i:04d}-depth.png', cv2.IMREAD_ANYDEPTH
    )
    color_img = cv2.imread(
        f'{folder}/color/{i:04d}-color.jpg', cv2.IMREAD_UNCHANGED
    )
    cam_pose = np.loadtxt(f'{folder}/poses/{i:04d}-pose.txt')

    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_o3d.intrinsic_matrix = cam_intr
    depth_im_o3d = o3d.geometry.Image(depth_img)
    color_im_o3d = o3d.geometry.Image(color_img)
    rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(
        color_im_o3d,
        depth_im_o3d,
        depth_scale=1000,
        convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud().create_from_rgbd_image(
        rgbd, intrinsic_o3d, extrinsic=np.linalg.inv(cam_pose)
    )
    total_pcd.points = o3d.utility.Vector3dVector(
        np.concatenate((total_pcd.points, pcd.points))
    )
    total_pcd.colors = o3d.utility.Vector3dVector(
        np.concatenate((total_pcd.colors, pcd.colors))
    )

o3d.visualization.draw_geometries([total_pcd])
