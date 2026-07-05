from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
import rospy
import sys
import numpy as np
import copy
import cv2
import numpy as np
import open3d as o3d
from PIL import Image as PIL_Image
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
from lang_sam import LangSAM
# ROS library
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose



def create_pcd(
    depth_im: np.ndarray,
    cam_intr: np.ndarray,
    color_im: np.ndarray = None,
    cam_extr: np.ndarray = np.eye(4)
):
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_o3d.intrinsic_matrix = cam_intr
    depth_im_o3d = o3d.geometry.Image(depth_im)
    if color_im is not None:
        color_im_o3d = o3d.geometry.Image(color_im)
        rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(
            color_im_o3d,
            depth_im_o3d,
            depth_scale=1,
            convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud().create_from_rgbd_image(
            rgbd, intrinsic_o3d, extrinsic=cam_extr
        )
    else:
        pcd = o3d.geometry.PointCloud().create_from_depth_image(
            depth_im_o3d, intrinsic_o3d, extrinsic=cam_extr, depth_scale=1
        )
    return pcd


def get_result(target_name: str, camera_info: CameraInfo, rgb_img: Image, depth_img: Image, debug_mode: bool = True):
    bridge = CvBridge()
    model = LangSAM('vit_b')  #,'./sam_vit_b_01ec64.pth')

    cam_intr = np.array(camera_info.K).reshape((3, 3))
    rgb_im = bridge.imgmsg_to_cv2(rgb_img, 'rgb8')
    depth_im = bridge.imgmsg_to_cv2(depth_img,
                                    '32FC1').astype(np.float32) / 1000
    image_pil = PIL_Image.fromarray(rgb_im)

    # for debugging
    #rgb_im_path = "data/color.png"
    #depth_im_path = "data/depth.png"
    #rgb_im = cv2.cvtColor(cv2.imread(rgb_im_path), cv2.COLOR_BGR2RGB)  # for debug
    #depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
    #fx, fy = 459.906, 460.156
    #cx, cy = 347.191, 256.039
    #cam_intr = np.array([
    #    [fx, 0, cx],
    #    [0, fy, cy],
    #    [0, 0, 1]
    #])

    if debug_mode:
        plt.imshow(rgb_im)
        plt.show()
        # image_pil.show()
        plt.imshow(depth_im)
        plt.show()

    masks, boxes, phrases, logits = model.predict(image_pil, target_name, box_threshold=0.3, text_threshold=0.2)
    # target_mask = np.asarray(masks[0])
    select_idx = np.argmax(logits)
    target_mask = np.asarray(masks[select_idx]) 
    # target_mask = cv2.erode(target_mask.astype(float), np.ones((7, 7))).astype(bool)
    if debug_mode:
        plt.imshow(target_mask)
        plt.show()


    for i in range(len(logits)):
        mask = masks[i]
        box = boxes[i]
        print('logits: ', logits[i])
        if debug_mode:
            image_cv = np.array(rgb_im)
            x1, y1, x2, y2 = [int(c) for c in box]
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
            print("mask: ")
            print(mask)
            mask = np.asarray(mask).astype(int)
            print('mask shape:', mask.shape)
            print('image shape: ', image_cv.shape)
            color = np.array([0,255,0], dtype='uint8')
            masked_img = np.where(mask[...,None], color, image_cv)
            out = cv2.addWeighted(image_cv, 0.8, masked_img, 0.2,0)
            cv2.imshow("Image", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    select_idx = np.argmax(logits)
    mask = masks[select_idx]
    box = boxes[select_idx]

    if debug_mode:
        image_cv = np.array(rgb_im)
        x1, y1, x2, y2 = [int(c) for c in box]
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
        print("mask: ")
        print(mask)
        mask = np.asarray(mask).astype(int)
        print('mask shape:', mask.shape)
        print('image shape: ', image_cv.shape)
        color = np.array([0,255,0], dtype='uint8')
        masked_img = np.where(mask[...,None], color, image_cv)
        out = cv2.addWeighted(image_cv, 0.8, masked_img, 0.2,0)
        cv2.imshow("Image", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    scene_pcd = create_pcd(depth_im, cam_intr, color_im=rgb_im)
    scene_pts = np.asarray(scene_pcd.points)
    scene_rgb = np.asarray(scene_pcd.colors)

    masked_depth_im = depth_im * target_mask
    target_pcd = create_pcd(masked_depth_im, cam_intr, color_im=rgb_im)

    scene_kdtree = KDTree(scene_pts)
    target_pts = np.asarray(target_pcd.points)
    _, target_indices = scene_kdtree.query(target_pts)
    target_mask_3d = np.zeros(scene_pts.shape[0], dtype=np.bool_)
    target_mask_3d[target_indices] = True

    background_mask_3d = np.ones(scene_pts.shape[0], dtype=np.bool_)
    # background_mask_3d[table_indices] = False
    background_mask_3d[target_indices] = False

    if debug_mode:
        pcd_vis = copy.deepcopy(scene_pcd)
        pcd_colors = np.asarray(pcd_vis.colors)
        pcd_colors[target_mask_3d] = (1, 0, 0)
        pcd_colors[background_mask_3d] = (0, 0, 1)
        o3d.visualization.draw_geometries([pcd_vis])


if __name__ == "__main__":
    rospy.init_node('perception_test')

    rospy.sleep(1.0)
    target = sys.argv[1]
    prefix = sys.argv[2]
    # prefix = "camera1"
    cam_info = rospy.wait_for_message("/%s/color/camera_info" %(prefix), CameraInfo)
    color_img = rospy.wait_for_message('/%s/color/image_raw' %(prefix), Image)
    depth_img = rospy.wait_for_message('/%s/aligned_depth_to_color/image_raw' %(prefix), Image)
    # target = "something"

    get_result(target, cam_info, color_img, depth_img, True)