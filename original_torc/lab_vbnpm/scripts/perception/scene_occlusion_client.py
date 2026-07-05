import rospy
from lab_vbnpm.msg import SceneOcclusion, VoxelGridBool
from lab_vbnpm.srv import InitSceneOcclusion, InitSceneOcclusionRequest, InitSceneOcclusionResponse, \
                            UpdateSceneOcclusion, UpdateSceneOcclusionRequest, UpdateSceneOcclusionResponse, \
                            GetSceneOcclusion, GetSceneOcclusionRequest, GetSceneOcclusionResponse, \
                            GetSceneOcclusionPCD, GetSceneOcclusionPCDRequest, GetSceneOcclusionPCDResponse
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Pose, Transform
import transformations as tf
import tf2_ros
import numpy as np
import rospy
import cv_bridge
import open3d as o3d
import utils.visual_utils as vutils
import matplotlib.pyplot as plt



def msg_transform_to_pose(transform: Transform):
    qx = transform.rotation.x
    qy = transform.rotation.y
    qz = transform.rotation.z
    qw = transform.rotation.w

    x = transform.translation.x
    y = transform.translation.y
    z = transform.translation.z

    mat = tf.quaternion_matrix([qw,qx,qy,qz])
    mat[0,3] = x
    mat[1,3] = y
    mat[2,3] = z
    return mat

def main():
    rospy.init_node("scene_occlusion_client")
    # rospy.sleep(1.0)
    bridge = cv_bridge.CvBridge()
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    # scene1
    padding_x = 0.02
    padding_y = 0.06
    padding_z = 0.02

    pose_x = 0.4+0.925-0.175+padding_x  # adding some padding
    pose_y = 0-0.45+padding_y
    pose_z = 1.15-0.15+padding_z
    size_x = 0.175*2-padding_x*2-padding_x*2
    size_y = 0.45*2-padding_y*2
    size_z = 0.15*2-padding_z*2

    # middle point: 0.4+0.925-0.175=
    frame = vutils.visualize_coordinate_frame_centered()

    vbox = o3d.geometry.TriangleMesh.create_box(width=size_x, height=size_y, depth=size_z)
    vbox.translate(np.array([pose_x, pose_y, pose_z]))

    o3d.visualization.draw_geometries([vbox, frame])

    base_link_name = 'world' #'base'
    camera_link_name = 'camera_depth_link'#'camera_color_optical_frame'
    cam_info_topic = '/depth/camera_info'#'/camera/aligned_depth_to_color/camera_info'
    depth_img_topic = '/depth/image_raw'#'/camera/aligned_depth_to_color/image_raw'
    color_img_topic ='/rgb/image_raw'#'/camera/color/image_raw'


    # visualize the pcd with the box
    rospy.sleep(1.0)
    t = tfBuffer.lookup_transform(base_link_name, camera_link_name, rospy.Time())
    # assuming base_link is the world frame
    cam_extrinsics = msg_transform_to_pose(t.transform)
    cam_info = rospy.wait_for_message(cam_info_topic, CameraInfo)
    cam_intrinsics = np.array(cam_info.P).reshape((3,4))[:3,:3]
    fx = cam_intrinsics[0][0]
    fy = cam_intrinsics[1][1]
    cx = cam_intrinsics[0][2]
    cy = cam_intrinsics[1][2]
    pcd = np.zeros((cam_info.height, cam_info.width, 3))
    depth_img = rospy.wait_for_message(depth_img_topic, Image)
    depth_img = bridge.imgmsg_to_cv2(depth_img, 'passthrough')# / 1000

    color_img = rospy.wait_for_message(color_img_topic, Image)
    color_img = bridge.imgmsg_to_cv2(color_img, 'passthrough')

    depth_y, depth_x = np.indices((cam_info.height, cam_info.width)).astype(float)
    depth_y_ind = np.array(depth_y).astype(int)
    depth_x_ind = np.array(depth_x).astype(int)
    pcd_y_ind = depth_y_ind.reshape((-1))
    pcd_x_ind = depth_x_ind.reshape((-1))


    print('depth_x shape: ')
    print(depth_x.shape)
    print('depth shape: ')
    print(depth_img.shape)
    depth_x = depth_x - cx
    depth_x = depth_x / fx * depth_img
    depth_x = depth_x.reshape((-1))
    depth_y = depth_y - cy
    depth_y = depth_y / fy * depth_img
    depth_y = depth_y.reshape((-1))
    depth_z = np.array(depth_img)
    depth_z = depth_z.reshape((-1))

    valid_mask = depth_z <= 1.25  # clip the depth


    pcd = np.array([depth_x, depth_y, depth_z]).T
    pcd = pcd[valid_mask]
    pcd = cam_extrinsics[:3,:3].dot(pcd.T).T + cam_extrinsics[:3,3]
    pcd_color = color_img[pcd_y_ind, pcd_x_ind] / 255
    pcd_color = pcd_color[valid_mask]


    cam_pcd_from_depth = np.array(pcd)  # in the world frame
    cam_pcd_color = np.array(pcd_color)




    vpcd = vutils.visualize_pcd(pcd, color=pcd_color)

    o3d.visualization.draw_geometries([vbox, frame, vpcd])


    resol = 0.01

    rospy.wait_for_service('init_scene_occlusion')
    init_scene_occlusion = rospy.ServiceProxy('init_scene_occlusion', InitSceneOcclusion)
    req = InitSceneOcclusionRequest()
    req.pose.position.x = pose_x
    req.pose.position.y = pose_y
    req.pose.position.z = pose_z
    req.pose.orientation.w = 1
    req.pose.orientation.x = 0
    req.pose.orientation.y = 0
    req.pose.orientation.z = 0
    occluded_pose = np.eye(4)
    occluded_pose[0,3] = pose_x
    occluded_pose[1,3] = pose_y
    occluded_pose[2,3] = pose_z
    req.resols.x = resol
    req.resols.y = resol
    req.resols.z = resol
    req.xyz_size.x = size_x
    req.xyz_size.y = size_y
    req.xyz_size.z = size_z
    init_scene_occlusion(req)


    world_in_occ = np.linalg.inv(occluded_pose)
    cam_pcd_in_occ = world_in_occ[:3,:3].dot(cam_pcd_from_depth.T).T + world_in_occ[:3,3]
    cam_pcd_in_occ = cam_pcd_in_occ / resol
    vpcd_cam = vutils.visualize_pcd(cam_pcd_in_occ, color=pcd_color)


    while True:
        input('start...')
        rospy.wait_for_service("update_scene_occlusion")
        update_scene_occlusion = rospy.ServiceProxy("update_scene_occlusion", UpdateSceneOcclusion)
        req = UpdateSceneOcclusionRequest()
        req.depth_img = rospy.wait_for_message(depth_img_topic, Image)
        req.img_frame = camera_link_name
        req.cam_info_topic = cam_info_topic
        update_scene_occlusion(req)

        rospy.wait_for_service("get_scene_occlusion")
        get_scene_occlusion = rospy.ServiceProxy("get_scene_occlusion", GetSceneOcclusion)

        msg = get_scene_occlusion()
        occluded = msg.data.voxel.data
        shape_x = msg.data.voxel.size_x
        shape_y = msg.data.voxel.size_y
        shape_z = msg.data.voxel.size_z
        occluded = np.array(occluded).reshape((shape_x,shape_y,shape_z)).astype(bool)
        resols = [msg.data.voxel.resols.x,msg.data.voxel.resols.y,msg.data.voxel.resols.z]
        resols = np.array(resols)
        voxel_x, voxel_y, voxel_z = np.indices((shape_x,shape_y,shape_z)).astype(float)

        vvoxel = vutils.visualize_voxel(voxel_x, voxel_y, voxel_z, occluded, [1,0,0])

        # transform image pcd to voxel


        o3d.visualization.draw_geometries([vvoxel, vpcd_cam])

        cam_pcd_from_depth = np.array(pcd)
        cam_pcd_color = np.array(pcd_color)

        # pcd
        rospy.wait_for_service("get_scene_occlusion_pcd")
        get_scene_occlusion_pcd = rospy.ServiceProxy("get_scene_occlusion_pcd", GetSceneOcclusionPCD)

        msg = get_scene_occlusion_pcd()
        # get the pcd in voxel pose
        pcd = msg.data
        pcd = np.array(pcd).reshape((-1,3))
        print('pcd: ')
        print(pcd)
        world_in_occ = np.linalg.inv(occluded_pose)
        pcd_in_occ = world_in_occ[:3,:3].dot(pcd.T).T + world_in_occ[:3,3]
        pcd_in_occ = pcd_in_occ / resol
        v_pcd = vutils.visualize_pcd(pcd_in_occ, [0,0,1])

        o3d.visualization.draw_geometries([vvoxel, v_pcd, vpcd_cam])


if __name__ == "__main__":
    main()