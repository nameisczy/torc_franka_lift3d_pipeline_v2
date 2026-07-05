"""
convert from RGBD images to point cloud
TODO:
merge this and occlusion handling code together as a single service code
"""
import numpy as np
import rospy
import message_filters
import tf2_ros
import cv_bridge
import transformations as tf
import numpy as np
import open3d as o3d
import utils.visual_utils as vutils
from sensor_msgs.msg import Image, CameraInfo, PointField, PointCloud2
from std_msgs.msg import Header


def points_to_pcd2(points, rgba, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx3 array of xyz positions (m)
        rgba: Nx4 rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = np.concatenate([points, rgba], axis=1)
    data = data.astype(dtype).tobytes()

    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzrgba')]

    header = Header(frame_id=parent_frame, stamp=rospy.Time.now())

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 7),
        row_step=(itemsize * 7 * points.shape[0]),
        data=data
    )



class VisiblePCD:
    def __init__(self, camera_num=2):
        # for extrinsics
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.tf_listen = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = cv_bridge.CvBridge()

        self.pcd_pub = rospy.Publisher('velodyne_points', PointCloud2)

        # get the camera infos
        self.cam_infos = []
        for i in range(camera_num):
            # print('camera%d/color/camera_info'%(i))
            cam_info = rospy.wait_for_message('camera%d/color/camera_info'%(i), CameraInfo)
            self.cam_infos.append(cam_info)
            # self.intrinsics.append(np.array(cam_info.P).reshape((3,4)))

        self.depth_img_subs = []
        for i in range(camera_num):
            depth_img_sub_i = message_filters.Subscriber(
            'camera%d/aligned_depth_to_color/image_raw'%(i), Image
            )
            self.depth_img_subs.append(depth_img_sub_i)
        #     depth_cam_info_sub_i = message_filters.Subscriber(
        #     'camera%d/color/camera_info', CameraInfo
        # )
        self.depth_sub = message_filters.ApproximateTimeSynchronizer(
            self.depth_img_subs,
            10,
            1.0,
        )
        self.depth_sub.registerCallback(self.depth_image_cb)

        print('finished setting up.')

    def depth_image_cb(self, *args):
        """
        convert the depth images to point cloud
        """
        depth_imgs = []
        extrinsics = []

        for i in range(len(args)):
            frame_i = args[i].header.frame_id

            world_T_cam_i = self.tf_buffer.lookup_transform(
                'base_link',
                frame_i,
                rospy.Time(args[i].header.stamp.secs, args[i].header.stamp.nsecs),
                # rospy.Time(),
                rospy.Duration(5.0),
            )
            world_T_cam_i = world_T_cam_i.transform
            qx = world_T_cam_i.rotation.x
            qy = world_T_cam_i.rotation.y
            qz = world_T_cam_i.rotation.z
            qw = world_T_cam_i.rotation.w
            x = world_T_cam_i.translation.x
            y = world_T_cam_i.translation.y
            z = world_T_cam_i.translation.z
            mat = tf.quaternion_matrix([qw, qx, qy, qz])
            mat[:3,3] = np.array([x,y,z])
            extrinsics.append(mat)

            depth_img = self.bridge.imgmsg_to_cv2(args[i], 'passthrough') / 1000.0            
            depth_imgs.append(depth_img)


        pcd = []
        for i in range(len(args)):
            pcd_i = self.depth_to_pcd(depth_imgs[i], self.cam_infos[i], extrinsics[i])
            pcd.append(pcd_i)

        pcd = np.concatenate(pcd, axis=0)
        rgba = np.zeros((len(pcd),4))
        rgba[:,3] = 1
        rgba[:,0] = 1
        pcd_msg = points_to_pcd2(pcd, rgba, 'base_link')
        self.pcd_pub.publish(pcd_msg)
        # debug: visualize
        # v_frame = vutils.visualize_coordinate_frame_centered()
        # vpcd = vutils.visualize_pcd(pcd, [1,0,0])
        # o3d.visualization.draw_geometries([v_frame, vpcd])
        # input('done.')


    def depth_to_pcd(self, depth_img, cam_info: CameraInfo, world_T_cam):
        # depth_img: unit is meters (divide by 1000 for real case)
        cam_intrinsics = np.array(cam_info.P).reshape((3,4))[:3,:3]
        fx = cam_intrinsics[0][0]
        fy = cam_intrinsics[1][1]
        cx = cam_intrinsics[0][2]
        cy = cam_intrinsics[1][2]

        depth_y, depth_x = np.indices((cam_info.height, cam_info.width)).astype(float)
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
        pcd = world_T_cam[:3,:3].dot(pcd.T).T + world_T_cam[:3,3]

        return pcd
    
if __name__ == "__main__":
    rospy.init_node('visible_pcd_server')
    # rospy.sleep(1)
    server = VisiblePCD(2)
    rospy.spin()
