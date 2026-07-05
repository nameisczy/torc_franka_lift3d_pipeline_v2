"""
create a ros service that creates scene occlusion
given depth image, and target segmentation, obtain
occlusion pcds

services:
InitSceneOcclusion: initialize the occlusion pose, size and resols
UpdateSceneOcclusion: update the scene occlusion from depth image
GetSceneOcclusion: obtain current scene occlusion voxel
GetSceneOcclusionPCD: obtain current scene occlusion pcd

"""

from geometry_msgs.msg import Pose, Transform
from sensor_msgs.msg import CameraInfo
import transformations as tf
from lab_vbnpm.msg import SceneOcclusion, VoxelGridBool
from lab_vbnpm.srv import InitSceneOcclusion, InitSceneOcclusionRequest, InitSceneOcclusionResponse, \
                            UpdateSceneOcclusion, UpdateSceneOcclusionRequest, UpdateSceneOcclusionResponse, \
                            GetSceneOcclusion, GetSceneOcclusionRequest, GetSceneOcclusionResponse, \
                            GetSceneOcclusionPCD, GetSceneOcclusionPCDRequest, GetSceneOcclusionPCDResponse
# import tf2_ros
import numpy as np
import rospy
import cv_bridge
import open3d as o3d
from utils import visual_utils as vutils
import matplotlib.pyplot as plt

class SceneOcclusionServer:
    def __init__(self):
        rospy.Service("init_scene_occlusion", InitSceneOcclusion, self.init_scene_occlusion)
        rospy.Service("update_scene_occlusion", UpdateSceneOcclusion, self.update_scene_occlusion)
        rospy.Service("get_scene_occlusion", GetSceneOcclusion, self.get_scene_occlusion)
        rospy.Service('get_scene_occlusion_pcd', GetSceneOcclusionPCD, self.get_scene_occlusion_pcd)
        self.bridge = cv_bridge.CvBridge()
        # self.tfBuffer = tf2_ros.Buffer()
        # self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.pose = None
        self.resols = None
        self.xyz_size = None  # float
        self.xyz_shape = None # int
        self.occluded = None
        self.voxel_x = None
        self.voxel_y = None
        self.voxel_z = None
        pass

    def msg_pose_to_pose(self, pose: Pose):
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w

        x = pose.position.x
        y = pose.position.y
        z = pose.position.z
    
        mat = tf.quaternion_matrix([qw,qx,qy,qz])
        mat[0,3] = x
        mat[1,3] = y
        mat[2,3] = z
        return mat

    def pose_to_msg_pose(self, pose):
        quat = tf.quaternion_from_matrix(pose)
        msg = Pose()
        msg.orientation.w = quat[0]
        msg.orientation.x = quat[1]
        msg.orientation.y = quat[2]
        msg.orientation.z = quat[3]

        msg.position.x = pose[0,3]
        msg.position.y = pose[1,3]
        msg.position.z = pose[2,3]

        return msg


    def msg_transform_to_pose(self, transform: Transform):
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


    def init_scene_occlusion(self, req: InitSceneOcclusionRequest):
        pose = self.msg_pose_to_pose(req.pose)
        resols = np.array([req.resols.x,req.resols.y,req.resols.z])
        xyz_size = np.array([req.xyz_size.x,req.xyz_size.y,req.xyz_size.z])
        xyz_shape = np.ceil(xyz_size/resols).astype(int)
        occluded = np.ones(xyz_shape).astype(bool)
        voxel_x, voxel_y, voxel_z = np.indices(occluded.shape).astype(float)
        self.pose = pose
        self.resols = resols
        self.xyz_size = xyz_size
        self.xyz_shape = xyz_shape
        self.occluded = occluded
        self.voxel_x = voxel_x
        self.voxel_y = voxel_y
        self.voxel_z = voxel_z
        msg = InitSceneOcclusionResponse()
        msg.success = True
        return msg

    def update_scene_occlusion(self, req: UpdateSceneOcclusionRequest):
        depth_img = self.bridge.imgmsg_to_cv2(req.depth_img, 'passthrough') / 1000.0
        #img_frame = req.img_frame

        #t = self.tfBuffer.lookup_transform('world', img_frame, rospy.Time())
        # assuming base_link is the world frame
        cam_extrinsics = self.msg_transform_to_pose(req.cam_transform)
        # cam_info = rospy.wait_for_message(req.cam_info_topic, CameraInfo)
        cam_info = req.cam_info
        cam_intrinsics = np.array(cam_info.P).reshape((3,4))[:3,:3]

        if req.debug_mode:
            vworld_frame = vutils.visualize_coordinate_frame_centered()
            vcam_frame = vutils.visualize_coordinate_frame_centered(size=0.5, transform=cam_extrinsics)
            vpose_frame = vutils.visualize_coordinate_frame_centered(size=0.3, transform=self.pose)
            o3d.visualization.draw_geometries([vworld_frame, vcam_frame, vpose_frame])

        # * call the function
        occluded = self.get_occlusion(occlusion_pose=self.pose, occlusion_resols=self.resols,
                                      occlusion_xyz_size=self.xyz_size,
                                      depth_img=depth_img, camera_extrinsics=cam_extrinsics,
                                      camera_intrinsics=cam_intrinsics)
        # update by doing AND operation on the occluded cells
        self.occluded = self.occluded & occluded
        # NOTE: occluded has 1 for places that haven't been observed

        msg = UpdateSceneOcclusionResponse()
        msg.success = True
        return msg

    def get_scene_occlusion(self, req: GetSceneOcclusionRequest):
        if self.occluded is None:
            return GetSceneOcclusionResponse()

        msg = GetSceneOcclusionResponse()
        msg.data.voxel.header.frame_id = "world"
        msg.data.voxel.data = self.occluded.reshape((-1)).astype(np.uint32).tolist()
        msg.data.voxel.pose = self.pose_to_msg_pose(self.pose)
        msg.data.voxel.resols.x = self.resols[0]
        msg.data.voxel.resols.y = self.resols[1]
        msg.data.voxel.resols.z = self.resols[2]

        msg.data.voxel.size_x = self.occluded.shape[0]
        msg.data.voxel.size_y = self.occluded.shape[1]
        msg.data.voxel.size_z = self.occluded.shape[2]
        return msg

    def get_scene_occlusion_pcd(self, req: GetSceneOcclusionPCDRequest):
        if self.occluded is None:
            return GetSceneOcclusionPCDResponse()
        pcd = self.sample_pcd(self.occluded.shape, self.occluded)
        pcd = pcd * self.resols
        pcd = self.pose[:3,:3].dot(pcd.T).T + self.pose[:3,3]
        msg = GetSceneOcclusionPCDResponse()
        msg.data = pcd.astype(np.float64).reshape((-1)).tolist()
        return msg

    def get_occlusion(
        self,
        occlusion_pose, occlusion_resols, occlusion_xyz_size,
        depth_img,
        camera_extrinsics,
        camera_intrinsics,
    ):
        """
        generate the occlusion for the entire scene
        occlusion includes: 
        - object occupied space (after object is fully reconstructed)
        - occlusion due to known object, 
        - occlusion due to unknown object

        TODO: we might need to consider the seg_img when robot comes in the view
        NOTE: space that is not visible will be filled as 1
        """
        # * create the occlusion voxel

        occluded_shape = np.ceil(occlusion_xyz_size/occlusion_resols).astype(int)
        occluded = np.zeros(occluded_shape).astype(bool)
        voxel_x, voxel_y, voxel_z = np.indices(occluded.shape).astype(float)
        pt = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
                [0.5, 0.5, 0.5],
            ]
        )

        total_valid_mask = np.zeros(occluded_shape).astype(bool)
        for i in range(len(pt)):
            voxel_vecs = np.array([
                voxel_x,
                voxel_y,
                voxel_z,
            ]).transpose((1, 2, 3, 0)).reshape(-1, 3)
            voxel_vecs = voxel_vecs + pt[i].reshape(
                1, -1
            )  # get the middle point
            voxel_vecs = voxel_vecs * occlusion_resols
            transformed_voxels = occlusion_pose[:3, :3].dot(voxel_vecs.T).T
            transformed_voxels += occlusion_pose[:3, 3]

            # get to the image space
            cam_transform = np.linalg.inv(camera_extrinsics)
            transformed_voxels = cam_transform[:3, :3].dot(
                transformed_voxels.T
            ).T + cam_transform[:3, 3]

            cam_to_voxel_depth = np.array(transformed_voxels[:, 2])
            # intrinsics
            cam_intrinsics = camera_intrinsics
            fx = cam_intrinsics[0][0]
            fy = cam_intrinsics[1][1]
            cx = cam_intrinsics[0][2]
            cy = cam_intrinsics[1][2]
            transformed_voxels[:, 0] /= transformed_voxels[:, 2]
            transformed_voxels[:, 0] *= fx
            transformed_voxels[:, 0] += cx
            transformed_voxels[:, 1] /= transformed_voxels[:, 2]
            transformed_voxels[:, 1] *= fy
            transformed_voxels[:, 1] += cy
            transformed_voxels = np.floor(transformed_voxels).astype(int)
            voxel_depth = np.zeros((len(transformed_voxels)))
            valid_mask = transformed_voxels[:, 0] >= 0
            valid_mask &= transformed_voxels[:, 0] < len(depth_img[0])
            valid_mask &= transformed_voxels[:, 1] >= 0
            valid_mask &= transformed_voxels[:, 1] < len(depth_img)
            voxel_depth[valid_mask] = \
                depth_img[
                    transformed_voxels[valid_mask][:, 1],
                    transformed_voxels[valid_mask][:, 0]
                ]
            valid_mask = valid_mask.reshape(voxel_x.shape)
            voxel_depth = voxel_depth.reshape(voxel_x.shape)

            cam_to_voxel_depth = cam_to_voxel_depth.reshape(voxel_x.shape)
            included = cam_to_voxel_depth - voxel_depth >= 0.
            # depth > 0 in case we might want to mask certain regions
            included &= voxel_depth > 0.
            included &= valid_mask
            occluded |= included
            total_valid_mask |= (valid_mask & (voxel_depth > 0.))
            # as long as the space is valid for some calculation, it should be marked as valid

        # for space that is not valid (not updated), we set them as occupied
        occluded[~total_valid_mask] = 1
            
        # print(occluded.astype(int).sum() / valid_mask.astype(int).sum())
        # del cam_to_voxel_depth
        # del voxel_depth
        # del voxel_vecs
        # del transformed_voxels
        # del valid_mask

        return occluded


    def sample_pcd(self, voxel_shape, mask, n_sample=10):
        # sample voxels in te mask
        # obtain sample in one voxel cell
        # pcd is raw in the voxel. Need to transform later using resols and pose
        grid_sample = np.random.uniform(
            low=[0, 0, 0],
            high=[1, 1, 1],
            size=(n_sample, 3),
        )
        voxel_x, voxel_y, voxel_z = np.indices(
            voxel_shape
        ).astype(float)

        voxel_x = voxel_x[mask]
        voxel_y = voxel_y[mask]
        voxel_z = voxel_z[mask]

        total_sample = np.zeros((len(voxel_x), n_sample, 3))
        total_sample = total_sample + grid_sample
        total_sample = total_sample + np.array([
            voxel_x,
            voxel_y,
            voxel_z,
        ]).T.reshape(len(voxel_x), 1, 3)
        total_sample = total_sample.reshape((-1,3))
        # total_sample = total_sample.reshape(-1, 3) * np.array(self.resols)
        # del voxel_x
        # del voxel_y
        # del voxel_z
        return total_sample


def main():
    rospy.init_node("scene_occlusion_server")
    server = SceneOcclusionServer()
    rospy.spin()

if __name__ == "__main__":
    main()
