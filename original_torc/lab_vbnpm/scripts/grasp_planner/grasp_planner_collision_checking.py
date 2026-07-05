from grasp_planner.grasp_planner import *

from utils.print_color import *

import transformations as tf
from utils.conversions import pose_to_matrix, matrix_to_pose

from collision_checker.hppfcl import *

# CONSTANTS
GRIPPER_EE_OFFSET = tf.identity_matrix()
GRIPPER_EE_OFFSET[:3, 3] = [0, 0, -.135]

# SETTINGS
STATIC_COLLISION = True
SENSOR_COLLISION = True
VISUALIZE = False

class GraspPlannerHPPFCL(GraspPlanner):
    def __init__(self, static_collision=STATIC_COLLISION, sensor_collision=SENSOR_COLLISION, visualize=VISUALIZE, **kwargs):
        super().__init__(**kwargs)

        printPurple("Starting up GraspPlanner with HPP-FCL collision-checker")

        rp = rospkg.RosPack()
        motoman_pkg_path = rp.get_path('motoman')
        lab_pkg_path = rp.get_path('lab_vbnpm')

        # robot_urdf = motoman_pkg_path + '/motoman_sda10f_moveit_config/config/gazebo_motoman_sda10f.urdf'
        gripper_urdf = lab_pkg_path + '/robots/robotiq.urdf'

        self.collision_checker = HPPFCL(urdf_path=gripper_urdf, 
                                        static_collision=static_collision,
                                        sensor_collision=sensor_collision,
                                        visualize=visualize)
        
        
    def gripper_collision(self, gripper_pose, static_only, s_height, voxels, point_cloud=None):

        gripper_pose = gripper_pose @ GRIPPER_EE_OFFSET
            
        quat_wxyz = tf.quaternion_from_matrix(gripper_pose)

        quat_xyzw = np.roll(quat_wxyz, -1)

        q = np.concatenate([gripper_pose[:3, 3], quat_xyzw, [0]*6])

        res = self.collision_checker.query(q)

#         printPink(res)
        
        return res
    
    def update_point_cloud(self, point_cloud):
        self.collision_checker.update_perception(point_cloud)
