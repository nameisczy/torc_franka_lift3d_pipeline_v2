from grasp_planner.grasp_planner import *
import pinocchio as pin
import hppfcl
from tracikpy import TracIKSolver
from utils.print_color import *

import transformations as tf
from utils.conversions import pose_to_matrix, matrix_to_pose

from pinocchio.visualize import MeshcatVisualizer

IK_LIM = 5

IS_EE_OFFSET = True

EE_OFFSET = tf.identity_matrix()
EE_OFFSET[:3, 3] = [0, 0, -.135]

SENSOR_COLLISION_GEOM_NAME = "octree"
SENSOR_VISUAL_GEOM_NAME = "bvh"

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
        gripper_urdf = lab_pkg_path + '/robots/robotiq.urdf' # '/robots/robotiq_arg85_description.URDF'
        robot_urdf = gripper_urdf
        package_dirs = []
        package_dirs.append(motoman_pkg_path)
        package_dirs.append(motoman_pkg_path + "/robotiq")
            
#        self.ik_solver = TracIKSolver(
#            robot_urdf, 
#            "base_link", 
#            "motoman_right_ee",
#        )

        self.gripper_geoms = []
        self.static_geoms = []
        self.sensor_geoms = [SENSOR_COLLISION_GEOM_NAME]

        self.visualize = visualize

        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(robot_urdf, package_dirs=package_dirs)

        # Record gripper geoms
        for geom in self.collision_model.geometryObjects:
            self.gripper_geoms.append(geom.name)

        # Add shelf to pinocchio
        self.static_collision = static_collision
        self.add_shelf(collision=static_collision)

        # Add empty octree object to pinocchio
        self.sensor_collision = sensor_collision
        self.add_octree(collision=sensor_collision)

        # Record shelf geoms
        for geom in self.collision_model.geometryObjects:
            if geom.name not in self.gripper_geoms:
                self.static_geoms.append(geom.name)  

        if visualize:
            self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
            self.viz.initViewer(open=True)

            self.viz.loadViewerModel()

        self.q = pin.neutral(self.model)

        self.create_data()

    def create_data(self, ):
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()

    def hppfcl(self, q):

        if self.visualize:
            self.q = q
            self.viz.display(q)

#        pin.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data, q)
        return pin.computeCollisions(self.model, self.data, self.collision_model, self.collision_data, q, True)
    
    def update_point_cloud(self, point_cloud):
#         printPink("UPDATING POINT CLOUD")

        if self.sensor_collision:

            # add pcd to fcl
            fcl_octree = hppfcl.makeOctree(point_cloud, .01)
            # add fcl_pcd to pinocchio
            octree_obj = pin.GeometryObject(SENSOR_COLLISION_GEOM_NAME, 0, fcl_octree, pin.SE3.Identity())
            octree_obj.meshColor[0] = 1.0

            pcl_id = self.collision_model.getGeometryId(SENSOR_COLLISION_GEOM_NAME)

            self.collision_model.geometryObjects[pcl_id] = octree_obj

            self.create_data()

        if self.visualize:

            pcl_bvh = hppfcl.BVHModelOBBRSS()
            pcl_bvh.beginModel(0, len(point_cloud))
            pcl_bvh.addVertices(point_cloud)
            bvh_obj = pin.GeometryObject(SENSOR_VISUAL_GEOM_NAME, 0, pcl_bvh, pin.SE3.Identity())
            bvh_obj.meshColor[0] = 1.0

            pcl_viz_id = self.visual_model.getGeometryId(SENSOR_VISUAL_GEOM_NAME)
            if pcl_viz_id == len(self.visual_model.geometryObjects):
                self.visual_model.addGeometryObject(bvh_obj)
            else:
                self.visual_model.geometryObjects[pcl_viz_id] = bvh_obj
            
            self.viz.rebuildData()
            self.viz.loadViewerModel()
            self.viz.display(self.q)

    def gripper_collision(self, gripper_pose, static_only, s_height, voxels, point_cloud=None):

        if IS_EE_OFFSET:
            gripper_pose @= EE_OFFSET
            
        quat_wxyz = tf.quaternion_from_matrix(gripper_pose)

        quat_xyzw = np.roll(quat_wxyz, -1)

        q = np.concatenate([gripper_pose[:3, 3], quat_xyzw, [0]*6])

        res = self.hppfcl(q)
        
        return res
    
    def add_box(self, name, pose, size, color=np.ones(4), collision=True):
        placement = pin.SE3(pin.Quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w), 
                            np.array([pose.position.x, pose.position.y, pose.position.z]), 
                            )
        box = pin.GeometryObject(name, 0, hppfcl.Box(size), placement)

        box.meshColor = color
        
        self.collision_model.addGeometryObject(box)
        if self.visualize:
             self.visual_model.addGeometryObject(box)

        # Add collision pairs: (any gripper geom, box geom)
        if collision:
            for gripper_geom in self.gripper_geoms:
                self.collision_model.addCollisionPair(pin.CollisionPair(self.collision_model.getGeometryId(gripper_geom),
                                                                self.collision_model.getGeometryId(name))) 
    
    def add_shelf(self, collision=STATIC_COLLISION):
#        if self.is_sim and not rospy.has_param('/workspace/pose'):
#        if not rospy.has_param('/workspace/pose'):
        if False:
            self.add_box(
                'table',
                conv.list_to_pose_stamped([0.9, 0.0, 0.4, 0, 0, 0], 'world'),
                (0.58, 1, 1),
                collision=collision,
            )
        else:
            pose = rospy.get_param('/workspace/pose', [0.95, 0.63, 0.95])
            size = rospy.get_param('/workspace/size', [0.53, 1.38, 0.53])
            padding = 0.1
            thick = 0.05
            size_top = [size[0], size[1] + 5 * thick, thick]
            size_bottom = [size[0], size[1] + 2 * thick, pose[2] + thick]
            size_left = [size[0], thick, size[2] + pose[2] + size_top[2]]
            size_right = [size[0], thick, size[2] + pose[2] + size_top[2]]
            size_back = [thick, size[1], size[2]]
            pose_top = [
                pose[0] + size[0] - size_top[0] / 2,
                pose[1] - size[1] / 2,
                pose[2] + size[2] + 0.5 * size_top[2],
            ]
            pose_bottom = [
                pose[0] + size_bottom[0] / 2,
                pose[1] - size[1] / 2,
                pose[2] - 0.5 * size_bottom[2],
            ]
            pose_left = [
                pose[0] + size_left[0] / 2,
                pose[1] + size_left[1] / 2,
                size_left[2] / 2,
            ]
            pose_right = [
                pose[0] + size_right[0] / 2,
                pose[1] - size[1] - size_right[1] / 2,
                size_right[2] / 2,
            ]
            pose_back = [
                pose[0] + size[0] + size_back[0] / 2,
                pose[1] - size[1] / 2,
                pose[2] + size[2] / 2,
            ]
            self.add_box(
                'shelf_top',
                conv.list_to_pose([*pose_top, 0, 0, 0]),
                np.add(size_top, [padding, padding, padding / 2]),
                color=np.array([0, 1, 0, 1]),
                collision=collision,
            )
            self.add_box(
                'shelf_bottom',
                conv.list_to_pose([*pose_bottom, 0, 0, 0]),
                np.add(size_bottom, [padding, padding, 0]),
                color=np.array([0, 1, 0, 1]),
                collision=collision,
            )
            self.add_box(
                'shelf_left',
                conv.list_to_pose([*pose_left, 0, 0, 0]),
                np.add(size_left, padding),
                color=np.array([0, 1, 0, 1]),
                collision=collision,
            )
            self.add_box(
                'shelf_right',
                conv.list_to_pose([*pose_right, 0, 0, 0]),
                np.add(size_right, padding),
                color=np.array([0, 1, 0, 1]),
                collision=collision,
            )
            self.add_box(
                'shelf_back',
                conv.list_to_pose([*pose_back, 0, 0, 0]),
                np.add(size_back, padding),
                color=np.array([0, 1, 0, 1]),
                collision=collision,
            )

    def add_octree(self, name=SENSOR_COLLISION_GEOM_NAME, point_cloud=np.empty((0, 3)), collision=SENSOR_COLLISION):
        fcl_octree = hppfcl.makeOctree(point_cloud, 0.01)
        # add fcl_pcd to pinocchio
        octree_obj = pin.GeometryObject(name, 0, fcl_octree, pin.SE3.Identity())
        octree_obj.meshColor[0] = 1.0

        self.collision_model.addGeometryObject(octree_obj)

        if collision:
            for gripper_geom in self.gripper_geoms:
                self.collision_model.addCollisionPair(pin.CollisionPair(self.collision_model.getGeometryId(gripper_geom),
                                                                self.collision_model.getGeometryId(name))) 
                
        if self.visualize:
            pcl_bvh = hppfcl.BVHModelOBBRSS()
            pcl_bvh.beginModel(0, len(point_cloud))
            pcl_bvh.addVertices(point_cloud)
            bvh_obj = pin.GeometryObject(SENSOR_VISUAL_GEOM_NAME, 0, pcl_bvh, pin.SE3.Identity())
            bvh_obj.meshColor[0] = 1.0
            
            self.visual_model.addGeometryObject(bvh_obj)

# TODO
# test tracik - hppfcl
# test bio-ik - hppfcl