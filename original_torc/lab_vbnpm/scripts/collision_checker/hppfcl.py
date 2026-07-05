import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

import hppfcl

from collision_checker.HPPFCL_CONSTANTS import *
from collision_checker.collision_checker import *

import transformations as tf
import numpy as np

from moveit_commander import conversions as conv
import rospkg
import rospy

class HPPFCL(CollisionChecker):
    def __init__(self, urdf_path, 
                 static_collision=STATIC_COLLISION_DEFAULT, 
                 sensor_collision=SENSOR_COLLISION_DEFAULT, visualize=False):
        self.urdf_path = urdf_path
        self.visualize = visualize

        self.robot_geoms = []
        self.static_geoms = []
        self.sensor_geoms = []

        self.set_package_dirs()
        self.reset()

        # Add shelf to pinocchio
        self.static_collision = static_collision
        self.add_shelf(collision=static_collision)

        # Add empty octree object to pinocchio
        self.sensor_collision = sensor_collision
        self.add_octree(collision=sensor_collision)
        
        if visualize:
            self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
            self.viz.initViewer(open=True)

            self.viz.loadViewerModel()

        self.q = pin.neutral(self.model)

        self.create_data()

    def set_package_dirs(self,):
        rp = rospkg.RosPack()
        motoman_pkg_path = rp.get_path('motoman')
        lab_pkg_path = rp.get_path('lab_vbnpm')

        package_dirs = []
        package_dirs.append(motoman_pkg_path)
        package_dirs.append(motoman_pkg_path + "/robotiq")

        self.package_dirs = package_dirs
    
    def reset(self, ):
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(self.urdf_path, package_dirs=self.package_dirs)
        
        for geom in self.collision_model.geometryObjects:
            self.robot_geoms.append(geom.name)

    def create_data(self, ):
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()

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
            self.static_geoms.append(name)
            for robot_geom in self.robot_geoms:
                self.collision_model.addCollisionPair(pin.CollisionPair(self.collision_model.getGeometryId(robot_geom),
                                                                self.collision_model.getGeometryId(name))) 
    
    def add_shelf(self, collision=STATIC_COLLISION_DEFAULT):
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

    def add_octree(self, name=SENSOR_COLLISION_GEOM_NAME, point_cloud=np.empty((0, 3)), collision=SENSOR_COLLISION_DEFAULT):
        fcl_octree = hppfcl.makeOctree(point_cloud, 0.01)
        # add fcl_pcd to pinocchio
        octree_obj = pin.GeometryObject(name, 0, fcl_octree, pin.SE3.Identity())
        octree_obj.meshColor[0] = 1.0

        self.collision_model.addGeometryObject(octree_obj)

        if collision:
            self.sensor_geoms.append(name)
            for robot_geom in self.robot_geoms:
                self.collision_model.addCollisionPair(pin.CollisionPair(self.collision_model.getGeometryId(robot_geom),
                                                                self.collision_model.getGeometryId(name))) 
                
        if self.visualize:
            pcl_bvh = hppfcl.BVHModelOBBRSS()
            pcl_bvh.beginModel(0, len(point_cloud))
            pcl_bvh.addVertices(point_cloud)
            bvh_obj = pin.GeometryObject(SENSOR_VISUAL_GEOM_NAME, 0, pcl_bvh, pin.SE3.Identity())
            bvh_obj.meshColor[0] = 1.0
            
            self.visual_model.addGeometryObject(bvh_obj)

    def update_perception(self, point_cloud):
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

    def query(self, q):

        if self.visualize:
            self.q = q
            self.viz.display(q)

#        pin.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data, q)
        return pin.computeCollisions(self.model, self.data, self.collision_model, self.collision_data, q, True)
