import sys
import time
import copy

import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from packaging.version import Version
from importlib_metadata import version

import transformations as tf
from tracikpy import TracIKSolver

import rospy
import moveit_commander
from moveit_commander import conversions as conv

MOVEIT_CMDR_VERSION = Version(version('moveit_commander'))

from std_srvs.srv import Empty
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pcl2
from moveit_msgs.srv import GetPlanningScene
from shape_msgs.msg import Mesh, MeshTriangle
from sensor_msgs.msg import JointState, PointCloud2
from geometry_msgs.msg import Pose, PoseStamped, Point
from moveit_msgs.msg import PlanningScene, PlanningSceneComponents
from moveit_msgs.msg import RobotState, CollisionObject, AttachedCollisionObject

from utils.conversions import pose_to_matrix, matrix_to_pose
from utils.visual_utils import decode_seg_img_rgb
from motion_planner.motion_planner import MotionPlanner
from lab_vbnpm.srv import GetObjectModels, GetObjectModelsResponse
from lab_vbnpm.srv import GetObjectOcclusions, GetObjectOcclusionsResponse


class MoveitPlanner(MotionPlanner):

    def __init__(
        self,
        robot_file,
        end_effector_links,
        move_groups,
        commander_args=[],
        is_sim=True,
    ):
        super().__init__(robot_file, end_effector_links)
        moveit_commander.roscpp_initialize(commander_args)
        # self.robot = moveit_commander.RobotCommander()
        self.move_groups = {}
        for group in move_groups:
            self.move_groups[group] = moveit_commander.MoveGroupCommander(group)
            self.move_groups[group].set_max_acceleration_scaling_factor(1)
            self.move_groups[group].set_max_velocity_scaling_factor(1)
            self.move_groups[group].set_num_planning_attempts(5)
            self.move_groups[group].set_planning_time(20.0)
            # self.move_groups[group].set_planner_id("LazyPRMLoad")
            # self.move_groups[group].set_planner_id("SBL")
            self.move_groups[group].set_planner_id("RRTConnect")
            # self.move_groups[group].set_planner_id("LazyPRMSemiPers")
            # self.move_groups[group].set_planner_id("PRM")

        self.octomap_pub = rospy.Publisher(
            '/octomap/points', PointCloud2, queue_size=3, latch=True
        )

        self.scene = moveit_commander.PlanningSceneInterface()
        self.is_sim = is_sim
        self.reset()

    @staticmethod
    def get_octomap_data():
        rospy.wait_for_service('/get_planning_scene')
        try:
            get_scene = rospy.ServiceProxy(
                '/get_planning_scene', GetPlanningScene
            )
            resp = get_scene(
                PlanningSceneComponents(PlanningSceneComponents.OCTOMAP)
            )
            return resp.scene.world.octomap.octomap.data
        except rospy.ServiceException as e:
            print('Service call failed: ', e, file=sys.stderr)
            return None

    def pub_octomap_till_recieved(self, msg=None):
        should_exist = msg is not None
        octomap_exists = not should_exist
        # print('waiting for octomap...', file=sys.stderr)
        while octomap_exists != should_exist:
            if msg:
                self.octomap_pub.publish(msg)
            try:
                octomap_exists = bool(self.get_octomap_data())
            except rospy.ROSException:
                octomap_exists = False
            rospy.sleep(0.01)
        # print('octomap exists:', octomap_exists, file=sys.stderr)

    def reset(self):
        # TODO load scene from xml
        self.scene.remove_attached_object()
        self.scene.remove_world_object()
        rospy.wait_for_service('clear_octomap')
        try:
            clear_srv = rospy.ServiceProxy('clear_octomap', Empty)
            clear_srv()
            self.pub_octomap_till_recieved()
        except rospy.ServiceException as e:
            print("Service call failed: ", e, file=sys.stderr)

        if self.is_sim and not rospy.has_param('/workspace/pose'):
            self.scene.add_box(
                'table',
                conv.list_to_pose_stamped([0.9, 0.0, 0.4, 0, 0, 0], 'world'),
                (0.58, 1, 1),
            )
        else:
            pose = rospy.get_param("/workspace/pose", [0.55, 0.655, 1.05])
            size = rospy.get_param("/workspace/size", [0.4, 1.31, 0.52])
            padding = 0.04
            size_top = [size[0], size[1] + 0.2, 0.1]
            size_bottom = [size[0], size[1] + 0.2, pose[2]]
            size_left = [size[0], 0.1, size[2]]
            size_right = [size[0], 0.1, size[2]]
            size_back = [0.05, size[1], size[2]]
            pose_top = [
                pose[0] + 0.5 * size_top[0],
                pose[1] - 0.5 * size[1],
                pose[2] + size[2] + 0.5 * size_top[2],
            ]
            pose_bottom = [
                pose[0] + 0.5 * size_bottom[0],
                pose[1] - 0.5 * size[1],
                pose[2] - 0.5 * size_bottom[2],
            ]
            pose_left = [
                pose[0] + 0.5 * size_left[0],
                pose[1] + 0.5 * size_left[1],
                pose[2] + 0.5 * size_left[2],
            ]
            pose_right = [
                pose[0] + 0.5 * size_right[0],
                pose[1] - size[1] - 0.5 * size_right[1],
                pose[2] + 0.5 * size_right[2],
            ]
            pose_back = [
                pose[0] + size[0] + 0.5 * size_back[0],
                pose[1] - 0.5 * size[1],
                pose[2] + 0.5 * size[2],
            ]
            # self.scene.add_box(
            #     'shelf_top',
            #     conv.list_to_pose_stamped([*pose_top, 0, 0, 0], 'world'),
            #     np.add(size_top, [padding, padding, padding / 2]),
            # )
            self.scene.add_box(
                'shelf_bottom',
                conv.list_to_pose_stamped([*pose_bottom, 0, 0, 0], 'world'),
                np.add(size_bottom, [padding, padding, 0]),
            )
            # self.scene.add_box(
            #     'shelf_left',
            #     conv.list_to_pose_stamped([*pose_left, 0, 0, 0], 'world'),
            #     np.add(size_left, padding),
            # )
            # self.scene.add_box(
            #     'shelf_right',
            #     conv.list_to_pose_stamped([*pose_right, 0, 0, 0], 'world'),
            #     np.add(size_right, padding),
            # )
            # self.scene.add_box(
            #     'shelf_back',
            #     conv.list_to_pose_stamped([*pose_back, 0, 0, 0], 'world'),
            #     np.add(size_back, padding),
            # )

    def reset_octomap(self):
        # TODO load scene from xml
        rospy.wait_for_service('clear_octomap')
        try:
            clear_srv = rospy.ServiceProxy('clear_octomap', Empty)
            clear_srv()
            self.pub_octomap_till_recieved()
        except rospy.ServiceException as e:
            print("Service call failed: ", e, file=sys.stderr)

    def add_mesh_to_scene(self, mesh, name, pose_mat=np.eye(4), frame='world'):
        # init collision object
        co = CollisionObject()
        co.operation = CollisionObject.ADD
        co.id = f'{name}'
        co.header.frame_id = frame
        co.pose = matrix_to_pose(pose_mat)

        # set faces and vertices
        assert (mesh.extents is not None)
        mesh_msg = Mesh()
        # first_face = mesh.faces[0]
        for face in mesh.faces:
            triangle = MeshTriangle()
            triangle.vertex_indices = [face[0], face[1], face[2]]
            mesh_msg.triangles.append(triangle)
        for vertex in mesh.vertices:
            point = Point()
            point.x = vertex[0]
            point.y = vertex[1]
            point.z = vertex[2]
            mesh_msg.vertices.append(point)
        co.meshes = [mesh_msg]

        # add to scene
        self.scene.add_object(co)

    def set_planning_scene(
        self,
        points,
        target_mesh=None,
        frame_id='world',
        colors=None,
        filter_outliers=False,
    ):
        # estimate target geometry and add to planning scene
        if target_mesh is not None:
            self.add_mesh_to_scene(target_mesh, 'target', frame=frame_id)
            surface, find = trimesh.sample.sample_surface_even(
                target_mesh, 10000
            )
            kdtree = KDTree(surface[:, :2])
            dist, ind = kdtree.query(points[:, :2])
            points = points[dist > 0.05]
        else:
            for obj_name in self.scene.get_known_object_names():
                if obj_name.startswith('target'):
                    self.scene.remove_world_object(obj_name)

        # push static scene to octomap server
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcl.colors = o3d.utility.Vector3dVector(colors)
        if filter_outliers:
            pcl, _removed = pcl.remove_statistical_outlier(*filter_outliers)
        header = Header()
        header.frame_id = frame_id
        msg = pcl2.create_cloud_xyz32(header, pcl.points)
        self.pub_octomap_till_recieved(msg)

    def make_robot_state_msg(
        self,
        joint_state,
        attach_objects=[],
        grasp_state=None,
        ee=None,
        ee_links=[],
        is_diff=False,
    ):
        robot_state = RobotState()
        robot_state.joint_state = joint_state
        robot_state.is_diff = is_diff

        self.remove_attached_objects()
        for obj_id in attach_objects:
            aco = self.attach_object(
                str(obj_id),
                grasp_state,
                joint_state,
                ee,
                ee_links,
            )
            robot_state.attached_collision_objects.append(aco)

        return robot_state

    def remove_attached_objects(self):
        self.scene.remove_attached_object()
        # for obj_name in self.scene.get_known_object_names():
        #     if obj_name.startswith('ATTACHED_'):
        #         self.scene.remove_world_object(obj_name)

    def attach_object(self, obj_name, grasp_jnt, current_jnt, ee, ee_links):
        ik = self.ik_for_ees[ee]

        g_joint_indices = list(map(grasp_jnt.name.index, ik.joint_names))
        grasp_joint_values = np.array(grasp_jnt.position)[g_joint_indices]
        grasp_pose = ik.fk(grasp_joint_values)

        c_joint_indices = list(map(current_jnt.name.index, ik.joint_names))
        current_joint_values = np.array(current_jnt.position)[c_joint_indices]
        current_pose = ik.fk(current_joint_values)

        obj_pose = pose_to_matrix(
            self.scene.get_object_poses([obj_name])[obj_name]
        )
        obj_pose_rel = np.linalg.inv(grasp_pose) @ obj_pose
        obj_abs_pose = current_pose @ obj_pose_rel
        pose = matrix_to_pose(obj_abs_pose)

        co = copy.deepcopy(self.scene.get_objects([obj_name])[obj_name])
        co.id = obj_name
        co.pose = pose
        aco = AttachedCollisionObject()
        aco.object = co
        aco.link_name = ee
        aco.touch_links = ee_links
        # aco.detach_posture = trajectory_msgs/JointTrajectory
        # aco.weight = 0
        # self.scene.attach_object(aco)
        self.scene.attach_mesh('base_link', obj_name)
        return aco

    def fill_joint_trajectory_from_state(self, joint_trajectory, joint_state):
        remainder = set(joint_state.name)
        remainder = remainder.difference(joint_trajectory.joint_names)
        if remainder:
            joint_indices = list(map(joint_state.name.index, remainder))
            joint_trajectory.joint_names += list(remainder)
            positions = tuple(np.array(joint_state.position)[joint_indices])
            velocities = (0.0, ) * len(remainder)
            for i in range(len(joint_trajectory.points)):
                joint_trajectory.points[i].positions += positions
                joint_trajectory.points[i].velocities += velocities

    def joint_motion_plan(
        self,
        start_jnt,
        goal_jnt,
        move_group,
        attach_objects=[],
        grasp_state=None,
        ee=None,
        ee_links=[],
        is_diff=False
    ):
        mg = self.move_groups[move_group]
        r_start_state = self.make_robot_state_msg(
            start_jnt, attach_objects, grasp_state, ee, ee_links, is_diff
        )
        mg.set_start_state(r_start_state)
        mg.set_joint_value_target(goal_jnt)
        success, robot_trajectory, planning_time, error_code = mg.plan()
        joint_trajectory = robot_trajectory.joint_trajectory
        self.fill_joint_trajectory_from_state(joint_trajectory, start_jnt)
        return joint_trajectory if success else error_code

    def pose_motion_plan(
        self,
        start_jnt,
        goal,  # pose as list, pose msg, or list of either
        move_group,
        constraints=[],
        attach_objects=[],
        grasp_state=None,
        ee=None,
        ee_links=[],
        is_diff=False
    ):
        mg = self.move_groups[move_group]
        r_start_state = self.make_robot_state_msg(
            start_jnt, attach_objects, grasp_state, ee, ee_links, is_diff
        )
        mg.set_start_state(r_start_state)
        if type(goal) is list \
                and len(goal) > 0 \
                and type(goal[0]) in (list, Pose):
            mg.set_pose_targets(goal)
        else:
            if type(goal) is list and len(goal) == 3:
                mg.set_position_target(goal)
            else:
                mg.set_pose_target(goal)
        success, robot_trajectory, planning_time, error_code = mg.plan()
        joint_trajectory = robot_trajectory.joint_trajectory
        self.fill_joint_trajectory_from_state(joint_trajectory, start_jnt)
        return joint_trajectory if success else error_code

    def cartesian_motion(
        self,
        start_jnt,
        xyz,
        move_group,
        ee,
        xyz_is_relative=True,
        eef_step=0.005,
        jump_threshold=5.0,
        avoid_collisions=True,
        attach_objects=[],
        grasp_state=None,
        ee_links=[],
        is_diff=False
    ):
        ik = self.ik_for_ees[ee]
        mg = self.move_groups[move_group]

        joint_indices = list(map(start_jnt.name.index, ik.joint_names))
        start_joint_values = np.array(start_jnt.position)[joint_indices]
        start_pose = ik.fk(start_joint_values)

        transform = tf.translation_matrix(xyz)
        if xyz_is_relative:
            goal_pose = tf.concatenate_matrices(start_pose, transform)
        else:
            goal_pose = tf.concatenate_matrices(transform, start_pose)
        pose_list = tf.translation_from_matrix(goal_pose).tolist()
        pose_list += tf.euler_from_matrix(goal_pose)
        pose_msg = conv.list_to_pose(pose_list)
        waypoints = [copy.deepcopy(pose_msg)]

        r_start_state = self.make_robot_state_msg(
            start_jnt, attach_objects, grasp_state, ee, ee_links, is_diff
        )
        mg.set_start_state(r_start_state)

        if MOVEIT_CMDR_VERSION >= Version('1.1.16'):
            robot_trajectory, fraction = mg.compute_cartesian_path(
                waypoints, eef_step, avoid_collisions
            )
        else:
            robot_trajectory, fraction = mg.compute_cartesian_path(
                waypoints, eef_step, jump_threshold, avoid_collisions
            )
        joint_trajectory = robot_trajectory.joint_trajectory
        self.fill_joint_trajectory_from_state(joint_trajectory, start_jnt)
        return fraction, joint_trajectory
