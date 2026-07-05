"""
implement the visibiltiy constraint in MoveIt for motion planning
visualize the target object pose
"""
import sys
import os
import rospy
import moveit_commander
from rospkg import RosPack
from interactive_markers.interactive_marker_server import *
import tf.transformations
from visualization_msgs.msg import *
from geometry_msgs.msg import Pose, PoseStamped
import tf
import multiprocessing

from moveit_msgs.msg import PlanningScene
from shape_msgs.msg import Mesh, MeshTriangle
from sensor_msgs.msg import JointState, PointCloud2
from geometry_msgs.msg import Pose, PoseStamped, Point
from moveit_msgs.msg import RobotState, CollisionObject, AttachedCollisionObject

from moveit_msgs.msg import (
    MoveItErrorCodes,
    TrajectoryConstraints,
    PlannerInterfaceDescription,
    MotionPlanRequest,
)
from moveit_msgs.msg import (
    RobotTrajectory,
    Grasp,
    PlaceLocation,
    Constraints,
    RobotState,
    VisibilityConstraint,
    OrientationConstraint
)
import numpy as np
import transformations as tfm

class GoalInteractiveMarkerServer:
    def __init__(self):
        self.lock = multiprocessing.Lock()
        self.pose_pub = rospy.Publisher('goal_pose', PoseStamped, queue_size=20)
        pose = Pose()
        pose.position.x = 1.0
        pose.position.z = 1.0
        pose.orientation.w = 1.0
        pose.orientation.x = 0.7071068
        pose.orientation.y = 0.0
        pose.orientation.z = 0.7071068
        pose.orientation.w = 0.0


        self.pose = pose
        rospy.sleep(1.0)
        self.br = tf.TransformBroadcaster()
        print("first publish, pose: ")
        print(self.pose)
        self.br.sendTransform([self.pose.position.x,self.pose.position.y,self.pose.position.z], 
                              [self.pose.orientation.x,self.pose.orientation.y,self.pose.orientation.z,self.pose.orientation.w], 
                              rospy.Time.now(), "goal_pose", "base_link")  
        server = InteractiveMarkerServer("goal_marker")

        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.name = "goal_marker"
        int_marker.pose = pose


        box_marker = Marker()
        box_marker.type = Marker.SPHERE
        box_marker.scale.x = 0.1
        box_marker.scale.y = 0.1
        box_marker.scale.z = 0.1
        box_marker.color.r = 1.0
        box_marker.color.g = 0.0
        box_marker.color.b = 0.0
        box_marker.color.a = 1.0

        box_control = InteractiveMarkerControl()
        box_control.always_visible = True
        box_control.name = "move_pose"
        box_control.markers.append(box_marker)
        int_marker.controls.append(box_control)

        # move_control = InteractiveMarkerControl()
        # move_control.name = "move"
        # move_control.interaction_mode = InteractiveMarkerControl.MOVE_3D
        # # add the control to the interactive marker
        # int_marker.controls.append(move_control)

        # rotate_control = InteractiveMarkerControl()
        # rotate_control.name = "rotate"
        # rotate_control.interaction_mode = InteractiveMarkerControl.ROTATE_3D

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.name = "rotate_x"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.name = "rotate_z"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.name = "rotate_y"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)



        # add the control to the interactive marker
        # int_marker.controls.append(rotate_control)

        server.insert(int_marker, self.processFeedback)
        server.applyChanges()
        self.server = server
        self.timer = rospy.timer.Timer(rospy.Duration(0.1), self.timer_callback)

    def processFeedback(self, feedback):
        # self.lock.acquire()
        self.pose = feedback.pose
        # self.lock.release()
        # pose = PoseStamped()
        # pose.header.frame_id = "base_link"
        # pose.header.stamp = rospy.Time.now()
        # pose.pose.position = feedback.pose.position
        # pose.pose.orientation = feedback.pose.orientation
        # self.pose_pub.publish(pose)
        # self.br.sendTransform([self.pose.position.x,self.pose.position.y,self.pose.position.z], 
        #                       [self.pose.orientation.x,self.pose.orientation.y,self.pose.orientation.z,self.pose.orientation.w], 
        #                       rospy.Time.now(), "goal_pose", "base_link")  
    def timer_callback(self, event):
        self.lock.acquire()
        # pose = PoseStamped()
        # pose.header.frame_id = "base_link"
        # pose.header.stamp = rospy.Time.now()
        # pose.pose = self.pose
        # pose.pose = self.pose
        # self.pose_pub.publish(pose)
        self.br.sendTransform([self.pose.position.x,self.pose.position.y,self.pose.position.z], 
                              [self.pose.orientation.x,self.pose.orientation.y,self.pose.orientation.z,self.pose.orientation.w], 
                              rospy.Time.now(), "goal_pose", "base_link")
        self.lock.release()


class MotomanSDA10F:
    def __init__(self):

        self.ignore_collision_ee_links = [
            "motoman_right_ee",
            "left_outer_knuckle",
            "left_outer_finger",
            "left_inner_finger",
            "left_inner_finger_pad",
            "left_inner_knuckle",
            "right_outer_knuckle",
            "right_outer_finger",
            "right_inner_finger",
            "right_inner_finger_pad",
            "right_inner_knuckle",
            "robotiq_arg2f_extra_link",
            "robotiq_arg2f_base_link",
            "arm_right_link_7_t",
            "arm_right_link_6_b",
        ]

    def init_motion_planner(self):
        rp = RosPack()
        robot_urdf = rp.get_path(
            'motoman_sda10f_moveit_config'
        ) + '/config/gazebo_motoman_sda10f.urdf'
        planner = MoveitPlanner(
            robot_urdf,
            ['motoman_left_ee', 'motoman_right_ee'],
            ['arm_right', 'arm_left'],
        )
        return planner


class MoveitPlanner:
    def __init__(
        self,
        robot_file,
        end_effector_links,
        move_groups,
        commander_args=[],
    ):
        moveit_commander.roscpp_initialize(commander_args)
        # self.robot = moveit_commander.RobotCommander()
        self.move_groups = {}
        for group in move_groups:
            self.move_groups[group] = moveit_commander.MoveGroupCommander(group)
            self.move_groups[group].set_num_planning_attempts(1)
            self.move_groups[group].set_planning_time(30.0)
            # self.move_groups[group].set_planner_id("LazyPRMLoad")
            # self.move_groups[group].set_planner_id("SBL")
            self.move_groups[group].set_planner_id("RRTConnect")
        self.scene = moveit_commander.PlanningSceneInterface()
        self.listener = tf.TransformListener()


    def set_trajectory_constraints(self, move_group):
        mg = self.move_groups[move_group]
        tc = TrajectoryConstraints()
        constraints = []
        cons = Constraints()
        cons.name = "visibility_constraints"
        vc = VisibilityConstraint()

        vc.target_radius = 0.01  # the cone radius at the target
        (trans,rot) = self.listener.lookupTransform('/goal_pose', 'base_link', rospy.Time(0))
        target_pose = PoseStamped()
        target_pose.header.frame_id = 'base_link'
        target_pose.pose.position.x = trans[0]
        target_pose.pose.position.y = trans[1]
        target_pose.pose.position.z = trans[2]
        target_pose.pose.orientation.x = rot[0]
        target_pose.pose.orientation.y = rot[1]
        target_pose.pose.orientation.z = rot[2]
        target_pose.pose.orientation.w = rot[3]

        vc.target_pose = target_pose
        vc.cone_sides = 4

        # obtain the current sensor pose
        (trans,rot) = self.listener.lookupTransform('/camera_arm_link', 'base_link', rospy.Time(0))
        sensor_pose = PoseStamped()
        sensor_pose.header.frame_id = 'base_link'
        sensor_pose.pose.position.x = trans[0]
        sensor_pose.pose.position.y = trans[1]
        sensor_pose.pose.position.z = trans[2]
        sensor_pose.pose.orientation.x = rot[0]
        sensor_pose.pose.orientation.y = rot[1]
        sensor_pose.pose.orientation.z = rot[2]
        sensor_pose.pose.orientation.w = rot[3]

        vc.sensor_pose = sensor_pose

        vc.max_view_angle = 10*np.pi/180
        vc.max_range_angle = 2*np.pi/180
        vc.sensor_view_direction = vc.SENSOR_Y
        vc.weight = 1.0
        cons.visibility_constraints.append(vc)

        # oc = OrientationConstraint()
        # oc.link_name = "motoman_right_ee"
        # oc.orientation.x = 0.7071068
        # oc.orientation.y = 0.0
        # oc.orientation.z = 0.7071068
        # oc.orientation.w = 0.0
        # oc.absolute_z_axis_tolerance = 5/180*np.pi
        # cons.orientation_constraints.append(oc)

        constraints.append(cons)
        tc.constraints = constraints
        mg.set_trajectory_constraints(tc)


    def set_path_constraints(self, move_group):
        mg = self.move_groups[move_group]
        cons = Constraints()
        cons.name = "visibility_constraints"
        vc = VisibilityConstraint()

        # the target is fixed relative to the base_link
        vc.target_radius = 0.01  # the cone radius at the target
        (trans,rot) = self.listener.lookupTransform('base_link', '/goal_pose', rospy.Time(0))
        target_pose = PoseStamped()
        target_pose.header.frame_id = 'base_link'
        target_pose.pose.position.x = trans[0]
        target_pose.pose.position.y = trans[1]
        target_pose.pose.position.z = trans[2]
        target_pose.pose.orientation.x = rot[0]
        target_pose.pose.orientation.y = rot[1]
        target_pose.pose.orientation.z = rot[2]
        target_pose.pose.orientation.w = rot[3]

        vc.target_pose = target_pose
        vc.cone_sides = 10

        # obtain the current sensor pose
        # (trans,rot) = self.listener.lookupTransform('base_link', '/camera_arm_link', rospy.Time(0))
        #****************************************************
        # notice that the ee link is not fixed relative to the base line
        # it is a mobile link
        # (trans,rot) = self.listener.lookupTransform('base_link', '/motoman_right_ee', rospy.Time(0))
        # sensor_pose = PoseStamped()
        # sensor_pose.header.frame_id = 'base_link'
        # sensor_pose.pose.orientation.x = rot[0]
        # sensor_pose.pose.orientation.y = rot[1]
        # sensor_pose.pose.orientation.z = rot[2]
        # sensor_pose.pose.orientation.w = rot[3]

        # # trans_np = np.array(trans)
        # # rot_mat = tfm.quaternion_matrix([rot[3],rot[0],rot[1],rot[2]])
        # # trans_np += 0.05*rot_mat[:3,2]  # z axis: 2
        

        # sensor_pose.pose.position.x = trans[0]
        # sensor_pose.pose.position.y = trans[1]
        # sensor_pose.pose.position.z = trans[2]

        # (trans,rot) = self.listener.lookupTransform('base_link', '/motoman_right_ee', rospy.Time(0))
        sensor_pose = PoseStamped()
        sensor_pose.header.frame_id = 'motoman_right_ee'
        sensor_pose.pose.orientation.x = 0
        sensor_pose.pose.orientation.y = 0
        sensor_pose.pose.orientation.z = 0
        sensor_pose.pose.orientation.w = 1

        # trans_np = np.array(trans)
        # rot_mat = tfm.quaternion_matrix([rot[3],rot[0],rot[1],rot[2]])
        # trans_np += 0.05*rot_mat[:3,2]  # z axis: 2
        

        sensor_pose.pose.position.x = 0
        sensor_pose.pose.position.y = 0
        sensor_pose.pose.position.z = 0



        vc.sensor_pose = sensor_pose

        # vc.max_view_angle = 30*np.pi/180
        vc.max_range_angle = 15*np.pi/180
        # vc.sensor_view_direction = vc.SENSOR_Y
        vc.sensor_view_direction = vc.SENSOR_Z
        vc.weight = 1.0
        cons.visibility_constraints.append(vc)

        # oc = OrientationConstraint()
        # oc.link_name = "motoman_right_ee"
        # oc.orientation.x = 0.7071068
        # oc.orientation.y = 0.0
        # oc.orientation.z = 0.7071068
        # oc.orientation.w = 0.0
        # oc.absolute_z_axis_tolerance = 5/180*np.pi
        # cons.orientation_constraints.append(oc)
        mg.set_path_constraints(cons)


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

    def joint_state_from_joint_dict(self, joint_dict, move_group):

        start_jnt = planner.move_groups[move_group].get_current_state().joint_state
        names = start_jnt.name
        position = list(start_jnt.position)
        for i in range(len(names)):
            if names[i] in joint_dict:
                position[i] = joint_dict[names[i]]
        start_jnt.position = tuple(position)
        return start_jnt


    def joint_motion_plan(
        self,
        start_jnt,
        goal_jnt,
        move_group,
        ee=None,
        ee_links=[],
        is_diff=False
    ):
        mg = self.move_groups[move_group]
        r_start_state = self.make_robot_state_msg(
            start_jnt, ee, ee_links, is_diff
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
        ee=None,
        ee_links=[],
        is_diff=False
    ):
        mg = self.move_groups[move_group]
        r_start_state = self.make_robot_state_msg(
            start_jnt, ee, ee_links, is_diff
        )
        mg.set_start_state(r_start_state)
        if type(goal) is list \
                and len(goal) > 0 \
                and type(goal[0]) in (list, Pose):
            mg.set_pose_targets(goal)
        else:
            mg.set_pose_target(goal)
        success, robot_trajectory, planning_time, error_code = mg.plan()
        joint_trajectory = robot_trajectory.joint_trajectory
        self.fill_joint_trajectory_from_state(joint_trajectory, start_jnt)
        return joint_trajectory if success else error_code
    
    def make_robot_state_msg(
        self,
        joint_state,
        ee=None,
        ee_links=[],
        is_diff=False,
    ):
        robot_state = RobotState()
        robot_state.joint_state = joint_state
        robot_state.is_diff = is_diff
        return robot_state


    def reset_robot(self):
        joint_goal = {
            "torso_joint_b1": 0,
            # "arm_left_joint_1_s": 1.75,
            # "arm_left_joint_2_l": 0.8,
            # "arm_left_joint_3_e": 0,
            # "arm_left_joint_4_u": -0.66,
            # "arm_left_joint_5_r": 0,
            # "arm_left_joint_6_b": 0,
            # "arm_left_joint_7_t": 0,
            # "arm_right_joint_1_s": 1.75,
            # "arm_right_joint_2_l": 0.8,
            # "arm_right_joint_3_e": 0.0,
            # "arm_right_joint_4_u": -0.66,
            # "arm_right_joint_5_r": 0,
            # "arm_right_joint_6_b": 0,
            # "arm_right_joint_7_t": 0
            "arm_right_joint_1_s": -0.2,
            "arm_right_joint_2_l": 0,
            "arm_right_joint_3_e": 0.2,
            "arm_right_joint_4_u": -0.8,
            "arm_right_joint_5_r": -0.25,
            "arm_right_joint_6_b": -1.85,
            "arm_right_joint_7_t": 0,
        }
        joint_start = planner.move_groups['arm_right'].get_current_state().joint_state
        plan = self.joint_motion_plan(
            joint_start,
            joint_goal,
            'arm_right',  # TODO both
        )
        return plan


# given the goal and distance to goal, generate a valid ee goal pose
# which points toward the goal
def generate_valid_goal(goal: Pose, distance):
    # the vector from the ee to goal should point toward positive x axis and negative z axis
    vector = np.random.normal(loc=0, size=3)
    if vector[0] < 0:
        vector[0] = -vector[0]
    if vector[2] > 0:
        vector[2] = -vector[2]
    vector = vector / np.linalg.norm(vector) * distance
    ee_pose = Pose()
    ee_pose.position.x = goal.position.x - vector[0]
    ee_pose.position.y = goal.position.y - vector[1]
    ee_pose.position.z = goal.position.z - vector[2]

    z_axis = vector
    x_axis = np.cross(z_axis, np.array([0, 0, 1]))
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    rot_mat = np.eye(4)
    rot_mat[:3, 0] = x_axis
    rot_mat[:3, 1] = y_axis
    rot_mat[:3, 2] = z_axis
    quat = tfm.quaternion_from_matrix(rot_mat)  # w x y z
    ee_pose.orientation.x = quat[1]
    ee_pose.orientation.y = quat[2]
    ee_pose.orientation.z = quat[3]
    ee_pose.orientation.w = quat[0]
    return ee_pose



if __name__ == "__main__":
    rospy.init_node("visibility_planner")
    server = GoalInteractiveMarkerServer()
    robot = MotomanSDA10F()
    # perception = robot.init_perception_interface()
    planner = robot.init_motion_planner()
    # grasp_planner = GraspPlanner()

    # motion planning to the end-effector pose
    joint_dict = {
        "torso_joint_b1": 0,
        # "arm_left_joint_1_s": 1.75,
        # "arm_left_joint_2_l": 0.8,
        # "arm_left_joint_3_e": 0,
        # "arm_left_joint_4_u": -0.66,
        # "arm_left_joint_5_r": 0,
        # "arm_left_joint_6_b": 0,
        # "arm_left_joint_7_t": 0,
        "arm_right_joint_1_s": -0.2,
        "arm_right_joint_2_l": 0,
        "arm_right_joint_3_e": 0.2,
        "arm_right_joint_4_u": -0.8,
        "arm_right_joint_5_r": -0.25,
        "arm_right_joint_6_b": -1.85,
        "arm_right_joint_7_t": 0,
    }

    # start_jnt = planner.move_groups['arm_right'].get_current_state().joint_state
    # start_jnt = planner.joint_state_from_joint_dict(joint_dict, 'arm_right')
    # print('start_jnt: ')
    # print(start_jnt)
    # print('name size: ')
    # print(len(start_jnt.name))
    # print('position size: ')
    # print(len(start_jnt.position))

    # r_start_state = planner.make_robot_state_msg(
    #     start_jnt,
    #     ee='motoman_right_ee',
    #     ee_links=robot.ignore_collision_ee_links,
    #     is_diff=False,
    # )

    # planner.move_groups['arm_right'].set_start_state(r_start_state)
    plan = planner.reset_robot()
    planner.move_groups['arm_right'].execute(plan, wait=True)
    input('selecting goal pose...')

    goal = server.pose
    ee_goal = generate_valid_goal(goal, 0.2)
    # visualzie the ee goal pose
    pose_pub = rospy.Publisher('ee_goal_pose', PoseStamped, queue_size=20)
    rospy.sleep(1.0)
    pose_stamp = PoseStamped()
    pose_stamp.header.frame_id = 'base_link'
    pose_stamp.pose = ee_goal
    pose_pub.publish(pose_stamp)
    input('next...')

    # TODO: update the goal as the arm pose that points to the goal
    ###############################################
    # planner.set_trajectory_constraints('arm_right')
    planner.set_path_constraints('arm_right')
    print('trajectory constraints:')
    print(planner.move_groups['arm_right'].get_trajectory_constraints())
    print('path constraints:')
    print(planner.move_groups['arm_right'].get_path_constraints())

    start_jnt = planner.move_groups['arm_right'].get_current_state().joint_state
    plan = planner.pose_motion_plan(
        start_jnt,
        ee_goal,
        'arm_right',
        is_diff=False,
    )

    rospy.spin()
