import os
import sys
import copy
import pickle
import numpy as np

import rospy
from rospkg import RosPack

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from industrial_msgs.msg import RobotStatus
from visualization_msgs.msg import InteractiveMarkerUpdate
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lab_vbnpm.srv import GetNextPlanningPoint, GetNextPlanningPointResponse
from motion_planner.curobo_planner import CuroboPlanner
from task_planner.open_loop import update_and_get_points
from task_planner.eutils import ee_open, ee_close, execute

goal_pose = Pose()
plan_traj = JointTrajectory()


def marker_callback(update_msg):
    poses = update_msg.poses
    if len(poses) > 0:
        global goal_pose
        goal_pose = poses[0].pose
        print(goal_pose)


rospy.init_node("planning")

rospy.Subscriber(
    '/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/update',
    InteractiveMarkerUpdate, marker_callback
)

## parse args ##
is_sim = sys.argv[1][0] not in (
    '0',
    'r',
    'R',
    'n',
    'N',
) if len(sys.argv) > 1 else True
gt = sys.argv[2][0] not in (
    '0',
    'r',
    'R',
    'n',
    'N',
) if len(sys.argv) > 2 else True
object_to_grasp = sys.argv[3] if len(sys.argv) > 3 else 35

if is_sim:
    ## instatiate perception ##
    from task_planner.motoman import MotomanSDA10F

    robot = MotomanSDA10F(is_sim, gt)
    perception = robot.init_perception_interface()

    ## sense the environment ##
    result = update_and_get_points(object_to_grasp, robot, perception)
    points, tgt_pts, tgt_rgb, all_pts, all_rgb, ds_all_pts = result

## instatiate planner ##
rp = RosPack()
robot_urdf = rp.get_path('motoman_sda10f_moveit_config')
robot_urdf += '/config/gazebo_motoman_sda10f.urdf'
curobo_config = rp.get_path('lab_vbnpm')
curobo_config += '/robots/motoman/curobo/motoman.yml'
# planner = MoveitPlanner(
planner = CuroboPlanner(
    robot_urdf,
    ['motoman_left_ee', 'motoman_right_ee'],
    curobo_config,
    is_sim=is_sim,
)

if is_sim:
    planner.set_planning_scene(all_pts)

t1 = None
prev_goal_pose = Pose()
while not rospy.is_shutdown():
    rospy.sleep(0.1)
    if goal_pose == prev_goal_pose:
        continue

    prev_goal_pose = copy.deepcopy(goal_pose)

    robot_status = rospy.wait_for_message(
        '/robot_status', RobotStatus, timeout=5
    )
    if robot_status.in_motion.val == 0:
        joint_state = rospy.wait_for_message(
            '/joint_states_all', JointState, timeout=5
        )
        plan = planner.pose_motion_plan(joint_state, prev_goal_pose)
        if type(plan) is JointTrajectory:
            execute(plan, retime=True)
        t = rospy.Time.now().to_sec()
    else:
        error = 888
        while error != 0:
            print('error loop:', error)
            plan = None
            t0 = rospy.Time.now().to_sec()
            if t1 is None or tft.to_sec() < (t1 - t):
                rospy.wait_for_service('get_next_planning_point', timeout=5)
                try:
                    get_next_planning_point = rospy.ServiceProxy(
                        'get_next_planning_point', GetNextPlanningPoint
                    )
                    planning_point = get_next_planning_point()
                    tft = planning_point.point.time_from_start
                    if planning_point.point == JointTrajectoryPoint():
                        t1 = None
                        break
                    print('Next PP: Done!')
                except rospy.ServiceException as e:
                    print("Service call failed: %s" % e)
                    break
                t2 = rospy.Time.now().to_sec() - t0

            joint_state = JointState(
                name=planning_point.joint_names,
                position=planning_point.point.positions,
                velocity=planning_point.point.velocities,
            )

            plan = planner.pose_motion_plan(joint_state, prev_goal_pose)
            t1 = rospy.Time.now().to_sec()
            if type(plan) is not JointTrajectory:
                error = -88
                continue
            for i in range(len(plan.points)):
                plan.points[i].time_from_start += tft
            print('Planning Time:', (t1 - t0))
            print('Get Point Time:', t2)
            print('t1-t:', t1 - t)
            print('ppt:', tft.to_sec())
            print('ppt > t1-t:', tft.to_sec() > t1 - t)

            error, points = execute(plan, retime=True)
            if error == 0:
                t1 = None
