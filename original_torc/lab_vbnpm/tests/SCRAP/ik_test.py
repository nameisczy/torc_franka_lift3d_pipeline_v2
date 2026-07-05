#!/usr/bin/env python
from __future__ import print_function
import rospy

import bio_ik_msgs
import bio_ik_msgs.msg
import bio_ik_msgs.srv
import moveit_msgs.msg
import moveit_msgs.srv
import trajectory_msgs.msg

rospy.init_node("bio_ik_service_example")
# from motoman import MotomanSDA10F
# robot = MotomanSDA10F(is_sim)
# planner = robot.init_motion_planner()

display = moveit_msgs.msg.DisplayTrajectory()
display.trajectory.append(moveit_msgs.msg.RobotTrajectory())
response = None
t00 = rospy.Time.now()
for i in range(10):
    rospy.wait_for_service("/bio_ik/get_position_ik")
    get_bio_ik = rospy.ServiceProxy("/bio_ik/get_bio_ik", bio_ik_msgs.srv.GetIK)

    request = bio_ik_msgs.msg.IKRequest()
    request.group_name = "arm_right_bioik"
    request.timeout = rospy.Duration(0.1)
    request.avoid_collisions = True
    request.approximate = True

    request.pose_goals.append(bio_ik_msgs.msg.PoseGoal())
    request.pose_goals[-1].link_name = "motoman_right_ee"
    request.pose_goals[-1].pose.position.x = 0.6
    request.pose_goals[-1].pose.position.y = 0
    request.pose_goals[-1].pose.position.z = 1.1
    request.pose_goals[-1].pose.orientation.x = 0.5
    request.pose_goals[-1].pose.orientation.y = -0.5
    request.pose_goals[-1].pose.orientation.z = -0.5
    request.pose_goals[-1].pose.orientation.w = -0.5
    request.pose_goals[-1].weight = 10.0
    # if response is not None:
    #     request.robot_state = response.solution
    #     if display.trajectory_start == moveit_msgs.msg.RobotState():
    #         display.trajectory_start = response.solution
    #         display.trajectory[0].joint_trajectory.points.append(trajectory_msgs.msg.JointTrajectoryPoint())
    #         display.trajectory[0].joint_trajectory.points[-1].positions = response.solution.joint_state.position
    #         display.trajectory[0].joint_trajectory.points[-1].time_from_start.secs = 0
    # for k in range(5):
    #     request.position_goals.append(bio_ik_msgs.msg.PositionGoal())
    #     request.position_goals[-1].link_name = "motoman_right_ee"
    #     request.position_goals[-1].position.x = 0.6
    #     request.position_goals[-1].position.y = -0.25 + 0.1 * k
    #     request.position_goals[-1].position.z = 1.0
    #     request.position_goals[-1].weight = 1.0
    # for j in range(1000):
    #     request.min_distance_goals.append(bio_ik_msgs.msg.MinDistanceGoal())
    #     request.min_distance_goals[-1].link_name = "motoman_right_ee"
    #     request.min_distance_goals[-1].target.x = 0.6
    #     request.min_distance_goals[-1].target.y = -1.0
    #     request.min_distance_goals[-1].target.z = 1.05 + 0.00015 * j
    #     request.min_distance_goals[-1].distance = 0.2
    #     request.min_distance_goals[-1].weight = 1000.0/2
    # # request.max_distance_goals.append(bio_ik_msgs.msg.MaxDistanceGoal())
    # # request.max_distance_goals[-1].link_name = "motoman_right_ee"
    # # request.max_distance_goals[-1].target.x = 0.6
    # # request.max_distance_goals[-1].target.y = -0.25
    # # request.max_distance_goals[-1].target.z = 1.0
    # # request.max_distance_goals[-1].distance = 0.0
    # # request.max_distance_goals[-1].weight = 1.0
    # request.look_at_goals.append(bio_ik_msgs.msg.LookAtGoal())
    # request.look_at_goals[-1].link_name = "motoman_right_ee"
    # request.look_at_goals[-1].target.x = 1.0
    # request.look_at_goals[-1].target.y = 0.0
    # request.look_at_goals[-1].target.z = 1.0
    # request.look_at_goals[-1].axis.z = 1.0
    # request.look_at_goals[-1].weight = 1.0
    request.avoid_joint_limits_goals.append(bio_ik_msgs.msg.AvoidJointLimitsGoal())
    request.avoid_joint_limits_goals[-1].weight = 1.0
    request.avoid_joint_limits_goals[-1].primary = False
    # request.minimal_displacement_goals.append(bio_ik_msgs.msg.MinimalDisplacementGoal())
    # request.minimal_displacement_goals[-1].weight = 50.0
    # request.minimal_displacement_goals[-1].primary = True

    t0 = rospy.Time.now()
    response = get_bio_ik(request).ik_response
    t1 = rospy.Time.now()

    print(response.error_code.val)
    # print(response.solution.joint_state.position)
    print(response.solution_fitness)
    print("Time:", (t1 - t0).to_sec())

    display.trajectory[0].joint_trajectory.joint_names = response.solution.joint_state.name
    display.trajectory[0].joint_trajectory.points.append(trajectory_msgs.msg.JointTrajectoryPoint())
    display.trajectory[0].joint_trajectory.points[-1].positions = response.solution.joint_state.position
    display.trajectory[0].joint_trajectory.points[-1].time_from_start = rospy.Time(0.2 * (i + 1))
t11 = rospy.Time.now()
print("Total time:", (t11 - t00).to_sec())
display_publisher = rospy.Publisher(
    "/move_group/display_planned_path",
    moveit_msgs.msg.DisplayTrajectory,
    latch=True,
    queue_size=10
)
# print(display)
display_publisher.publish(display)
rospy.spin()
