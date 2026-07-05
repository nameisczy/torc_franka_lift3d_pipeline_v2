import rospy
from actionlib_msgs.msg import GoalStatusArray, GoalID

rospy.init_node('stop_all')
rospy.sleep(0.5)

cancel_pub = rospy.Publisher(
    '/joint_trajectory_action/cancel', GoalID, queue_size=1
)
stat_msg = rospy.wait_for_message(
    '/joint_trajectory_action/status', GoalStatusArray
)
cancel_pub.publish(GoalID())
rospy.sleep(0.5)
cancel_pub.publish(GoalID())
# for stat in stat_msg.status_list:
#     print(stat.goal_id.id)
#     print(stat.status)
#     cancel_pub.publish(stat.goal_id)
rospy.spin()
