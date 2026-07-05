import rospy
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import MarkerArray, Marker

rospy.init_node('get_rviz_click')

marker_pub = rospy.Publisher('/plot_grasps', MarkerArray)

while 'q' != (x := input("Press 'q' to quit or any other key to continue: ")):
    msg = rospy.wait_for_message('/clicked_point', PointStamped)
    print(f"Clicked point: x={msg.point.x}, y={msg.point.y}, z={msg.point.z}")
    marker = MarkerArray()
    marker.markers.append(Marker())
    marker.markers[-1].header.frame_id = 'world'
    marker.markers[-1].header.stamp = rospy.Time()
    marker.markers[-1].ns = "clicked_points"
    marker.markers[-1].id = 0
    marker.markers[-1].type = Marker.SPHERE
    marker.markers[-1].action = Marker.ADD
    marker.markers[-1].pose.position.x = msg.point.x
    marker.markers[-1].pose.position.y = msg.point.y
    marker.markers[-1].pose.position.z = msg.point.z
    marker.markers[-1].scale.x = 0.1
    marker.markers[-1].scale.y = 0.1
    marker.markers[-1].scale.z = 0.1
    marker.markers[-1].color.a = 1
    marker.markers[-1].color.r = 1
    marker.markers[-1].color.g = 1
    marker.markers[-1].color.b = 0
    marker_pub.publish(marker)
