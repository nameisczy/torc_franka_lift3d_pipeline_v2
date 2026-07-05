#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker
from std_msgs.msg import Header

def publish_marker():
    # Initialize the ROS node
    rospy.init_node('hello_marker_node', anonymous=True)
    
    # Create a publisher to the 'visualization_marker' topic
    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    
    # Wait for RViz to connect to the topic
    rospy.sleep(1)
    
    # Create a marker
    marker = Marker()
    
    # Set the frame ID to define where the marker will appear in RViz
    marker.header.frame_id = "map"  # or any other frame in your tf tree
    marker.header.stamp = rospy.Time.now()
    
    # Set the type of marker
    marker.type = Marker.SPHERE
    
    # Set the action to "add" (you could remove a marker later by publishing a DELETE action)
    marker.action = Marker.ADD
    
    # Set the scale of the marker (1x1x1 meter sphere)
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    
    # Set the color of the marker (red, fully opaque)
    marker.color.a = 1.0  # Alpha, must be non-zero for the marker to appear
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    
    # Set the position of the marker in the "world" frame
    marker.pose.position.x = 1.0
    marker.pose.position.y = 1.0
    marker.pose.position.z = 1.0
    
    # Set orientation (no rotation)
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    
    # Publish the marker at a regular rate
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        marker.header.stamp = rospy.Time.now()  # Update timestamp
        marker_pub.publish(marker)
        rate.sleep()

if __name__ == "__main__":
    try:
        publish_marker()
    except rospy.ROSInterruptException:
        pass