#!/usr/bin/env python

import rospy
import sys
import numpy as np
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Point32
import sensor_msgs.point_cloud2 as pc2
import mujoco
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import tf.transformations

class InteractivePointCloudPublisher:
    def __init__(self, point_cloud):
        # Initialize the ROS node
        rospy.init_node('interactive_point_cloud_publisher', anonymous=True)
        self.pub = rospy.Publisher('interactive_point_cloud', PointCloud2, queue_size=10)
        rospy.Subscriber('/debug/target_points', PointCloud2, self.set_initial_pose)
        
        # Initialize the point cloud
        self.point_cloud = point_cloud
        self.position = np.array([0.0, 0.0, 0.0])  # Position: x, y, z
        self.orientation = np.array([0.0, 0.0, 0.0])  # Orientation (roll, pitch, yaw)
        self.fixed_pose_yet = False

        # Set the rate
        self.rate = rospy.Rate(10)  # 10 Hz

    def set_initial_pose(self, pcd_inp):
        if self.fixed_pose_yet == False:
            print("initial pose set")
            for point in pc2.read_points(pcd_inp, field_names=("x", "y", "z"), skip_nans=True):
                self.position[0] = point[0]
                self.position[1] = point[1]
                self.position[2] = point[2]
                self.fixed_pose_yet = True
                return #worst code ever written but it works

    def publish_point_cloud(self):
        while not rospy.is_shutdown():
            # Apply transformations based on position and orientation
            transformed_cloud = self.transform_point_cloud()

            # Create PointCloud2 message
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'base_link'  # Change to your frame_id

            # Publish the transformed point cloud
            pc2_msg = pc2.create_cloud_xyz32(header, transformed_cloud)
            self.pub.publish(pc2_msg)

            self.rate.sleep()

    def transform_point_cloud(self):
        # Rotate the point cloud
        rotation_matrix = self.rotation_matrix_from_euler(self.orientation[0], self.orientation[1], self.orientation[2])
        
        # Apply transformation
        transformed_points = np.dot(self.point_cloud, rotation_matrix.T) + self.position
        return transformed_points

    def rotation_matrix_from_euler(self, roll, pitch, yaw):
        # Calculate rotation matrix from Euler angles
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        return np.dot(R_z, np.dot(R_y, R_x))

    def keyboard_listener(self):
        print("Enter commands to move and rotate the point cloud.")
        print("Commands:")
        print(" 'up'     : Move up in Y")
        print(" 'down'   : Move down in Y")
        print(" 'left'   : Move left in X")
        print(" 'right'  : Move right in X")
        print(" 'w'      : Move up in Z")
        print(" 's'      : Move down in Z")
        print(" 'a'      : Rotate left around Z")
        print(" 'd'      : Rotate right around Z")
        print(" 'q'      : Rotate up around X")
        print(" 'e'      : Rotate down around X")
        print(" 'r'      : Rotate right around Y")
        print(" 'f'      : Rotate left around Y")
        print(" 'half'      : Half movement / rotation strength")
        print(" 'double'      : Double movement / rotation strength")
        print(" 'show'      : Show position and rotation")
        print(" 'exit'   : Exit the program")
        scale = 1.0

        while not rospy.is_shutdown():
            command = input("Enter command: ")

            if command == 'up':
                self.position[1] += (0.1 * scale)  # Move up in Y
            elif command == 'down':
                self.position[1] -= (0.1 * scale)  # Move down in Y
            elif command == 'left':
                self.position[0] -= (0.1 * scale)  # Move left in X
            elif command == 'right':
                self.position[0] += (0.1 * scale)  # Move right in X
            elif command == 'w':
                self.position[2] += (0.1 * scale)  # Move up in Z
            elif command == 's':
                self.position[2] -= (0.1 * scale)  # Move down in Z
            elif command == 'a':
                self.orientation[2] += (0.1 * scale)  # Rotate left around Z
            elif command == 'd':
                self.orientation[2] -= (0.1 * scale)  # Rotate right around Z
            elif command == 'q':
                self.orientation[0] += (0.1 * scale)  # Rotate up around X
            elif command == 'e':
                self.orientation[0] -= (0.1 * scale)  # Rotate down around X
            elif command == 'r':
                self.orientation[1] += (0.1 * scale)  # Rotate right around Y
            elif command == 'f':
                self.orientation[1] -= (0.1 * scale)  # Rotate left around Y
            elif command == 'half':
                scale /= 2.0
            elif command == 'double':
                scale *= 2.0
            elif command == 'show':
                quaternion = tf.transformations.quaternion_from_euler(self.orientation[0], 
                                                        self.orientation[1], 
                                                        self.orientation[2])
                print("position", self.position, "rotation euler", self.orientation, "rotation quaternion", quaternion)
            elif command == 'exit':
                print("Exiting...")
                break
            else:
                print("Invalid command! Please try again.")


def get_pcd(xml_file, body_name="004_sugar_box"):
        #Turn xml file into pcd vertices, get the rotation and position out too
        #Get pos and quat from xml
        mj_model = mujoco.MjModel.from_xml_path(xml_file) #Current directory is in task_planner
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        
        #Get mesh vertices from xml
        geom_id = mj_model.body_geomadr[body_id] #This gets the first geom address but here I just assume it is one geom associated
        mesh_id = mj_model.geom_dataid[geom_id]
        start_vert = mj_model.mesh_vertadr[mesh_id]  # Start vertex index for this mesh
        vert_count = mj_model.mesh_vertnum[mesh_id]
        vertices = mj_model.mesh_vert[start_vert:start_vert + vert_count].reshape(-1, 3) #so go from start vertex for that many vertices to get all the vertices, reshape [vertices, xyz]

        return vertices

#Provide the xml file and body name of the object that we are adjusting.
#Once you adjust it right you can print out the position and orientation of the object. You can save that to the testing xml file position and orientation. 
if __name__ == '__main__':
    # Example: Creating a point cloud with 100 points in the shape (n, 3)
    #num_points = 100
    xml_file = sys.argv[1] if len(sys.argv) > 1 else "./xmls/real_experiment_adjusted.xml"
    body_name = sys.argv[2] if len(sys.argv) > 2 else "005_tomato_soup_can"
    # Generate random points in 3D space
    point_cloud = get_pcd(xml_file, body_name) #np.random.rand(num_points, 3)  # Shape: (100, 3)
    #python adjust_object.py "../../xmls/ycb_boxes.xml" "004_sugar_box"

    try:
        interactive_publisher = InteractivePointCloudPublisher(point_cloud)
        # Start the keyboard listener in a separate thread
        from threading import Thread
        listener_thread = Thread(target=interactive_publisher.keyboard_listener)
        listener_thread.start()

        ## Publish the point cloud
        interactive_publisher.publish_point_cloud()
    except rospy.ROSInterruptException:
        pass