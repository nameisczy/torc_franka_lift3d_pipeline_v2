# USE_MUJOCO_PY_VIEWER = True

import os

# import pprint
# import traceback

os.environ["MUJOCO_GL"] = "egl"
# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_triton_gemm_any=True " "--xla_gpu_enable_latency_hiding_scheduler=true "
# )
# import sys
# import json
# import time

# # import glfw
# import pickle
# import numpy as np
# import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer

# if USE_MUJOCO_PY_VIEWER:
#     import mujoco_viewer
# else:
#     import mujoco.viewer

# import argparse
# import zmq
# import cv2
import rospy
from sensor_msgs.msg import Image

# import rospkg
# import tf2_ros
# import actionlib
# import transformations as tf
# from typing import Optional, List
from cv_bridge import CvBridge
# from rosgraph_msgs.msg import Clock
# from std_msgs.msg import Int32, String
# from geometry_msgs.msg import TransformStamped, Pose
# from visualization_msgs.msg import MarkerArray, Marker
# from sensor_msgs.msg import Image, JointState, CameraInfo
# from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle
# from control_msgs.msg import (
#     FollowJointTrajectoryAction,
#     FollowJointTrajectoryFeedback,
#     FollowJointTrajectoryResult,
# )

# from lab_vbnpm.msg import ObjectPoses, ObjectIdsToNames
# from lab_vbnpm.srv import FakeObjectControl, FakeObjectControlResponse

# from utils.visual_utils import encode_seg_img_rgb
# from utils.conversions import joint_state_to_dict, float_to_ros_duration

# Set MuJoCo's GL backend for headless rendering
print("MUJOCO ENV: ", os.environ["MUJOCO_GL"])

# A simple XML model with two cameras
xml_string = """
<mujoco>
  <worldbody>
    <light pos="0 0 1" directional="true"/>
    <camera name="front_view" pos="0 0 1.5"/>
    <camera name="side_view" pos="1.5 0 0.5" xyaxes="0 1 0 0 0 -1"/>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.5">
      <joint type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.3" pos="0 0 -0.3"/>
    </body>
  </worldbody>
</mujoco>
"""

# Load the MuJoCo model from the XML string
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# viewer = mujoco.viewer.launch_passive(model, data)

# Renderers
front_renderer = None # mujoco.Renderer(model)
side_renderer = None # mujoco.Renderer(model)

bridge = CvBridge()

# * set up ROS interfaces
# ** clock
# * init clock pub
# clock_pub = rospy.Publisher("clock", Clock, queue_size=10)
# sim_clock = Clock()
# sim_clock.clock = rospy.Time(0)

# publish image publishers
front_image_pub = rospy.Publisher("/mujoco/front_camera/image_raw", Image, queue_size=10)
side_image_pub = rospy.Publisher("/mujoco/side_camera/image_raw", Image, queue_size=10)


def publish_images(event):
    """Callback function for the rospy.Timer to publish images from both cameras."""
    global front_renderer, side_renderer
    if front_renderer is None or side_renderer is None:
        front_renderer = mujoco.Renderer(model)
        side_renderer = mujoco.Renderer(model)
    rospy.loginfo("Start publish image")

    # Render and publish from the front camera
    front_cam_id = model.camera("front_view").id
    front_renderer.update_scene(data, camera=front_cam_id)
    front_image = front_renderer.render()
    rospy.loginfo("Published front camera image.")
    front_image_pub.publish(bridge.cv2_to_imgmsg(front_image, "bgr8"))

    # Render and publish from the side camera
    side_cam_id = model.camera("side_view").id
    side_renderer.update_scene(data, camera=side_cam_id)
    side_image = side_renderer.render()
    rospy.loginfo("Published side camera image.")
    side_image_pub.publish(bridge.cv2_to_imgmsg(side_image, "bgr8"))


def main():
    """Initializes the ROS node and runs the simulation."""
    global front_image_pub, side_image_pub
    rospy.init_node("mujoco_image_publisher", anonymous=True)

    # Start the timer to publish images every 3 seconds
    rospy.Timer(rospy.Duration(1.0), publish_images)
    rospy.loginfo("Starting MuJoCo simulation loop...")

    while not rospy.is_shutdown(): #and viewer.is_running():
        # publish_images(None)
        mujoco.mj_step(model, data)
        # viewer.sync()

        # sim_clock.clock += float_to_ros_duration(model.opt.timestep)
        # clock_pub.publish(sim_clock)


if __name__ == "__main__":
    main()
