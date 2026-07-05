import os
import sys
import json

import rospy
import numpy as np
import transformations as tf
from dm_control import mjcf

import tf2_ros
from rospkg import RosPack
from sensor_msgs.msg import JointState
from curobo.geom.types import WorldConfig
from geometry_msgs.msg import TransformStamped

from task_planner.ur5e import Ur5e
from task_planner.eutils import execute


def get_object_info(xml: str, num_obj: int):
    """Extract object poses and dimensions from MuJoCo XML scene."""
    world = mjcf.from_path(xml)
    obj_stats = {}

    for i in range(num_obj):
        st = world.worldbody.body[f"obj_{i}"]
        st_quat = list(st.quat) if st.quat is not None else [1, 0, 0, 0]
        st_g_quat = (
            list(st.geom[0].quat) if st.geom[0].quat is not None else [1, 0, 0, 0]
        )
        st_t_quat = tf.quaternion_multiply(st_quat, st_g_quat)

        obj_stats[f"obj_{i}"] = {
            "pose": list(st.pos) + list(st_t_quat),
            "dims": list(2 * st.geom[0].size),
        }

    static_stats = {}
    shelf_body = world.worldbody.body["shelf"]
    body_pos = shelf_body.pos
    for child in shelf_body.geom:
        pos = list(np.add(body_pos, child.pos))
        quat = list(child.quat) if child.quat is not None else [1, 0, 0, 0]
        static_stats[child.name] = {
            "pose": pos + quat,
            "dims": list(2 * child.size),
        }

    # Create motion queries - positioning above each object
    motion_stats = {}
    top_down_quat = tf.quaternion_from_euler(np.pi, 0, 0).tolist()

    for i in range(num_obj):
        start_pos = obj_stats[f"obj_{i}"]["pose"][:3]
        start_pos[2] += obj_stats[f"obj_{i}"]["dims"][1] / 2
        # start_pos[1] += obj_stats[f"obj_{i}"]["dims"][2] / 4
        start_pos[1] -= obj_stats[f"obj_{i}"]["dims"][2] / 2
        start_pos[2] = 0.058

        motion_stats[f"obj_{i}"] = {
            "pose": start_pos + top_down_quat,
        }

    return obj_stats, static_stats, motion_stats


# Initialize ROS and get scene files
rp = RosPack()
base_path = rp.get_path("lab_vbnpm")
default_xml = base_path + "/tests/packing0.xml"
default_json = base_path + "/tests/packing0.json"
scene_xml = sys.argv[1] if len(sys.argv) > 1 else default_xml
scene_json = sys.argv[2] if len(sys.argv) > 2 else default_json

# Load scene configuration
with open(scene_json, "r") as f:
    data = json.load(f)
placement_order = data["placement_order"]
if "names" in data:
    obj_names = data["names"]
else:
    obj_names = [f"obj_{i}" for i in range(len(placement_order))]

# Get object information from scene
obj_info, static_info, motion_info = get_object_info(
    scene_xml,
    len(placement_order),
)

goto_order = sorted(
    range(len(placement_order)),
    key=lambda k: tuple(reversed(motion_info[f"obj_{k}"]["pose"][0:2])),
)
print(f"Go-to order: {[(i, obj_names[i]) for i in goto_order]}")
print(motion_info["obj_" + str(goto_order[0])])

# Initialize ROS node and robot
rospy.init_node("ur5e_position_above_objects")
is_sim = True if len(sys.argv) > 3 else False

# Initialize robot and planner
robot = Ur5e(is_sim, False)
planner = robot.init_motion_planner(planner="curobo")

# Get current joint state
joint_state = rospy.wait_for_message("/joint_states_all", JointState, timeout=5)

# Set robot parameters
rospy.set_param("/robot/acc_ang_lim", 687)
rospy.set_param("/robot/vel_ang_lim", 20)

# Setup world collision model
world_dict = {
    "cuboid": {
        "plane_floor": {
            "pose": [0, 0, -0.01, 1, 0, 0, 0],
            "dims": [10, 10, 0.01],
        }
    },
}
if not is_sim:
    world_dict["cuboid"]["plane_back"] = {
        "pose": [-0.59, 0, 0, 1, 0, 0, 0],
        "dims": [0.01, 5, 5],
    }
world_dict["cuboid"].update(static_info)
world_dict["cuboid"].update(obj_info)
planner.world_config = WorldConfig.from_dict(world_dict)
planner.update_world_motion_gen()
planner.visualize_rviz()
planner.visualize_spheres_rviz(joint_state)

# Tool offset for suction cup
tool_offset = 0.175
offset_dist = 0.15  # Additional clearance above object

try:
    RETIME = True  # is_sim

    # Go to top right corner of shelf bottom
    shelf_bottom = static_info["shelf_base"]
    shelf_pose = shelf_bottom["pose"]
    shelf_dims = shelf_bottom["dims"]

    # Calculate top right corner position
    # x + width/2 (right), y - depth/2 (front), z + height/2 (top)
    corner_pos = [
        shelf_pose[0] - shelf_dims[0] / 2,  # right edge
        shelf_pose[1] - shelf_dims[1] / 2,  # front edge
        shelf_pose[2] + shelf_dims[2] / 2,  # top surface
    ]
    forward_quat = tf.quaternion_from_euler(-np.pi / 2, 0, -np.pi / 2).tolist()
    corner_pose = corner_pos + forward_quat
    corner_pose[0] -= tool_offset + 0.02  # Offset for tool length and clearance

    # br = tf2_ros.TransformBroadcaster()
    # # Publish TF for visualization
    # t = TransformStamped()
    # t.header.stamp = rospy.Time.now()
    # t.header.frame_id = "world"
    # t.child_frame_id = f"test_pose_obj"
    # t.transform.translation.x = corner_pos[0]
    # t.transform.translation.y = corner_pos[1]
    # t.transform.translation.z = corner_pos[2]
    # t.transform.rotation.w = forward_quat[0]
    # t.transform.rotation.x = forward_quat[1]
    # t.transform.rotation.y = forward_quat[2]
    # t.transform.rotation.z = forward_quat[3]
    # br.sendTransform(t)

    # # joint_state.position = np.add(joint_state.position,[3.14,0,0,0,0,0]).tolist()
    # planner.pink_cartesian_motion(
    #     joint_state,
    #     corner_pose,
    #     offset=[0, 0, 0, 1, 0, 0, 0],
    #     threshold=1e-4,
    #     constraint_in_goal_frame=False,
    #     return_all=False,
    #     dt=0.1,
    # )
    # input("next?")

    # Plan motion to shelf corner
    plan = planner.pose_motion_plan(joint_state, corner_pose)

    input("Go to shelf corner?")

    execute(plan, window=0, retime=RETIME)
    print("\nPositioned at top right corner of shelf bottom")

    FIRST = True
    prev_y = motion_info[f"obj_{goto_order[0]}"]["pose"][1]
    for i in goto_order:
        # Get target pose above object
        target_pose = motion_info[f"obj_{i}"]["pose"].copy()
        target_pose[2] += tool_offset  # Offset for tool length
        target_pose[2] += offset_dist  # Additional clearance

        input("Read to plan?")
        joint_state = rospy.wait_for_message("/joint_states_all", JointState, timeout=5)
        # Plan motion to position above object
        if FIRST or (np.sign(target_pose[1]) != np.sign(prev_y)):
            plan = planner.pose_motion_plan(joint_state, target_pose)
            FIRST=False
        else:
            plan = planner.pink_cartesian_motion(
                joint_state,
                target_pose,
                offset=[0, 0, 0, 1, 0, 0, 0],
                threshold=1e-4,
                # constraint_in_goal_frame=False,
                return_all=False,
            )
        prev_y = target_pose[1]

        # Wait for user confirmation
        input(f"next?")

        # Execute motion
        execute(plan, window=0, retime=RETIME)

        print(f"\nPositioned above {obj_names[i]}")

    input("Read to plan?")
    joint_state = rospy.wait_for_message("/joint_states_all", JointState, timeout=5)

    reset = {
        "shoulder_pan_joint": 0,
        "shoulder_lift_joint": -2.2,
        "elbow_joint": 1.9,
        "wrist_1_joint": -1.383,
        "wrist_2_joint": -1.57,
        "wrist_3_joint": 0.00,
    }
    plan = planner.joint_motion_plan(joint_state, reset)

    input("Reset?")
    execute(plan, window=0, retime=RETIME)

except Exception as e:
    print(f"\nERROR: {e}")
