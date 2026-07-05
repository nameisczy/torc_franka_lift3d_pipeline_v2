import os
import sys
import json
import copy

import cv2
import numpy as np
import trimesh as tm
from dm_control import mjcf
import transformations as tf
from cv_bridge import CvBridge

import rospy
from rospkg import RosPack
from sensor_msgs.msg import JointState, Image
import tf2_ros
from geometry_msgs.msg import TransformStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from curobo.types.math import Pose
from curobo.geom.types import WorldConfig, Mesh
from curobo.geom.sphere_fit import SphereFitType


from task_planner.ur5e import Ur5e
from task_planner.eutils import ee_suction_on, ee_suction_off, execute
from utils.conversions import list_to_matrix


def adjust_traj_base_rotation(traj):
    index = traj.joint_names.index("shoulder_pan_joint")
    for point in traj.points:
        point.positions[index] += np.pi / 2
        if point.positions[index] > np.pi:
            point.positions[index] -= 2 * np.pi
    return traj


def get_object_info(xml: str, num_obj: int):
    world = mjcf.from_path(xml)
    obj_stats = {}
    for i in range(num_obj):
        st = world.worldbody.body[f"obj_{i}"]
        st_quat = list(st.quat) if st.quat is not None else [1, 0, 0, 0]
        st_g_quat = (
            list(st.geom[0].quat) if st.geom[0].quat is not None else [1, 0, 0, 0]
        )
        st_t_quat = tf.quaternion_multiply(st_quat, st_g_quat)

        # # Account for geometry offset from body frame
        # st_g_pos = st.geom[0].pos if st.geom[0].pos is not None else [0, 0, 0]
        # # Transform geometry offset by body rotation to get actual geometry center in world coordinates
        # rotation_matrix = tf.quaternion_matrix(st_quat)[:3, :3]
        # transformed_offset = np.dot(rotation_matrix, st_g_pos)
        # actual_pos = np.add(st.pos, transformed_offset)

        obj_stats[f"obj_{i}"] = {
            "pose": list(st.pos) + list(st_t_quat),
            "dims": list(2 * st.geom[0].size),
        }

    tgt_stats = {}
    for i in range(num_obj):
        pl = world.worldbody.body[f"obj_{i}_tgt"]
        pl_quat = list(pl.quat) if pl.quat is not None else [1, 0, 0, 0]
        pl_g_quat = (
            list(pl.geom[0].quat) if pl.geom[0].quat is not None else [1, 0, 0, 0]
        )
        pl_t_quat = tf.quaternion_multiply(pl_quat, pl_g_quat)

        tgt_stats[f"obj_{i}_tgt"] = {
            "pose": list(pl.pos) + list(pl_t_quat),
            "dims": list(2 * pl.geom[0].size),
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

    # get motion queries from xml
    motion_stats = {}
    top_down_quat = tf.quaternion_from_euler(np.pi, 0, 0).tolist()
    front_side_quat = tf.quaternion_from_euler(-np.pi / 2, 0, -np.pi / 2).tolist()
    for i in range(num_obj):
        gt = world.worldbody.body[f"obj_{i}_tgt"]
        target_pos = [
            gt.pos[0] - gt.geom[0].size[1],
            gt.pos[1],
            gt.pos[2] + gt.geom[0].size[2] / 2,
        ]

        start_pos = obj_stats[f"obj_{i}"]["pose"][:3]
        start_pos[2] += obj_stats[f"obj_{i}"]["dims"][1] / 2
        start_pos[1] += obj_stats[f"obj_{i}"]["dims"][2] / 4

        motion_stats[f"obj_{i}"] = {
            "start": start_pos + top_down_quat,
            "goal": target_pos + front_side_quat,
        }

    return obj_stats, tgt_stats, static_stats, motion_stats


bridge = CvBridge()
rp = RosPack()
base_path = rp.get_path("lab_vbnpm")
default_xml = base_path + "/tests/packing0.xml"
default_json = base_path + "/tests/packing0.json"
scene_xml = sys.argv[1] if len(sys.argv) > 1 else default_xml
scene_json = sys.argv[2] if len(sys.argv) > 2 else default_json

with open(scene_json, "r") as f:
    data = json.load(f)
placement_order = data["placement_order"]
if "names" in data:
    obj_names = data["names"]
else:
    obj_names = [f"obj_{i}" for i in range(len(placement_order))]

obj_info, tgt_info, static_info, motion_info = get_object_info(
    scene_xml,
    len(placement_order),
)

rospy.init_node("ur5e_main_run")
is_sim = True if len(sys.argv) > 3 else False
## instatiate planner ##
robot = Ur5e(is_sim, False)
# perception = robot.init_perception_interface()
planner = robot.init_motion_planner(planner="curobo")  # ,warmup=False)

joint_state = rospy.wait_for_message("/joint_states_all", JointState, timeout=5)

rospy.set_param("/robot/acc_ang_lim", 687)
rospy.set_param("/robot/vel_ang_lim", 20)

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

offset_dist = 0.05
offset_small = 0.01
tool_offset = 0.175
shelf_height = 0.3
offset_up = 0.01

# Create TF broadcaster for debugging
# br = tf2_ros.TransformBroadcaster()

# for i in range(len(placement_order)):
#     # test_pose = motion_info[f"obj_{i}"]["start"]
#     # test_pose[2] += tool_offset  # offset for tool

#     test_pose = motion_info[f"obj_{i}"]["goal"]
#     test_pose[0] -= tool_offset  # offset for tool

#     # Publish TF for visualization
#     # t = TransformStamped()
#     # t.header.stamp = rospy.Time.now()
#     # t.header.frame_id = "world"
#     # t.child_frame_id = f"test_pose_obj_{i}"
#     # t.transform.translation.x = test_pose[0]
#     # t.transform.translation.y = test_pose[1]
#     # t.transform.translation.z = test_pose[2]
#     # t.transform.rotation.w = test_pose[3]
#     # t.transform.rotation.x = test_pose[4]
#     # t.transform.rotation.y = test_pose[5]
#     # t.transform.rotation.z = test_pose[6]
#     # br.sendTransform(t)

#     # print(f"Published TF for obj_{i}: pos=[{test_pose[0]:.3f}, {test_pose[1]:.3f}, {test_pose[2]:.3f}], quat=[{test_pose[3]:.3f}, {test_pose[4]:.3f}, {test_pose[5]:.3f}, {test_pose[6]:.3f}]")

#     # joint_state.position = np.add(joint_state.position,[3.14,0,0,0,0,0]).tolist()
#     planner.pink_cartesian_motion(
#         joint_state,
#         test_pose,
#         offset=[0, 0, 0, 1, 0, 0, 0],
#         threshold=1e-4,
#         constraint_in_goal_frame=False,
#         return_all=False,
#     )
#     input("next?")
# exit()
try:
    PAUSE = not is_sim
    FIRST = is_sim
    RETIME = True  # is_sim
    for i in placement_order:
        if PAUSE:
            to_skip = input(f"Skip {obj_names[i]}?")
            if to_skip in ("Y", "y"):
                continue
        suction_pose = motion_info[f"obj_{i}"]["start"]
        suction_pose[2] += tool_offset  # offset for tool
        suction_pose[2] += offset_dist - 0.005  # offset for approach

        suction_pose_end = motion_info[f"obj_{i}"]["goal"]
        suction_pose_end[0] -= tool_offset  # offset for tool

        end_z = suction_pose_end[2] + 0.005
        while end_z - 0.05 <= shelf_height:
            end_z += 0.01
        end_z += offset_up
        suction_pose_end[2] = end_z

        obj_depth = obj_info[f"obj_{i}"]["dims"][1]
        shelf_pose = static_info["shelf_base"]["pose"]
        shelf_dims = static_info["shelf_base"]["dims"]
        shelf_begins = shelf_pose[0] - shelf_dims[0] / 2 - tool_offset
        offset_forward = obj_depth + suction_pose_end[0] - shelf_begins
        suction_pose_end[0] -= offset_forward + offset_small
        # suction_pose_end[0] -= obj_depth + offset_small  # offset backward
        # suction_pose_end[1] -= offset_small  # offset right a little bit

        ## pre pick ##
        plan1 = JointTrajectory()
        while len(plan1.points) == 0:
            joint_state = rospy.wait_for_message(
                "/joint_states_all", JointState, timeout=5
            )
            plan1 = planner.pose_motion_plan(joint_state, suction_pose)

            if PAUSE or FIRST:
                to_skip = input("Execute?")
                FIRST = False
                if to_skip:
                    break
        # rospy.set_param("/robot/vel_ang_lim", 180)
        execute(plan1, window=0, retime=RETIME)

        ## approach ##
        if PAUSE:
            input("Ready to plan approach?")
            joint_state2 = rospy.wait_for_message(
                "/joint_states_all", JointState, timeout=5
            )
        else:
            joint_state2 = JointState()
            joint_state2.name = plan1.joint_names
            joint_state2.position = plan1.points[-1].positions
        plan2, success = planner.pink_cartesian_motion(
            joint_state2,
            suction_pose,
            offset=[0, 0, -offset_dist, 1, 0, 0, 0],
            threshold=1e-4,
            constraint_in_goal_frame=False,
            return_all=True,
        )
        # suction_pose[2] += offset_dist
        # plan2 = planner.pose_motion_plan(
        #     joint_state2,
        #     suction_pose,
        #     path_constraint=[0.9, 0.9, 0.9, 0.9, 0.9, 0],
        #     # path_constraint=[1, 1, 1, 1, 1, 0],
        #     constraint_in_goal_frame=True,
        # )

        if PAUSE:
            input("Execute?")
        # rospy.set_param("/robot/vel_ang_lim", 180)
        execute(plan2, window=0, retime=True)
        rospy.sleep(1)
        if PAUSE:
            print(f"About to pick {obj_names[i]}")
            input("Suction on?")
        ee_suction_on()

        ## lift ##
        joint_state3 = JointState()
        joint_state3.name = plan2.joint_names
        joint_state3.position = plan2.points[-1].positions
        plan3, success = planner.pink_cartesian_motion(
            joint_state3,
            suction_pose,
            offset=[0, 0, obj_depth + 0.5 * offset_dist, 1, 0, 0, 0],
            constraint_in_goal_frame=False,
            return_all=True,
        )
        # suction_pose[2] += offset_dist
        # plan3 = planner.pose_motion_plan(
        #     joint_state3,
        #     suction_pose,
        #     path_constraint=[0.9, 0.9, 0.9, 0.9, 0.9, 0],
        #     # path_constraint=[1, 1, 1, 1, 1, 0],
        #     constraint_in_goal_frame=False,
        # )

        if PAUSE:
            input("Execute?")
        # rospy.set_param("/robot/vel_ang_lim", 20)
        execute(plan3, window=0, retime=True)

        ## move to pre-place pose ##
        if PAUSE:
            input("Ready to plan approach?")
            joint_state4 = rospy.wait_for_message(
                "/joint_states_all", JointState, timeout=5
            )
        else:
            joint_state4 = JointState()
            joint_state4.name = plan3.joint_names
            joint_state4.position = plan3.points[-1].positions

        # attach object to robot #
        mat = list_to_matrix(obj_info[f"obj_{i}"]["pose"])
        box = tm.primitives.Box(obj_info[f"obj_{i}"]["dims"], mat)
        sf = tm.sample.sample_surface_even(box, 10000)
        surface, find = sf
        target_mesh = Mesh.from_pointcloud(
            surface,
            planner.resolution,
            "target",
        )
        attach_state = planner.parse_joint_state(joint_state4)
        offset = Pose.from_list(
            [0, 0, obj_depth + 1.5 * offset_dist, 1, 0, 0, 0],
            planner.tensor_args,
        )
        planner.motion_gen.attach_external_objects_to_robot(
            attach_state,
            [target_mesh],
            link_name="tool0",
            surface_sphere_radius=0.01,
            sphere_fit_type=SphereFitType.SAMPLE_SURFACE,
            world_objects_pose_offset=offset,
        )
        planner.update_world_motion_gen()
        planner.visualize_spheres_rviz(joint_state4)

        plan4 = JointTrajectory()
        while len(plan4.points) == 0:
            joint_state4 = rospy.wait_for_message(
                "/joint_states_all", JointState, timeout=5
            )
            plan4 = planner.pose_motion_plan(
                joint_state4,
                suction_pose_end,
            )
            if PAUSE:
                to_skip = input("Execute?")
                if to_skip:
                    break
        # rospy.set_param("/robot/vel_ang_lim", 20)
        execute(plan4, window=0, retime=RETIME)

        ## approach place pose ##
        if PAUSE:
            input("Ready to plan approach?")
            joint_state5 = rospy.wait_for_message(
                "/joint_states_all", JointState, timeout=5
            )
        else:
            joint_state5 = JointState()
            joint_state5.name = plan4.joint_names
            joint_state5.position = plan4.points[-1].positions
        plan5, success = planner.pink_cartesian_motion(
            joint_state5,
            suction_pose_end,
            offset=[offset_forward + offset_small + 0.005, 0, -offset_up, 1, 0, 0, 0],
            constraint_in_goal_frame=False,
            return_all=True,
        )
        # suction_pose_end[0] += offset_forward + offset_small + 0.005
        # planner.motion_gen.detach_object_from_robot("tool0")
        # plan5 = planner.pose_motion_plan(
        #     joint_state5,
        #     suction_pose_end,
        #     # path_constraint=[0.9, 0.9, 0.9, 0.9, 0.9, 0],
        #     path_constraint=[1, 1, 1, 1, 1, 0],
        #     constraint_in_goal_frame=True,
        # )

        if PAUSE:
            input("Execute?")
        # rospy.set_param("/robot/vel_ang_lim", 20)
        execute(plan5, window=0, retime=True)
        rospy.sleep(1)
        if PAUSE:
            input("Release?")
        ee_suction_off()

        ## retreat ##
        plan6 = copy.deepcopy(plan5)
        plan6.points = list(reversed(plan5.points))

        # update collision and joint info for next iteration
        planner.motion_gen.detach_object_from_robot("tool0")
        world_dict["cuboid"][f"obj_{i}_tgt"] = tgt_info[f"obj_{i}_tgt"]
        planner.world_config = WorldConfig.from_dict(world_dict)
        planner.update_world_motion_gen()

        if PAUSE:
            input("Execute?")
        # rospy.set_param("/robot/vel_ang_lim", 180)
        execute(plan6, window=0, retime=True)

        if PAUSE:
            input("Ready to plan approach?")
            joint_state = rospy.wait_for_message(
                "/joint_states_all", JointState, timeout=5
            )
        else:
            joint_state.name = plan6.joint_names
            joint_state.position = plan6.points[-1].positions

except Exception as e:
    save_name = os.path.splitext(scene_json)[0] + ".txt"
    with open(f"{save_name}", "w") as f:
        print(e, file=f)
        print(f"Failed for obj_{i}", file=f)

img_msg = rospy.wait_for_message("/camera0/color/image_raw", Image, timeout=10)
image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
save_name = os.path.splitext(scene_json)[0] + ".png"
cv2.imwrite(f"{save_name}", image)
