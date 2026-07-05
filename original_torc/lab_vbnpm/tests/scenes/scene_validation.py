import os
import cv2
import sys
import rospy
import rospkg
import numpy as np
import open3d as o3d
import trimesh as tm
import transformations as tf

import time
import mujoco
from tracikpy import TracIKSolver

import torch

from curobo.types.robot import JointState
from curobo.types.base import TensorDeviceType
from curobo.geom.types import WorldConfig, Mesh
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.types.robot import RobotConfig
from curobo.types.file_path import ContentPath
from curobo.cuda_robot_model.util import load_robot_yaml

from load_gc6d_grasps import load_grasps
from graspclutter6dAPI import GraspClutter6D
from grasp_planner.curobo_grasp_planner import GraspPlanner

from utils.print_color import *

from geometry_msgs.msg import Transform, Vector3, Quaternion, Pose, PoseStamped

# Set the path to the GraspClutter6D API
graspclutter_mujoco_path = "/data/local/kc1317/graspclutter6d_mujoco_sim"

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python scene_validation.py <scene> <structure> <scene_idx>")
        exit(1)

    if sys.argv[1] in ["s", "shelf"]:
        scene = "shelf"
    elif sys.argv[1] in ["t", "tabletop"]:
        scene = "tabletop"
    # else:
    #     print("Invalid scene type. Use 's' for shelf or 't' for tabletop.")
    #     exit(1)

    if sys.argv[2] in ["s", "structured"]:
        structure = "structured"
    elif sys.argv[2] in ["u", "unstructured"]:
        structure = "unstructured"
    # else:
    #     print("Invalid structure type. Use 's' for structured or 'u' for unstructured.")
    #     exit(1)

    category = f"{scene}_{structure}"
    scene_idx = int(sys.argv[3])

    if len(sys.argv) > 4:
        out_file = sys.argv[4]
    else:
        out_file = None

    if "GC6D_ROOT" not in os.environ:
        print(
            "Please set the environment variable GC6D_ROOT (e.g. export GC6D_ROOT=/path/to/GraspClutter6D)"
        )
        exit(0)

    gc6d_path = os.environ["GC6D_ROOT"]

    ## Get all objects in the scene, as well as their transforms

    # scene_idx = 120

    print("Scene idx:", scene_idx)

    rp = rospkg.RosPack()
    lab_path = rp.get_path("lab_vbnpm")

    category_path = f"{lab_path}/tests/scenes/{category}/"

    scene_path = f"{category_path}/scene{scene_idx}.xml"

    if not os.path.exists(scene_path):
        print(f"Scene {scene_idx} does not exist in category {category}.")
        print(f"Empty path: {scene_path}")
        exit(0)

    m = mujoco.MjModel.from_xml_path(scene_path)
    d = mujoco.MjData(m)

    # GraspClutter6D objects are expected to be named obj_{id:06d}_{instance}
    # Collect all objects that start with "obj_"

    obj_names = []
    for i in range(m.nbody):
        if m.body(i).name.startswith("obj_"):
            # print(f"Object {i}: {m.body(i).name}")
            obj_names.append(m.body(i).name)

    # dict: object name -> GC6D id

    object_name2idx = {obj_name: int(obj_name.split("_")[1]) for obj_name in obj_names}
    unique_objects = set(object_name2idx.values())

    # dict: object name -> transform

    def pos_quat_to_transform(pos, quat):
        """Convert position and quaternion to a transformation matrix."""
        transform = tf.quaternion_matrix(quat)
        transform[:3, 3] = pos

        # print("TAG", pos, quat)
        return transform

    object_name2transform = {
        obj_name: pos_quat_to_transform(m.body(obj_name).pos, m.body(obj_name).quat)
        for i, obj_name in enumerate(obj_names)
    }

    ## Load grasps for all objects in the scene

    # dict: object id -> grasps

    print("Unique objects:", unique_objects)
    object_grasps = {}
    for obj_idx in unique_objects:
        # object_index = object_name2idx[obj_name]
        # print(f"Loading grasps for object {obj_idx}")

        # Check if the grasps file exists
        grasps_file = f"{lab_path}/tests/scenes/grasps/grasps_{obj_idx:06d}.npy"
        if not os.path.exists(grasps_file):
            object_grasps[obj_idx] = None
        else:
            object_grasps[obj_idx] = np.load(
                f"{lab_path}/tests/scenes/grasps/grasps_{obj_idx:06d}.npy",
                allow_pickle=True,
            )

    ## Obtain list of candidate target objects in the scene

    # print(object_grasps)

    # list: [candidate target object names]

    # read txt

    txt_path = f"{graspclutter_mujoco_path}/scenes/{category}/{category}.txt"

    # split by newlines

    with open(txt_path, "r") as f:
        txt_lines = f.read().splitlines()

    scene2candidates = {}
    for line in txt_lines:
        entries = line.split(" ")
        if len(entries) < 2:
            continue
        elif not entries[1].startswith("obj_"):
            continue

        scene_id = int(entries[0])
        obj_names = entries[1:]

        scene2candidates[scene_id] = obj_names

    ## For each candidate target object, define a rectangle in front of it, and obtain all objects in the rectangle

    # max: [0, 0.2]
    # min: [-0.2, -0.2]

    def pose_to_matrix(pose_msg):
        translation = [
            pose_msg.position.x,
            pose_msg.position.y,
            pose_msg.position.z,
        ]
        quaternion = [
            pose_msg.orientation.w,
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
        ]
        transform = tf.quaternion_matrix(quaternion)
        transform[:3, 3] = translation
        return transform

    def matrix_to_pose(matrix):
        translation = tf.translation_from_matrix(matrix)
        quaternion = tf.quaternion_from_matrix(matrix)
        pose_msg = Pose()
        pose_msg.position.x = translation[0]
        pose_msg.position.y = translation[1]
        pose_msg.position.z = translation[2]
        pose_msg.orientation.w = quaternion[0]
        pose_msg.orientation.x = quaternion[1]
        pose_msg.orientation.y = quaternion[2]
        pose_msg.orientation.z = quaternion[3]
        return pose_msg

    def get_transformed_mesh(obj_id, transform):
        mesh_path = f"{gc6d_path}/models_obj_m/obj_{obj_id:06d}/obj_{obj_id:06d}.obj"
        mesh = tm.load_mesh(mesh_path)
        mesh.apply_transform(transform)

        return mesh

    # list: [candidate object id, all objects in rectangle]

    # dict: candidate object id name -> list of objects in rectangle
    candidate2objects = {}
    for candidate_obj_name in scene2candidates[scene_idx]:
        candidate_obj_id = object_name2idx[candidate_obj_name]
        candidate_transform = object_name2transform[candidate_obj_name]

        print(f"Candidate object: {candidate_obj_name} (id: {candidate_obj_id})")
        print(f"Transform:\n{candidate_transform}")

        # Define a rectangle in front of the candidate object
        max_inc = [0.05, 0.2]
        min_inc = [-1, -0.2]
        z_epsilon = 0.04

        # Iterate through objects in the scene and check if they are within the rectangle
        objects_in_rectangle = []

        candidate_obj_mesh = get_transformed_mesh(candidate_obj_id, candidate_transform)

        for obj_name, obj_transform in object_name2transform.items():
            obj_id = object_name2idx[obj_name]

            # Skip the candidate object itself
            if obj_id == candidate_obj_id:
                continue

            # Check if the object is within the rectangle
            obj_pos = obj_transform[:3, 3]
            candidate_pos = candidate_transform[:3, 3]

            # Check if the object is within the rectangle in front of the candidate object
            if (
                candidate_pos[0] + min_inc[0]
                <= obj_pos[0]
                <= candidate_pos[0] + max_inc[0]
                and candidate_pos[1] + min_inc[1]
                <= obj_pos[1]
                <= candidate_pos[1] + max_inc[1]  # and
                # candidate_pos[2] - 0.2 <= obj_pos[2] <= candidate_pos[2] + 0.2
            ):

                obj_mesh = get_transformed_mesh(obj_id, obj_transform)

                # If object's z_max is greater than candidate's z_max + epsilon, add it
                if obj_mesh.bounds[1][2] > candidate_obj_mesh.bounds[1][2] + z_epsilon:
                    objects_in_rectangle.append(obj_name)
                else:
                    # Object is not high enough. Skip it.
                    pass

        candidate2objects[candidate_obj_name] = objects_in_rectangle
        print(
            f"Objects in rectangle for candidate {candidate_obj_name} (id: {candidate_obj_id}): {objects_in_rectangle}"
        )

    ## Check that at least one IK feasible, collision-free grasp exists for each object in the list
    curobo_root = f"{lab_path}/robots/motoman/curobo/"
    content_path = ContentPath(
        robot_config_absolute_path=curobo_root + "motoman.yml",
        robot_urdf_absolute_path=curobo_root + "motoman.urdf",
        robot_usd_absolute_path=curobo_root + "motoman.usd",
        robot_asset_absolute_path=curobo_root,
    )
    robot_dict = load_robot_yaml(content_path)
    curobo_config = robot_dict["robot_cfg"]

    pose, size = (
        [0.6, 0.65, 1.0],
        [0.4, 1.3, 0.5],
        # [0.8017768655705504, 0.6474097242263454, 0.8897288837175601],
        # [0.6779224837368386, 1.3143877218440854, 0.5593604978067703],
    )
    padding = 0.1
    thick = 0.05
    size_top = [size[0], size[1] + 5 * thick, thick]
    size_bottom = [size[0], size[1] + 2 * thick, pose[2] + thick]
    size_left = [size[0], 3 * thick, size[2] + pose[2] + size_top[2]]
    size_right = [size[0], 3 * thick, size[2] + pose[2] + size_top[2]]
    size_back = [thick, size[1], size[2]]
    pose_top = [
        pose[0] + size[0] - size_top[0] / 2,
        pose[1] - size[1] / 2,
        pose[2] + size[2] + 1.5 * size_top[2],
    ]
    pose_bottom = [
        pose[0] + size_bottom[0] / 2,
        pose[1] - size[1] / 2,
        pose[2] - 0.5 * size_bottom[2] + thick,
    ]
    pose_left = [
        pose[0] + size_left[0] / 2,
        pose[1] + size_left[1] / 2,
        size_left[2] / 2,
    ]
    pose_right = [
        pose[0] + size_right[0] / 2,
        pose[1] - size[1] - size_right[1] / 2,
        size_right[2] / 2,
    ]
    pose_back = [
        pose[0] + size[0] + size_back[0] / 2,
        pose[1] - size[1] / 2,
        pose[2] + size[2] / 2,
    ]

    world_config = {
        # cuboid:
        #   name:
        #       dims: x, y, z
        #       pose: x, y, z, qw, qx, qy, qz
        "cuboid": {
            "shelf_top": {
                "pose": [*pose_top, 1, 0, 0, 0],
                "dims": np.add(size_top, [padding, padding, 0.5 * padding]),
            },
            "shelf_bottom": {
                "pose": [*pose_bottom, 1, 0, 0, 0],
                "dims": np.add(size_bottom, [padding, padding, 0]),
            },
            "shelf_left": {
                "pose": [*pose_left, 1, 0, 0, 0],
                "dims": np.add(size_left, padding),
            },
            "shelf_right": {
                "pose": [*pose_right, 1, 0, 0, 0],
                "dims": np.add(size_right, padding),
            },
            "shelf_back": {
                "pose": [*pose_back, 1, 0, 0, 0],
                "dims": np.add(size_back, padding),
            },
        },
        # "voxel": {
        #     "base": {
        #         "dims": size,
        #         "pose": [0, 0, 0, 0, 1, 0, 0, 0],
        #         "voxel_size": resolution,
        #         "feature_dtype": torch.bfloat16,
        #     },
        # }
    }
    urdf = f"{lab_path}/robots/motoman/curobo/motoman.urdf"

    ignore_collision_ee_links = [
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

    robot_config = RobotWorldConfig.load_from_config(
        curobo_config,
        world_config,
        collision_activation_distance=0.0,
    )

    robot_world = RobotWorld(robot_config)
    ik_solver = TracIKSolver(urdf, "base_link", "motoman_right_ee")

    def valid_pose(pose, qinit=None):

        if not isinstance(pose, np.ndarray):
            pose = pose_to_matrix(pose)
        js = ik_solver.ik(pose, qinit=qinit)
        if js is None:
            for i in range(3):
                js = ik_solver.ik(pose)
            if js is None:
                return False

        # print("IK success")

        joint_state_tensor = JointState.from_position(
            torch.tensor(np.array(js)[np.newaxis, :], dtype=torch.float32).cuda(),
            joint_names=list(ik_solver.joint_names),
        )
        joint_state_tensor = robot_world.get_active_js(joint_state_tensor)
        q = joint_state_tensor.position
        # collision check
        res = robot_world.get_world_self_collision_distance_from_joints(q)
        d_world, d_self = res
        # print(d_self, d_world)
        valid = ((d_world <= 0) & (d_self <= 0)).cpu().numpy()

        # print("[k_user] ================= valid_pose", valid)
        return valid[0]

    candidate_obj_name2success = {}
    obj_name2graspable = {}
    for candidate_obj_name, objects_in_rectangle in candidate2objects.items():
        printBlue(
            f"Candidate object name: {candidate_obj_name}, objects in rectangle: {objects_in_rectangle}"
        )

        res = True

        objects = [candidate_obj_name] + objects_in_rectangle

        for obj_name in objects:
            obj_idx = object_name2idx[obj_name]
            # print(obj_idx, obj_name)

            # Check if the object has already been marked as graspable or not
            if obj_name in obj_name2graspable:
                if obj_name2graspable[obj_name]:
                    print(
                        f"[SUCCESS] Object {obj_name}, ID {obj_idx} is already marked as graspable."
                    )
                    continue
                else:
                    print(
                        f"[FAILURE] Object {obj_name}, ID {obj_idx} is already marked as not graspable."
                    )
                    res = False
                    break

            if object_grasps[obj_idx] is None:
                print(f"No grasps found for object {obj_idx}.")
                break

            obj_res = False

            obj_transform = object_name2transform[obj_name]
            for grasp in object_grasps[obj_idx]:

                # print(obj_name)
                # print(object_name2transform[obj_name])
                grasp = obj_transform @ grasp

                for z_rotation in [
                    0,
                    np.pi,
                    # np.pi / 2,
                    # 3 * np.pi / 2
                ]:
                    z_quat = np.concatenate(
                        [[np.cos(z_rotation)], np.sin(z_rotation) * np.array([0, 0, 1])]
                    )
                    z_transform = tf.quaternion_matrix(z_quat)

                    # Obtain rotated end-effector pose
                    rotated_grasp = grasp @ z_transform

                    if valid_pose(rotated_grasp):
                        obj_res = True
                        obj_name2graspable[obj_name] = True
                        print(
                            f"[SUCCESS] Valid grasp found for object {obj_name}, ID {obj_idx}."
                        )
                        break

                if obj_res:
                    break

            if not obj_res:
                print(
                    f"[FAILURE] No valid grasp found for object {obj_name}, ID {obj_idx}."
                )
                obj_name2graspable[obj_name] = False
                res = False
                break

        candidate_obj_name2success[candidate_obj_name] = res
        if res:
            printGreen(f"[SUCCESS] Candidate object {candidate_obj_name} validated.")
        else:
            printRed(f"[FAILURE] Candidate object {candidate_obj_name} invalidated.")

    print("Candidate object name -> success dictionary:", candidate_obj_name2success)

    if out_file:
        # scene index: [list of validated target objects]
        target_objects = [
            candidate_obj_name
            for candidate_obj_name, success in candidate_obj_name2success.items()
            if success
        ]
        with open(out_file, "w") as f:
            f.write(f"{scene_idx}: {target_objects}\n")
