import sys
import re
import glob
import json
import random

import numpy as np
import mujoco_viewer
from dm_control import mjcf
from dm_control import mujoco

import rospkg


def tgt_obj_coordinates(obj, is_sim: bool):
    if is_sim:
        clearance = 0.028
        translate = np.array([0.73, 0.09, 0.33])
        scale1 = 0.037
        scale2 = 1
    else:
        clearance = 0.0
        translate = np.array([0.72, 0.135, 0.33])
        scale1 = 0.01
        scale2 = 1.1
    pos = scale1 * np.array(
        [
            obj["bottom_left_pos"]["y"] + 0.5 * obj["dimensions"]["depth_y"],
            scale2 * (obj["bottom_left_pos"]["x"] + 0.5 * obj["dimensions"]["width_x"]),
            obj["bottom_left_pos"]["z"] + 0.5 * obj["dimensions"]["height_z"],
            # obj["bottom_left_pos"]["z"] + 0.5 * obj["dimensions"]["depth_y"],
            # obj["bottom_left_pos"]["x"] + 0.5 * obj["dimensions"]["width_x"],
            # obj["bottom_left_pos"]["y"] + 0.5 * obj["dimensions"]["height_z"],
        ]
    )
    pos[1] *= -1

    shape = scale1 * np.array(
        [
            0.5 * obj["dimensions"]["width_x"] - clearance,
            0.5 * obj["dimensions"]["depth_y"] - clearance,
            0.5 * obj["dimensions"]["height_z"],
        ],
        dtype=np.float64,
    )

    pos += translate

    return pos, shape


def load_objects(world, object_json: str, is_sim: bool = True):
    scene_dict = None
    with open(object_json, "r") as f:
        scene_dict = json.load(f)
    if scene_dict is None:
        print("Failed to read object_json")
        return

    objects = scene_dict["objects"]
    print(objects)

    x_offset = 0
    max_x = 0.6
    floor_y = -0.3
    for i, obj in enumerate(objects):
        # print(obj)
        pos, shape = tgt_obj_coordinates(obj, is_sim)
        print(pos, shape)
        target_body = world.worldbody.add(
            "body", name=f"obj_{obj['id']}_tgt", pos=f"{pos[0]} {pos[1]} {pos[2]}"
        )

        # body.add("joint", type="free")
        color = f"{random.random()} {random.random()} {random.random()}"
        target_body.add(
            "geom",
            name=f"obj_{obj['id']}_tgt_vis",
            type="box",
            size=f"{shape[0]} {shape[1]} {shape[2]}",
            rgba=f"{color} 0.2",
            quat="0.707107 0 0 0.707107",
            contype="0",
            conaffinity="0",
            group="4",
        )

        # Place objects on floor in 2 lines:
        # Robot faces +X direction, so:
        if x_offset > max_x:
            x_offset = 0
            floor_y *= -1
        elif i > 0:
            x_offset += shape[0]
        floor_x = x_offset
        x_offset += shape[0] + 0.025

        if is_sim:
            floor_x -= 0.2  # Offset from robot base

        floor_z = shape[2]  # Half the object height to place on floor

        # Initial object rotated 90° around Y relative to target orientation
        # Target has quat="0.707107 0 0 0.707107", this adds 90° Y rotation
        obj_body = world.worldbody.add(
            "body",
            name=f"obj_{obj['id']}",
            pos=f"{floor_x} {floor_y} {floor_z}",
            quat="0.5 -0.5 0.5 0.5",
        )
        obj_body.add(
            "geom",
            name=f"obj_{obj['id']}_vis",
            type="box",
            size=f"{shape[0]} {shape[1]} {shape[2]}",
            rgba=f"{color} 1",
            quat="0.707107 0 0 0.707107",
            # friction="1 0.9 0.01",
            # mass=".0001",
            # density=".01",
            gap=".002",
        )
        obj_body.add(
            "site",
            name=f"obj_{obj['id']}_site",
            pos=f"-{shape[1]} 0 {shape[2]/2}",
            size="0.01",
            rgba="0 1 0 0",
        )
        obj_body.add("joint", type="free")

        world.equality.add(
            "weld",
            name=f"obj_{obj['id']}_weld",
            site1=f"obj_{obj['id']}_site",
            site2=f"suction_site",
            active="false",
        )

    world.worldbody.add("site", name="suction_site")


def init(scene_xml: str, scene_objects: str, gui: bool = True, is_sim: bool = True):
    world = mjcf.from_path(scene_xml)

    load_objects(world, scene_objects, is_sim)

    xml_str = re.sub("-[a-f0-9]+.obj", ".obj", world.to_xml_string())
    xml_str = re.sub('\/\/|<default class="\/".+|class="\/"', "", xml_str)
    xml_str = re.sub("    <\/default>", "", xml_str)
    xml_str = re.sub('<default class="main".+|class="main"', "", xml_str)
    print(xml_str)
    world = mujoco.MjModel.from_xml_string(xml=xml_str)
    data = mujoco.MjData(world)

    viewer = mujoco_viewer.MujocoViewer(world, data) if gui else None

    return world, data, viewer


def main():
    rp = rospkg.RosPack()
    base_path = rp.get_path("lab_vbnpm")
    default_xml = base_path + "/xmls/ur5e_small_shelf.xml"
    scene_xml = sys.argv[1] if len(sys.argv) > 1 else default_xml
    default_json = base_path + "/tests/robot_experiment_small.json"
    scene_json = sys.argv[2] if len(sys.argv) > 2 else default_json

    world, data, viewer = init(
        scene_xml, scene_json, gui=len(sys.argv) > 3, is_sim=len(sys.argv) <= 4
    )

    if viewer is not None:
        while viewer.is_alive:
            mujoco.mj_step(world, data)
            viewer.render()
        viewer.close()
    else:
        for i in range(1000):
            mujoco.mj_step(world, data)

    mujoco.mj_saveLastXML("temp.xml", world)


if __name__ == "__main__":
    main()
