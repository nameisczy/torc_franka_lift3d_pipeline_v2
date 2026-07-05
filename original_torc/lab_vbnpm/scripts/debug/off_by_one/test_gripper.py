import time
from typing import Set
import mujoco
import mujoco.viewer
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_XML = os.path.join(SCRIPT_DIR, "robotiq_2f85/scene.xml")

model = mujoco.MjModel.from_xml_path(os.path.abspath(MODEL_XML))
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)
gripper_geom_ids = [
    model.geom("left_pad1").id,
    model.geom("left_pad2").id,
    model.geom("right_pad1").id,
    model.geom("right_pad2").id,
]


def get_gripped_objs() -> Set[str]:
    grasping = set()
    for g1, g2 in data.contact.geom:
        gripper_collision = g1 in gripper_geom_ids or g2 in gripper_geom_ids
        if gripper_collision:
            # print(f"gripper_collision: {model.geom(g1).name} <-> {model.geom(g2).name}")
            for g in (g1, g2):
                if g in gripper_geom_ids:
                    continue
                objid = model.geom(g2).bodyid[0]
                name = model.geom(objid).name
                if len(name) == 0:
                    name = f"body_id_{objid}"
                grasping.add(name)
    return grasping


curr_gripped_objs = set()

while viewer.is_running():
    viewer.sync()
    mujoco.mj_step(model, data)

    gripped_objs = get_gripped_objs()
    if curr_gripped_objs != gripped_objs:
        now = time.time()
        print(f"[{now}]: {' '.join(gripped_objs)}")
        # print(f"  DIFF: {gripped_objs - curr_gripped_objs}")
        curr_gripped_objs = gripped_objs
