import os
import sys
import rospy
import numpy as np
import transformations as tf
from dm_control import mjcf

from curobo.geom.types import WorldConfig
from curobo.types.robot import RobotConfig
from curobo.types.file_path import ContentPath
from curobo.cuda_robot_model.util import load_robot_yaml

from rospkg import RosPack
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from lab_vbnpm.srv import ExecuteTrajectory

from task_planner.ur5e import Ur5e
from task_planner.eutils import ee_suction_on, ee_suction_off, execute


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

    return obj_stats, static_stats


# Get object information from scene
obj_info = {}
static_info = {}
if len(sys.argv) > 1:
    is_sim = True
    scene_xml = sys.argv[1]
    if os.path.isfile(scene_xml):
        is_sim = False
        obj_info, static_info = get_object_info(
            scene_xml,
            1,
            # len(placement_order),
        )

rospy.init_node("reset")
is_sim = True if len(sys.argv) > 1 else False
## instatiate planner ##
robot = Ur5e(is_sim, False)
# perception = robot.init_perception_interface()
planner = robot.init_motion_planner(planner="curobo")  # ,warmup=False)
# grasp_planner = GraspPlanner(
#     robot.curobo_config,
#     planner.static_world_config,
#     robot.urdf,
#     ignore_collision_ee_links=robot.ignore_collision_ee_links,
# )

joint_state = rospy.wait_for_message("/joint_states_all", JointState, timeout=5)

goal = {
    "shoulder_pan_joint": 0,
    "shoulder_lift_joint": -2.2,
    "elbow_joint": 1.9,
    "wrist_1_joint": -1.383,
    "wrist_2_joint": -1.57,
    "wrist_3_joint": 0.00,
}

world_dict = {
    "cuboid": {
        "plane_floor": {
            "pose": [0, 0, -0.01, 1, 0, 0, 0],
            "dims": [10, 10, 0.01],
        },
        "plane_back": {
            "pose": [-0.59, 0, 0, 1, 0, 0, 0],
            "dims": [0.01, 5, 5],
        },
    },
}

world_dict["cuboid"].update(static_info)
world_dict["cuboid"].update(obj_info)
planner.world_config = WorldConfig.from_dict(world_dict)
planner.update_world_motion_gen()
planner.visualize_rviz()
planner.visualize_spheres_rviz(joint_state)
plan = planner.joint_motion_plan(joint_state, goal)

rospy.set_param("/robot/acc_ang_lim", 687)
rospy.set_param("/robot/vel_ang_lim", 20)
input("Suction off?")
ee_suction_off()
input("Execute?")
execute(plan, window=0, retime=True)
