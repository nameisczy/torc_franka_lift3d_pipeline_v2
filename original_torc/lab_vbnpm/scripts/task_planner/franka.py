import sys
import os
from pathlib import Path
from typing import Any

import yaml


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from robot_interface.franka_robot import FrankaRobot  # noqa: E402


PROJECT_ROOT = Path("/mnt/ssd/ziyaochen/torc_franka_lift3d_pipeline_v2")
FRANKA_URDF = PROJECT_ROOT / "assets/franka/urdf/franka_panda_phase4.urdf"
FRANKA_CUROBO_YAML = PROJECT_ROOT / "assets/franka/config/curobo/franka_panda.yml"


class FrankaTORCRobot(FrankaRobot):
    def __init__(self, is_sim: bool, ground_truth: bool = False):
        self.is_sim = bool(is_sim)
        self.ground_truth = bool(ground_truth)
        self.camera = ["camera0", "camera1"] if is_sim else ["zedm"]
        self.urdf = str(FRANKA_URDF)
        with FRANKA_CUROBO_YAML.open("r", encoding="utf-8") as f:
            self.curobo_config = yaml.safe_load(f)["robot_cfg"]
        base_pose = self.curobo_config.get("planning", {}).get("base_pose_world_m")
        if base_pose is not None:
            os.environ["TORC_FRANKA_BASE_POSE_WORLD"] = ",".join(str(float(v)) for v in base_pose[:3])
        self.curobo_config.pop("planning", None)
        self.ignore_collision_ee_links = [
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
            "panda_tcp",
        ]

    def init_perception_interface(self) -> Any:
        import rospy
        from perception.perception_fast import PerceptionInterface

        if self.ground_truth:
            mode = "gt"
        elif self.is_sim:
            mode = "cam"
        else:
            mode = "fs"
        if self.is_sim and not rospy.has_param("/workspace/pose"):
            return PerceptionInterface(
                self.camera,
                pose_x=0.9 - 0.29,
                pose_y=0 - 0.5,
                pose_z=0.90,
                size_x=0.58,
                size_y=1.0,
                size_z=0.5,
                resolution=0.0025,
                mode=mode,
                urdf_file=self.urdf,
            )
        pose_x, pose_y, pose_z = rospy.get_param("/workspace/pose", [0.55, 0.63, 1.05])
        size_x, size_y, size_z = rospy.get_param("/workspace/size", [0.4, 1.26, 0.52])
        padding = 0.01
        return PerceptionInterface(
            self.camera,
            pose_x - padding,
            pose_y - size_y + padding,
            pose_z + padding / 10,
            size_x,
            size_y - 2 * padding,
            size_z - 2 * padding,
            resolution=0.002,
            mode=mode,
            urdf_file=self.urdf,
        )

    def init_motion_planner(self, planner: str = "curobo", warmup: bool = True) -> Any:
        if planner != "curobo":
            from motion_planner.motion_planner import MotionPlanner

            return MotionPlanner(self.urdf, ["panda_tcp"])

        from motion_planner.curobo_planner import CuroboPlanner

        return CuroboPlanner(
            self.urdf,
            ["panda_tcp"],
            self.curobo_config,
            self.ignore_collision_ee_links,
            is_sim=self.is_sim,
            warmup=warmup,
        )

    def init_adapter_node(self, scene_xml: str):
        from execution_scene.franka_node import FrankaNode

        return FrankaNode()

    def init_execution_bridge(self, scene_xml: str, experiment_dir: str = None):
        from execution_scene.franka_interface import FrankaInterface

        return FrankaInterface()
