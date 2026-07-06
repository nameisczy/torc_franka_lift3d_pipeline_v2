import sys
import os
from pathlib import Path
from typing import Any

import yaml


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from robot_interface.franka_robot import FrankaRobot  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[4]
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
            base_pose_env = ",".join(str(float(v)) for v in base_pose[:3])
            os.environ["TORC_FRANKA_MUJOCO_BASE_POSE_WORLD"] = base_pose_env
            os.environ["TORC_FRANKA_PLANNER_BASE_POSE_WORLD"] = base_pose_env
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

        for path in (str(SCRIPTS_ROOT), str(SCRIPTS_ROOT.parent)):
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)
        for module_name in list(sys.modules):
            if not (
                module_name == "motion_planner"
                or module_name.startswith("motion_planner.")
            ):
                continue
            module = sys.modules.get(module_name)
            module_file = getattr(module, "__file__", "") if module is not None else ""
            if module is not None and (
                not module_file or not module_file.startswith(str(SCRIPTS_ROOT))
            ):
                del sys.modules[module_name]

        from motion_planner.curobo_planner import CuroboPlanner

        motion_planner = CuroboPlanner(
            self.urdf,
            ["panda_tcp"],
            self.curobo_config,
            self.ignore_collision_ee_links,
            is_sim=self.is_sim,
            warmup=warmup,
        )
        return motion_planner

    def init_adapter_node(self, scene_xml: str):
        from execution_scene.franka_node import FrankaNode

        return FrankaNode()

    def init_execution_bridge(self, scene_xml: str, experiment_dir: str = None):
        from execution_scene.franka_interface import FrankaInterface

        return FrankaInterface()
