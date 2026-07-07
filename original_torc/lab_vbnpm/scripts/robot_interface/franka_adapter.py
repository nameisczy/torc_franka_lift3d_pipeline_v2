from dataclasses import dataclass
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

try:
    from grasp_representation.canonical_grasp import CanonicalGrasp
except ImportError:
    from ..grasp_representation.canonical_grasp import CanonicalGrasp


@dataclass(frozen=True)
class RobotGraspCommand:
    tcp_contact_pose_world: np.ndarray
    tcp_pregrasp_pose_world: np.ndarray
    gripper_opening_width: float
    tcp_site: str
    collision_model: str


class RobotAdapter:
    tcp_site = "gripper0_right_grip_site"
    validated_open_width = 0.08
    opening_clearance_m = 0.012
    pregrasp_distance = 0.10
    # Positive retreat from the canonical contact point, along local -Z.
    # Derived from the current Panda pad geometry, with a small pad-thickness
    # clearance so the fingertip face approaches the surface instead of
    # placing the TCP inside it.
    tcp_contact_retreat_m = None
    tcp_lateral_calibration_m = 0.0
    tcp_approach_calibration_m = 0.0
    collision_model = "franka_panda_collision_spheres"

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[4]

    @classmethod
    def _derive_tcp_contact_retreat_m(cls) -> float:
        override = os.environ.get("TORC_FRANKA_TCP_CONTACT_RETREAT_M")
        if override is not None:
            return float(override)

        gripper_xml = (
            cls._project_root()
            / "assets/franka/mjcf/robosuite/panda_gripper.xml"
        )
        root = ET.parse(gripper_xml).getroot()

        def vec(node, attr):
            return np.fromstring(node.attrib[attr], sep=" ", dtype=np.float64)

        eef = root.find(".//body[@name='eef']")
        finger = root.find(".//body[@name='leftfinger']")
        tip = root.find(".//body[@name='finger_joint1_tip']")
        pad = root.find(".//geom[@name='finger1_pad_collision']")
        if any(node is None for node in (eef, finger, tip, pad)):
            raise RuntimeError(f"cannot derive Franka TCP retreat from {gripper_xml}")

        tcp_z = float(vec(eef, "pos")[2])
        pad_center_z = (
            float(vec(finger, "pos")[2])
            + float(vec(tip, "pos")[2])
            + float(vec(pad, "pos")[2])
        )
        pad_half_depth = float(vec(pad, "size")[2])
        pad_front_z = pad_center_z + pad_half_depth
        front_offset = pad_front_z - tcp_z
        extra_clearance = float(
            os.environ.get("TORC_FRANKA_PAD_APPROACH_CLEARANCE_M", "0.001")
        )
        return float(front_offset + extra_clearance)

    @classmethod
    def derive_tcp_pad_front_m(cls) -> float:
        gripper_xml = (
            cls._project_root()
            / "assets/franka/mjcf/robosuite/panda_gripper.xml"
        )
        root = ET.parse(gripper_xml).getroot()

        def vec(node, attr):
            return np.fromstring(node.attrib[attr], sep=" ", dtype=np.float64)

        eef = root.find(".//body[@name='eef']")
        finger = root.find(".//body[@name='leftfinger']")
        tip = root.find(".//body[@name='finger_joint1_tip']")
        pad = root.find(".//geom[@name='finger1_pad_collision']")
        if any(node is None for node in (eef, finger, tip, pad)):
            raise RuntimeError(f"cannot derive Franka pad front from {gripper_xml}")
        tcp_z = float(vec(eef, "pos")[2])
        pad_front_z = (
            float(vec(finger, "pos")[2])
            + float(vec(tip, "pos")[2])
            + float(vec(pad, "pos")[2])
            + float(vec(pad, "size")[2])
        )
        return float(pad_front_z - tcp_z)

    def _frame_from(self, approach_direction, grasp_axis):
        z_axis = np.asarray(approach_direction, dtype=float).reshape(3)
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_seed = np.asarray(grasp_axis, dtype=float).reshape(3)
        x_seed = x_seed / np.linalg.norm(x_seed)
        x_axis = x_seed - z_axis * float(np.dot(x_seed, z_axis))
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        return np.stack([x_axis, y_axis, z_axis], axis=1)

    def adapt(self, grasp: CanonicalGrasp) -> RobotGraspCommand:
        grasp.validate()
        contact = np.array(grasp.contact_pose, dtype=float, copy=True)
        contact[:3, :3] = self._frame_from(grasp.approach_direction, grasp.grasp_axis)
        retreat = (
            float(self.tcp_contact_retreat_m)
            if self.tcp_contact_retreat_m is not None
            else self._derive_tcp_contact_retreat_m()
        )
        contact[:3, 3] -= retreat * contact[:3, 2]
        lateral_offset = float(
            os.environ.get("TORC_FRANKA_TCP_LATERAL_OFFSET_M", self.tcp_lateral_calibration_m)
        )
        approach_offset = float(
            os.environ.get("TORC_FRANKA_TCP_APPROACH_OFFSET_M", self.tcp_approach_calibration_m)
        )
        contact[:3, 3] += lateral_offset * contact[:3, 0]
        contact[:3, 3] += approach_offset * contact[:3, 2]
        pregrasp = contact.copy()
        pregrasp[:3, 3] -= self.pregrasp_distance * contact[:3, 2]
        requested = float(grasp.opening_width)
        opening = min(max(requested + self.opening_clearance_m, 0.0), self.validated_open_width)
        if not math.isfinite(opening):
            raise ValueError("opening must be finite")
        return RobotGraspCommand(
            tcp_contact_pose_world=contact,
            tcp_pregrasp_pose_world=pregrasp,
            gripper_opening_width=opening,
            tcp_site=self.tcp_site,
            collision_model=self.collision_model,
        )
