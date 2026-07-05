from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class SceneObjectObservation:
    name: str
    object_id: int
    pose_world: np.ndarray
    visible_points_world: np.ndarray
    occluded_points_world: np.ndarray
    all_points_world: np.ndarray
    color: np.ndarray
    depth_order: float


@dataclass(frozen=True)
class PerceptionScene:
    objects: tuple
    support_top_z: float
    all_points_world: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    obj_mask: np.ndarray
    all_pts: np.ndarray
    all_rgb: np.ndarray
    all_mask: np.ndarray
    object_depths: dict
    object_id_to_name: dict
    object_name_to_id: dict


def _vec(text, default=(0.0, 0.0, 0.0)):
    if text is None:
        return np.asarray(default, dtype=float)
    return np.asarray([float(x) for x in text.split()], dtype=float)


def _body_pose(body):
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = _vec(body.get("pos"))
    return pose


def _geom_samples(body_pose: np.ndarray, geom) -> np.ndarray:
    center = body_pose[:3, 3] + _vec(geom.get("pos"))
    size = _vec(geom.get("size"), default=(0.02, 0.02, 0.02)).reshape(-1)
    sx = float(size[0]) if len(size) > 0 else 0.02
    sy = float(size[1]) if len(size) > 1 else sx
    sz = float(size[2]) if len(size) > 2 else sx
    offsets = [np.zeros(3, dtype=float)]
    for axis, extent in enumerate((sx, sy, sz)):
        delta = np.zeros(3, dtype=float)
        delta[axis] = extent
        offsets.extend((delta, -delta))
    for ox in (-sx, 0.0, sx):
        for oy in (-sy, 0.0, sy):
            for oz in (-sz, 0.0, sz):
                offsets.append(np.asarray([ox, oy, oz], dtype=float))
    return np.asarray([center + offset for offset in offsets], dtype=float)


def _object_color(object_id: int) -> np.ndarray:
    palette = np.asarray(
        [
            [0.82, 0.21, 0.18],
            [0.12, 0.48, 0.78],
            [0.19, 0.63, 0.33],
            [0.93, 0.58, 0.13],
            [0.54, 0.31, 0.70],
            [0.13, 0.68, 0.72],
        ],
        dtype=float,
    )
    return palette[int(object_id) % len(palette)]


def capture_scene_from_mujoco_xml(xml_path, object_prefix="obj_"):
    root = ET.parse(Path(xml_path)).getroot()
    objects = []
    points = []
    colors = []
    masks = []
    object_bodies = [body for body in root.findall(".//body") if (body.get("name", "")).startswith(object_prefix)]
    object_bodies = sorted(object_bodies, key=lambda body: body.get("name", ""))
    for compact_id, body in enumerate(object_bodies):
        name = body.get("name", "")
        pose = _body_pose(body)
        samples = []
        for geom in body.findall(".//geom"):
            samples.extend(_geom_samples(pose, geom))
        all_obj_points = np.asarray(samples or [pose[:3, 3]], dtype=float).reshape(-1, 3)
        median_z = float(np.median(all_obj_points[:, 2]))
        visible_mask = all_obj_points[:, 2] >= median_z - 1.0e-6
        visible = all_obj_points[visible_mask]
        occluded = all_obj_points[~visible_mask]
        color = _object_color(compact_id)
        points.append(visible)
        colors.append(np.repeat(color.reshape(1, 3), len(visible), axis=0))
        masks.append(np.full((len(visible),), compact_id, dtype=np.int32))
        objects.append(
            SceneObjectObservation(
                name=name,
                object_id=compact_id,
                pose_world=pose,
                visible_points_world=visible,
                occluded_points_world=occluded,
                all_points_world=all_obj_points,
                color=color,
                depth_order=float(pose[2, 3]),
            )
        )
    table_tops = []
    for geom in root.findall(".//geom"):
        name = (geom.get("name") or "").lower()
        if "table" in name or "shelf_bottom" in name:
            pos = _vec(geom.get("pos"))
            size = _vec(geom.get("size"), default=(0.0, 0.0, 0.0))
            if len(size) >= 3:
                table_tops.append(float(pos[2] + size[2]))
    all_points = np.vstack(points) if points else np.zeros((0, 3), dtype=float)
    all_colors = np.vstack(colors) if colors else np.zeros((0, 3), dtype=float)
    all_masks = np.concatenate(masks) if masks else np.zeros((0,), dtype=np.int32)
    object_depths = {obj.object_id: obj.depth_order for obj in objects}
    id_to_name = {obj.object_id: obj.name for obj in objects}
    name_to_id = {obj.name: obj.object_id for obj in objects}
    return PerceptionScene(
        objects=tuple(objects),
        support_top_z=max(table_tops) if table_tops else 0.0,
        all_points_world=all_points,
        points=all_points,
        colors=all_colors,
        obj_mask=all_masks,
        all_pts=all_points,
        all_rgb=all_colors,
        all_mask=all_masks,
        object_depths=object_depths,
        object_id_to_name=id_to_name,
        object_name_to_id=name_to_id,
    )
