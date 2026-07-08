#!/usr/bin/env python3
"""Render the TORC perception target point cloud over the MuJoCo objects."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import render_franka_asset_alignment as phase41
from output_paths import artifact_path, result_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = (
    PROJECT_ROOT
    / "phase4_artifacts/torc_franka_pipeline_1783268501"
    / "exp_2026-07-05_20-21-53__difficult_116_obj_000070_0_dg_only"
)
POINTS_TXT = EXP_DIR / "output_target_points.txt"
DEBUG_NPZ = EXP_DIR / "selected_grasp_debug/pick_01_selected_grasp_debug.npz"
DEBUG_JSON = EXP_DIR / "selected_grasp_debug/pick_01_selected_grasp_debug.json"
SCENE_XML = PROJECT_ROOT / "original_torc/lab_vbnpm/tests/scenes/final/difficult_116.xml"
OVERLAY_XML = artifact_path("tsdf_pointcloud_overlay_scene.xml")
OUT_PNG = result_path("tsdf_pointcloud_object_overlay.png")
OUT_MANIFEST = artifact_path("tsdf_pointcloud_object_overlay_manifest.json")


def body_positions_from_xml(xml_path: Path) -> dict[str, np.ndarray]:
    root = ET.parse(xml_path).getroot()
    out = {}
    for body in root.findall(".//body"):
        name = body.attrib.get("name", "")
        if name.startswith("obj_"):
            out[name] = np.fromstring(body.attrib.get("pos", "0 0 0"), sep=" ", dtype=np.float64)
    return out


def add_sphere(scene: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba: list[float]) -> bool:
    if scene.ngeom >= len(scene.geoms):
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
    return True


def camera(lookat: np.ndarray, mode: str) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    if mode == "top":
        cam.distance = 0.72
        cam.azimuth = 90.0
        cam.elevation = -89.0
    else:
        cam.distance = 0.88
        cam.azimuth = 138.0
        cam.elevation = -24.0
    return cam


def render_view(model: mujoco.MjModel, data: mujoco.MjData, pts: np.ndarray, markers: dict[str, np.ndarray], mode: str) -> Image.Image:
    renderer = mujoco.Renderer(model, height=800, width=1100)
    renderer.update_scene(data, camera=camera(pts.mean(axis=0), mode))
    scene = renderer.scene

    rng = np.random.default_rng(7)
    max_points = min(len(pts), 3500)
    sample = pts[rng.choice(len(pts), max_points, replace=False)] if len(pts) > max_points else pts
    drawn = 0
    for p in sample:
        if add_sphere(scene, p, 0.0022, [0.0, 0.95, 1.0, 0.82]):
            drawn += 1
    add_sphere(scene, markers["point_centroid"], 0.018, [0.0, 0.1, 1.0, 1.0])
    add_sphere(scene, markers["selected_object_center"], 0.018, [1.0, 1.0, 0.0, 1.0])
    add_sphere(scene, markers["selected_tcp"], 0.015, [1.0, 0.05, 0.05, 1.0])

    image = Image.fromarray(renderer.render()).convert("RGB")
    renderer.close()
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, 1100, 86), fill=(0, 0, 0, 145))
    draw.text((16, 12), f"{mode} view: cyan target point cloud over MuJoCo objects", fill=(255, 255, 255, 255))
    draw.text((16, 38), "yellow: selected object body center; blue: point centroid; red: selected grasp TCP", fill=(255, 245, 180, 255))
    draw.text((16, 62), f"rendered points: {drawn}/{len(pts)}", fill=(220, 240, 255, 255))
    return image


def main() -> None:
    pts = np.loadtxt(POINTS_TXT, dtype=np.float64).reshape(-1, 3)
    debug = json.loads(DEBUG_JSON.read_text(encoding="utf-8"))
    selected_obj = debug["intended_object_name"]
    selected_tcp = np.load(DEBUG_NPZ)["selected_transformed_grasp_matrix"][:3, 3]
    body_positions = body_positions_from_xml(SCENE_XML)
    nearest = sorted(
        (
            (float(np.linalg.norm(pos - pts.mean(axis=0))), name, pos.tolist())
            for name, pos in body_positions.items()
        ),
        key=lambda item: item[0],
    )

    phase41.TORC_SCENE_XML = SCENE_XML
    phase41.patch_torc_scene()
    OVERLAY_XML.parent.mkdir(parents=True, exist_ok=True)
    OVERLAY_XML.write_text(phase41.PATCHED_XML.read_text(encoding="utf-8"), encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(OVERLAY_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    markers = {
        "point_centroid": pts.mean(axis=0),
        "selected_object_center": body_positions[selected_obj],
        "selected_tcp": selected_tcp,
    }
    side = render_view(model, data, pts, markers, "side")
    top = render_view(model, data, pts, markers, "top")
    canvas = Image.new("RGB", (2200, 800), (255, 255, 255))
    canvas.paste(side, (0, 0))
    canvas.paste(top, (1100, 0))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PNG)

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(
        json.dumps(
            {
                "output_png": str(OUT_PNG),
                "points_txt": str(POINTS_TXT),
                "point_count": int(len(pts)),
                "point_min_world": pts.min(axis=0).tolist(),
                "point_max_world": pts.max(axis=0).tolist(),
                "point_centroid_world": pts.mean(axis=0).tolist(),
                "selected_object": selected_obj,
                "selected_object_center_world": body_positions[selected_obj].tolist(),
                "selected_tcp_world": selected_tcp.tolist(),
                "distance_centroid_to_selected_object_center_m": float(
                    np.linalg.norm(pts.mean(axis=0) - body_positions[selected_obj])
                ),
                "nearest_object_centers_to_point_centroid": [
                    {"object": name, "distance_m": dist, "center_world": pos}
                    for dist, name, pos in nearest[:12]
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUT_PNG)


if __name__ == "__main__":
    main()
