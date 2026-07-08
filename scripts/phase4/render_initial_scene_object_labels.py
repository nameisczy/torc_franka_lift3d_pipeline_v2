#!/usr/bin/env python3
"""Render the initial scene with visible object body tags overlaid."""

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

from output_paths import artifact_path, result_path
import render_franka_asset_alignment as phase41


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = PROJECT_ROOT / "original_torc/lab_vbnpm/tests/scenes/final/difficult_116.xml"
PATCHED_XML = artifact_path("initial_scene_object_labels.xml")
OUT_PNG = result_path("initial_scene_object_labels.png")
OUT_MANIFEST = artifact_path("initial_scene_object_labels_manifest.json")
RENDER_WIDTH = 1400
RENDER_HEIGHT = 950
GEOM_OBJTYPE = int(mujoco.mjtObj.mjOBJ_GEOM)


def object_tags_from_xml(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    return [
        body.attrib["name"]
        for body in root.findall(".//body")
        if body.attrib.get("name", "").startswith("obj_")
    ]


def camera(mode: str) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if mode == "top":
        cam.lookat[:] = np.array([0.78, -0.02, 1.08], dtype=np.float64)
        cam.distance = 0.74
        cam.azimuth = 90.0
        cam.elevation = -89.0
    else:
        cam.lookat[:] = np.array([0.58, 0.02, 1.10], dtype=np.float64)
        cam.distance = 1.18
        cam.azimuth = 138.0
        cam.elevation = -18.0
    return cam


def geom_to_object_map(model: mujoco.MjModel, object_tags: list[str]) -> dict[int, str]:
    object_body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tag): tag
        for tag in object_tags
    }
    object_body_ids = {body_id: tag for body_id, tag in object_body_ids.items() if body_id >= 0}

    geom_to_obj: dict[int, str] = {}
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        cursor = body_id
        while cursor > 0:
            if cursor in object_body_ids:
                geom_to_obj[geom_id] = object_body_ids[cursor]
                break
            cursor = int(model.body_parentid[cursor])
    return geom_to_obj


def draw_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    used_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    text_box = draw.textbbox((0, 0), text)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    x = int(np.clip(anchor[0] - text_w // 2, 4, RENDER_WIDTH - text_w - 10))
    y = int(np.clip(anchor[1] - text_h - 10, 4, RENDER_HEIGHT - text_h - 10))

    def overlaps(box: tuple[int, int, int, int]) -> bool:
        return any(
            not (box[2] < other[0] or other[2] < box[0] or box[3] < other[1] or other[3] < box[1])
            for other in used_boxes
        )

    box = (x - 4, y - 3, x + text_w + 5, y + text_h + 4)
    step = text_h + 8
    for _ in range(30):
        if not overlaps(box):
            break
        y = int(np.clip(y + step, 4, RENDER_HEIGHT - text_h - 10))
        box = (x - 4, y - 3, x + text_w + 5, y + text_h + 4)

    draw.line((anchor[0], anchor[1], x + text_w // 2, y + text_h // 2), fill=(*color, 220), width=2)
    draw.rectangle(box, fill=(0, 0, 0, 175), outline=(*color, 255), width=2)
    draw.text((x, y), text, fill=(255, 255, 255, 255))
    used_boxes.append(box)
    return box


def render_labeled_view(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_tags: list[str],
    geom_to_obj: dict[int, str],
    mode: str,
) -> tuple[Image.Image, dict[str, dict[str, object]]]:
    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
    cam = camera(mode)
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam)
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    renderer.close()

    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, RENDER_WIDTH, 72), fill=(0, 0, 0, 145))
    draw.text((16, 12), f"Initial scene object tags ({mode} view)", fill=(255, 255, 255, 255))
    draw.text(
        (16, 40),
        "labels are placed at visible MuJoCo segmentation centroids",
        fill=(220, 240, 255, 255),
    )

    colors = [
        (255, 80, 80),
        (80, 220, 255),
        (255, 210, 70),
        (140, 255, 120),
        (255, 120, 230),
        (180, 160, 255),
    ]
    label_info: dict[str, dict[str, object]] = {}
    used_boxes: list[tuple[int, int, int, int]] = []
    seg_objid = seg[:, :, 0]
    seg_objtype = seg[:, :, 1]

    for index, tag in enumerate(object_tags):
        geom_ids = [geom_id for geom_id, geom_tag in geom_to_obj.items() if geom_tag == tag]
        if not geom_ids:
            label_info[tag] = {"visible_pixels": 0, "label_pixel": None}
            continue
        mask = (seg_objtype == GEOM_OBJTYPE) & np.isin(seg_objid, np.asarray(geom_ids, dtype=np.int32))
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            label_info[tag] = {"visible_pixels": 0, "label_pixel": None}
            continue
        anchor = (int(np.median(xs)), int(np.median(ys)))
        color = colors[index % len(colors)]
        draw.ellipse(
            (anchor[0] - 5, anchor[1] - 5, anchor[0] + 5, anchor[1] + 5),
            fill=(*color, 240),
            outline=(0, 0, 0, 255),
        )
        box = draw_label(draw, tag, anchor, color, used_boxes)
        label_info[tag] = {
            "visible_pixels": int(len(xs)),
            "label_pixel": [anchor[0], anchor[1]],
            "label_box": list(box),
            "geom_ids": geom_ids,
        }

    hidden = [tag for tag, info in label_info.items() if int(info["visible_pixels"]) == 0]
    draw.text(
        (16, RENDER_HEIGHT - 30),
        f"visible labels: {len(object_tags) - len(hidden)}/{len(object_tags)}"
        + (f"; hidden/fully occluded in this view: {', '.join(hidden)}" if hidden else ""),
        fill=(255, 245, 180, 255),
    )
    return image, label_info


def main() -> None:
    object_tags = object_tags_from_xml(SCENE_XML)
    phase41.TORC_SCENE_XML = SCENE_XML
    PATCHED_XML.parent.mkdir(parents=True, exist_ok=True)
    phase41.PATCHED_XML = PATCHED_XML
    phase41.patch_torc_scene()

    model = mujoco.MjModel.from_xml_path(str(PATCHED_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom_to_obj = geom_to_object_map(model, object_tags)

    side, side_info = render_labeled_view(model, data, object_tags, geom_to_obj, "side")
    top, top_info = render_labeled_view(model, data, object_tags, geom_to_obj, "top")

    canvas = Image.new("RGB", (RENDER_WIDTH * 2, RENDER_HEIGHT), (16, 16, 16))
    canvas.paste(side, (0, 0))
    canvas.paste(top, (RENDER_WIDTH, 0))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PNG)

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(
        json.dumps(
            {
                "output_png": str(OUT_PNG),
                "scene_xml": str(SCENE_XML),
                "patched_xml": str(PATCHED_XML),
                "object_tags": object_tags,
                "side_view": side_info,
                "top_view": top_info,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUT_PNG)


if __name__ == "__main__":
    main()
