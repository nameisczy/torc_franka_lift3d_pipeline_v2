import os
import sys
import json
import base64

import cv2
import numpy as np

# import supervision as sv
from openai import OpenAI

from utils.visual_utils import from_color_map


class Prompter:

    def __init__(self, url="http://lab.cs.lab.edu:4000/v1", key=""):
        # Initialize Prompting
        if url:
            self.client = OpenAI(api_key=key, base_url=url)
            if url.count("google") > 0:
                # model = 'gemini-2.5-pro'
                model = "gemini-robotics-er-1.5-preview"
            else:
                model = self.client.models.list().model_dump()["data"][0]["id"]
            self.model = model

        filedir = os.path.dirname(__file__)
        filepath = os.path.join(filedir, "prompt_templates", "system.txt")
        with open(filepath) as f:
            self.system_text = f.read()

        # Initialize Annotators
        # self.box_annotator = sv.BoxAnnotator(
        #     color_lookup=sv.ColorLookup.INDEX,
        #     thickness=2,
        # )
        # self.mask_annotator = sv.MaskAnnotator(
        #     color_lookup=sv.ColorLookup.INDEX,
        #     opacity=0.25,
        # )
        # self.polygon_annotator = sv.PolygonAnnotator(
        #     color_lookup=sv.ColorLookup.INDEX,
        #     thickness=2,
        # )
        # self.label_annotator = sv.LabelAnnotator(
        #     color=sv.Color.BLACK,
        #     text_color=sv.Color.WHITE,
        #     color_lookup=sv.ColorLookup.INDEX,
        #     text_position=sv.Position.CENTER_OF_MASS,
        #     text_padding=3,
        #     text_thickness=2,
        #     text_scale=0.6
        # )

    def prompt_model(self, text, images=[]):

        base64_images = []
        for img in images:
            retval, buffer = cv2.imencode(".jpg", img)
            img_b64 = base64.b64encode(buffer).decode("utf-8")
            base64_images.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                    },
                }
            )

        messages = [
            # {
            #     "role": "system",
            #     "content": [
            #         {
            #             "type": "text",
            #             "text": self.system_text
            #         },
            #     ]
            # },
            {
                "role": "user",
                "content": [*base64_images, {"type": "text", "text": text}],
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            reasoning_effort="medium",
        )
        return response.choices[0].message.content

    # def create_labeled_image(self, image, mask):

    #     labels = []
    #     masks = []
    #     for bit in range(32):
    #         binary_mask = ((mask >> bit) & 1).astype(bool)
    #         if np.count_nonzero(binary_mask) == 0:
    #             continue
    #         # print(binary_mask.sum())
    #         if binary_mask.sum() < 500:
    #             continue
    #         labels.append(bit)
    #         masks.append(binary_mask)

    #     masks = np.array(masks)
    #     marks = sv.Detections(
    #         mask=masks,
    #         xyxy=sv.mask_to_xyxy(masks=masks),
    #         class_id=np.asarray(labels),
    #     )

    #     with_box = False
    #     with_mask = True
    #     with_polygon = True
    #     with_label = True

    #     annotated_image = image.copy()
    #     if with_box:
    #         annotated_image = self.box_annotator.annotate(
    #             scene=annotated_image, detections=marks
    #         )
    #     if with_mask:
    #         annotated_image = self.mask_annotator.annotate(
    #             scene=annotated_image, detections=marks
    #         )
    #     if with_polygon:
    #         annotated_image = self.polygon_annotator.annotate(
    #             scene=annotated_image, detections=marks
    #         )
    #     if with_label:
    #         annotated_image = self.label_annotator.annotate(
    #             scene=annotated_image, detections=marks
    #         )
    #     return annotated_image


def boxes_overlap(box1, box2):
    """Check if two rectangles overlap."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (
        x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min
    )


def find_non_overlapping_position(
    cx, cy, box_w, box_h, occupied_boxes, max_attempts=50
):
    """
    Attempts to find a position near (cx, cy) where the box won't overlap with others.
    """
    directions = [
        (0, 0),
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0),
        (1, -1),
        (1, 1),
        (-1, 1),
        (-1, -1),
    ]
    spacing = 15  # pixels to shift per step

    for attempt in range(1, max_attempts + 1):
        for dx, dy in directions:
            new_cx = cx + dx * spacing * attempt
            new_cy = cy + dy * spacing * attempt

            top_left = (new_cx - box_w // 2, new_cy - box_h // 2)
            bottom_right = (top_left[0] + box_w, top_left[1] + box_h)
            candidate_box = (*top_left, *bottom_right)

            if all(
                not boxes_overlap(candidate_box, existing)
                for existing in occupied_boxes
            ):
                return top_left, bottom_right

    return None, None  # fallback if all attempts fail


def create_labeled_image(
    image: np.ndarray, depth: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    Highlights each bit-segment in the mask and labels it on the image.

    Parameters:
    - image: HxWx3 uint8 input image
    - mask:  HxW uint32 segmentation mask where each bit represents a different label

    Returns:
    - Annotated image
    - Object Labels
    """

    # sort bit indices by minimum depth
    min_depths = []
    max_depths = []
    avg_depths = []
    valid_indices = []
    for i in range(32):
        mask_i = (mask & (1 << i)).astype(bool)
        if np.count_nonzero(mask_i) == 0:
            continue
        min_d = np.min(depth[mask_i])
        max_d = np.max(depth[mask_i])
        avg_d = np.mean(depth[mask_i])
        min_depths.append(min_d)
        max_depths.append(max_d)
        avg_depths.append(avg_d)
        valid_indices.append(i)

    # Sort indices by corresponding minimum depths
    sorted_order = np.argsort(min_depths)
    sorted_order = list(reversed(sorted_order))
    sorted_indices = [valid_indices[i] for i in sorted_order]
    d_min_depths = {valid_indices[i]: min_depths[i] for i in sorted_order}
    d_max_depths = {valid_indices[i]: max_depths[i] for i in sorted_order}
    d_avg_depths = {valid_indices[i]: avg_depths[i] for i in sorted_order}

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 2
    contour_thickness = 4

    output = image.copy()

    if len(output.shape) == 2 or output.shape[2] == 1:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    height, width = mask.shape
    np.random.seed(42)
    colors = 255 * from_color_map(range(32), 32)
    r = colors[:, 0].copy()
    b = colors[:, 2].copy()
    colors[:, 0] = b
    colors[:, 2] = r
    occupied_boxes = []

    # List to store contours info (used for text placement after drawing contours)
    largest_contours = []

    for bit in sorted_indices:
        binary_mask = ((mask >> bit) & 1).astype(np.uint8)
        # if np.count_nonzero(binary_mask) == 0:
        #     continue

        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # print(f'Bit {bit}: Found {len(contours)} contours')

        # Find the largest contour for this bit
        largest_contour = max(contours, key=cv2.contourArea) if contours else None
        if largest_contour is None:
            continue

        label_text = f"{bit}"
        (tw, th), b_ = cv2.getTextSize(
            label_text,
            font,
            font_scale,
            text_thickness,
        )
        tw, th = tw + 4, th + 6
        x, y, cw, ch = cv2.boundingRect(largest_contour)
        # print(f'Bit: {bit}, Contour Area: {cv2.contourArea(largest_contour)}')
        # print(f'Bounding Box: {(x, y, cw, ch)}, Text Size: {(tw, th)}')
        if cw < tw and ch < th:  # Skip if insufficient space for label
            continue
        if cv2.contourArea(largest_contour) < 300:
            continue

        # Store the largest contour for later text label placement
        largest_contours.append((bit, largest_contour))

        # First, draw the filled contour (highlighting the segment)
        color = colors[bit]
        overlay = output.copy()
        cv2.drawContours(overlay, [largest_contour], -1, color, thickness=cv2.FILLED)
        cv2.addWeighted(overlay, 0.3, output, 0.7, 0, dst=output)

        # Draw the contour outline
        cv2.drawContours(
            output, [largest_contour], -1, color, thickness=contour_thickness
        )

    # Now, after all contours are drawn, add the labels for the largest contours
    visible = {}
    for bit, largest_contour in largest_contours:
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        label_text = f"{bit}"
        result = cv2.getTextSize(label_text, font, font_scale, text_thickness)
        (text_w, text_h), baseline = result
        box_w, box_h = text_w + 4, text_h + 6

        # Find non-overlapping position for the label
        top_left, bottom_right = find_non_overlapping_position(
            cx, cy, box_w, box_h, occupied_boxes
        )
        if top_left is None:
            continue  # Skip if no valid spot found

        occupied_boxes.append((*top_left, *bottom_right))

        # Draw label box with the same color as the segment
        cv2.rectangle(
            output,
            top_left,
            bottom_right,
            (0, 0, 0),
            thickness=cv2.FILLED,
        )

        # Draw the text on top of the box (always visible)
        text_org = (top_left[0] + 3, top_left[1] + text_h + 2)
        cv2.putText(
            output,
            label_text,
            text_org,
            font,
            font_scale,
            colors[bit],
            text_thickness,
            cv2.LINE_AA,
        )

        visible[bit] = (d_min_depths[bit], d_max_depths[bit], d_avg_depths[bit])

    return output, visible
