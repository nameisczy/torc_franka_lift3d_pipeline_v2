import os
import re
import sys
import json
import base64

import cv2
import numpy as np
from openai import OpenAI
from skimage.exposure import rescale_intensity

from rospkg import RosPack
from task_planner.prompt import create_labeled_image, Prompter
from task_planner.dep_graph import (
    DepGraph,
    get_behind_candidates,
    get_behind_below_dependencies,
)

identify = """
You will receive a request that contains:
 * An image of the current environment
 * A description of the target object desired to be picked and placed.

Your task is to idenify which object is the target object and return the pixel coordinates of its center in the image. You should consider the following:

You have to respond with a JSON object, containing two keys, one for the x coordinate in image pixel space and one for the y coordinate in image pixel space, corresponding to a valid location near the center of mass of the target object in the input image. The coordinates should be relative to the bottom-left corner of the image, with (0, 0) corresponding to the top-left corner and (640, 480) to the top right corner.:

Example response:
Thoughts: <some text goes here>

output:
```json
{"x": 125, "y": 262}
```
It's essential to stick to the above format. Before you produce the output, please write a small paragraph in the `Thoughts` section of the response, where you explaiyour choice of the object placement.
"""

describe = """
What objects do you see in the scene?
"""

simple_point = """
Point to the target object in the image. The answershould follow the jsonformat: {"name": "target object description", "point": [...]} The points are in [y, x] format normalized to 0-680 for x and 0-480 for y. The point 0,0 is the top-left corner of the image, and 640,480 is the bottom-right corner of the image.
"""

rp = RosPack()
tp_dir = os.path.join(rp.get_path("lab_vbnpm"), "scripts/task_planner/prompt_templates")
with open(os.path.join(tp_dir, "pick_object.txt"), "r") as f:
    pick_object = f.read()

with open(os.path.join(tp_dir, "below.txt"), "r") as f:
    below = f.read()

with open(os.path.join(tp_dir, "behind.txt"), "r") as f:
    behind = f.read()

with open(os.path.join(tp_dir, "make_dg.txt"), "r") as f:
    make_dg = f.read()

# task = identify
# task = describe
# task = simple_point
task = pick_object
# task = make_dg
# task = below
# task = behind

# img1 = cv2.imread('./TEST.jpg')

DIR = sys.argv[1]
TRIAL = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""
img0 = cv2.imread(f"{DIR}/color{TRIAL}.png")
img2 = cv2.imread(f"{DIR}/depth{TRIAL}.png", cv2.IMREAD_ANYDEPTH)
seg = np.load(f"{DIR}/mask{TRIAL}.npy")
with open(f"{DIR}/config.json", "r") as f:
    cam_intr = json.load(f)["intrinsic_matrix"]

img1, vis = create_labeled_image(img0, img2, seg)
# dg_edges = get_behind_below_dependencies(seg, img2, cam_intr, vis.keys())
# print(json.dumps(dg_edges, indent=2))
print(vis)
candidate_edges = get_behind_candidates(seg, img2, vis.keys())
candidates = ""
for s, t in candidate_edges:
    candidates += f"  * Object {s} is behind Object {t}\n"
# print(candidates)
task = task.replace("<candidates>", candidates)

depth_str = ""
mx0 = max(vis.values(), key=lambda x: x[1])
mx1 = max(vis.values(), key=lambda x: x[0])
mn0 = min(vis.values(), key=lambda x: x[1])
mn1 = min(vis.values(), key=lambda x: x[0])
mx = max(mx0[1], mx1[1])
mn = min(mn0[0], mn1[0])
for k, v in vis.items():
    min_d = 100 * (v[0] - mn) / (mx - mn)
    max_d = 100 * (v[1] - mn) / (mx - mn)
    avg_d = 100 * (v[2] - mn) / (mx - mn)
    # depth_str += f'  * Object {k}: min depth {min_d:.2f}, max depth {max_d:.2f}\n'
    depth_str += f"  * Object {k} has average depth {avg_d:.2f}\n"
    print(f"  * Object {k}: min depth {min_d:.2f}, max depth {max_d:.2f}")
# print(depth_str)
task = task.replace("<object_depths>", depth_str)

# img2 = np.log10(img2)
# img2 = rescale_intensity(
#     img2, in_range='image', out_range=(0, 255)
# ).astype(np.uint8)
# img2 = cv2.applyColorMap(img2, cv2.COLORMAP_JET)

S = 1.5
# cv2.imshow("Image", cv2.resize(img0, (int(S * 640), int(S * 480))))
cv2.imshow("Labeled Image", cv2.resize(img1, (int(S * 640), int(S * 480))))
# cv2.imshow("Depth Image", cv2.resize(img2, (int(S * 640), int(S * 480))))
cv2.waitKey(0)
# input(task)

target = "0"  # input("Which object do you want to pick?\n> ")
print("Thinking...")
# prompter = Prompter(url='http://lab.cs.lab.edu:4000/v1')

prompter = Prompter(
    url="https://generativelanguage.googleapis.com/v1beta/openai/",
    key=os.getenv("GEMINI_API_KEY"),
)
response = prompter.prompt_model(task.replace("<target>", target), [img1])
print(response)
str_json = re.findall("(?<=```json)[\s\S]*(?=```)", response)[0]
print(json.loads(str_json))

# pickable = {}
# obj_deps = {}
# for relation in json.loads(str_json):
# # for relation in dg_edges:
#     src = relation['source']
#     tgt = relation['target']
#     rel = relation['relation']
#     if src not in obj_deps:
#         obj_deps[src] = {}
#     if tgt not in obj_deps[src]:
#         obj_deps[src][tgt] = {'d': rel}
#     else:
#         obj_deps[src][tgt]['d'] = 'both'

# G = DepGraph(obj_deps, pickable)
# G.draw()
