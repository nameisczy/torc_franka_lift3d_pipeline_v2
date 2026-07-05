import os
import rospkg
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from load_gc6d_grasps import load_grasps

if 'GC6D_ROOT' not in os.environ:
    print('Please set the environment variable GC6D_ROOT (e.g. export GC6D_ROOT=/path/to/GraspClutter6D)')
    exit(0)

gc6d_path = os.environ['GC6D_ROOT']

rp = rospkg.RosPack()
lab_path = rp.get_path('lab_vbnpm')

OUT_DIR = f"{lab_path}/tests/scenes/grasps/"
os.makedirs(OUT_DIR, exist_ok=True)

def process_object(object_index):
    poses = load_grasps(gc6d_path, object_index)

    pose_mats = []
    for R, t in poses:
        transform = np.zeros((4, 4))
        transform[:3, :3] = R
        transform[:3, 3] = t
        transform[3, 3] = 1.0
        pose_mats.append(transform)

    pose_mats = np.array(pose_mats)

    if pose_mats.shape[0] > 0:
        out_file = f"{OUT_DIR}/grasps_{object_index:06d}.npy"
        np.save(out_file, pose_mats)
        return f"Saved {pose_mats.shape[0]} grasps for object {object_index} to {out_file}"
    else:
        return f"No grasps found for object {object_index}, skipping."

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_object, i) for i in range(1, 201)]
        for future in as_completed(futures):
            print(future.result())

# Objects with no grasp

# 118, 76, 37