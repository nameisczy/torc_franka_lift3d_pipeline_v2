import argparse
import mujoco
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/..")


def merge_summaries(dir_path: str, summary_name: str = "scene_summary.csv"):
    print("Merging: ".ljust(12), dir_path)
    dfs = []

    def add_summary(summary_path: str):
        df = pd.read_csv(summary_path)
        dfs.append(df)

    for file in os.listdir(dir_path):
        full_path = os.path.join(dir_path, file)
        if os.path.isfile(full_path):
            if file != summary_name and file.endswith(summary_name):
                add_summary(full_path)
        else:
            subdir_summary_path = os.path.join(full_path, summary_name)
            if os.path.exists(subdir_summary_path):
                add_summary(subdir_summary_path)

    if len(dfs) == 0:
        return
    merged = pd.concat(dfs, ignore_index=True)
    summary_file = f"{dir_path}/{summary_name}"
    merged.to_csv(summary_file, index=False)
    print("   Summary: ".ljust(18), summary_file)


def summarize_folder(folder: str, max_workers: int = 4):
    print("Extract folder: ".ljust(18), folder)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for base_dir_path, sub_dirs, files in os.walk(folder, topdown=False):
            futures = []
            for file in files:
                if file.endswith(".xml"):
                    file_path = os.path.join(base_dir_path, file)
                    future = executor.submit(summarize_scene, file_path)
                    futures.append(future)

            if futures:
                print(
                    f"  Submitted {len(futures)} scenes for parallel summarization in {base_dir_path}..."
                )
                [f.result() for f in futures]
                print(f"  All scenes in {base_dir_path} summarized.")

            merge_summaries(base_dir_path, "scene_summary.csv")


def summarize_scene(scene_path: str):
    print("  Extract scene: ".ljust(18), scene_path)
    try:
        model = mujoco.MjModel.from_xml_path(scene_path)
    except Exception:
        return

    data = mujoco.MjData(model)

    tick_count = 0
    while tick_count <= 1000:
        mujoco.mj_step(model, data)
        tick_count += 1

    all_objects = []
    floor_objects = []
    table_objects = []
    for i in range(model.nbody):
        body = model.body(i)
        dbody = data.body(i)
        name = body.name
        if (
            name.startswith("object_")
            or name.startswith("obj_")
            or name.startswith("0")
        ):
            all_objects.append(name)
            if dbody.xpos[2] < 0.5:
                floor_objects.append(name)
            else:
                table_objects.append(name)

    summary_prefix, _ = os.path.splitext(scene_path)

    data = {
        "Scene": Path(scene_path).stem,
        "Path": os.path.relpath(scene_path, ROOT_DIR),
        "All Objects": all_objects,
        "Floor Objects": floor_objects,
        "Table Objects": table_objects,
    }

    for key in data:
        if isinstance(data[key], list):
            data[key] = ",".join(data[key])

    df = pd.DataFrame(data, index=[0])
    df.to_csv(summary_prefix + "_scene_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="summarize_scene.sh",
        epilog="Summarizes metadata information for scenes.",
    )
    parser.add_argument("--scene", nargs="+", help="Scene(s) to look through.")
    parser.add_argument(
        "--folder",
        nargs="+",
        default=[ROOT_DIR + "/tests/scenes/final"],
        help="Folder(s) to look through.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Maximum number of worker threads to use.",
    )

    args = parser.parse_args()
    max_workers = args.workers

    if args.scene:
        print(f"Submitting {len(args.scene)} scenes for parallel summarization...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(summarize_scene, scene) for scene in args.scene]
            [f.result() for f in futures]

    if args.folder:
        for folder in args.folder:
            summarize_folder(folder, max_workers=max_workers)
