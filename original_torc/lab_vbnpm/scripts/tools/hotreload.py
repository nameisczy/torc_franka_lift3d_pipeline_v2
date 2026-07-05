from itertools import chain
import os
from pathlib import Path
import time
import datetime
import mujoco  # type: ignore[import-untyped]
import mujoco.viewer  # type: ignore[import-untyped]
import argparse
from rich.console import Console
from rich.theme import Theme
import signal
from typing import List, Tuple, Set, Union
import xml.etree.ElementTree as ET
import threading
from watchfiles import watch
import re

con = Console(highlight=False, theme=Theme({"orange": "#ffaa00"}))


def printerr(*msg):
    con.print(f"[red][bold][ERROR]:[/] {''.join(msg)}[/]")


parser = argparse.ArgumentParser(
    prog="hotreload.sh",
    description="Load and visualizes a mujoco scene file, and reloads the mujoco viewer whenever the scene file is modified.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("file", help="File to load.", type=str)
parser.add_argument(
    "-f", "--freeze", help="Loads simulation as frozen.", action="store_true"
)
parser.add_argument(
    "-d",
    "--debounce",
    help="Minimum seconds between hot reloads.",
    type=float,
    default=0.1,
)
args = parser.parse_args()

model = None
data = None
viewer = None
reload_flag = False
scene_file = os.path.abspath(args.file)
debounce_secs = args.debounce
freeze = args.freeze
shutdown_flag = False


def signal_handler(sig, frame):
    global shutdown_flag
    shutdown_flag = True


signal.signal(signal.SIGINT, signal_handler)

con.print("[orange][bold]Hot Reload 🔥")
con.print(f"[orange]  scene_file: {scene_file}")
con.print(f"[orange]  debounce:   {debounce_secs}s")
con.print(f"[orange]  freeze:     {freeze}")
watched_files: List[str] = []
watcher_thread = None
stop_watcher = threading.Event()


def get_xml_dependencies(xml_path: str) -> Set[str]:
    """Extract immediate XML file dependencies from include tags."""
    dependencies: Set[str] = set()
    if not os.path.exists(xml_path):
        return dependencies

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Iter over all
        #   <include file="..."/> elements
        #   <model file="..."/> elements
        for include in chain(root.iter("include"), root.iter("model")):
            file_attr = include.get("file")
            if file_attr:
                # Resolve relative paths from the XML file's directory
                xml_dir = os.path.dirname(xml_path)
                dep_path = os.path.abspath(os.path.join(xml_dir, file_attr))
                if os.path.exists(dep_path):
                    dependencies.add(dep_path)

    except Exception as e:  # noqa: E722s
        lineno, colno = 0, 0
        if isinstance(e, ET.ParseError):
            lineno, colno = e.position
        err_location = f"{xml_path}:{lineno}:{colno}"
        printerr(f"Failed to parse XML @ {err_location}: {e}")

    return dependencies


def get_all_xml_dependencies_depth(
    xml_path: str,
    include_root: bool = True,
) -> List[Tuple[str, int]]:
    """Recursively collect all XML dependencies and their depth levels."""
    all_deps = []
    to_process = [(xml_path, 0)]
    processed = set()

    while to_process:
        current, depth = to_process.pop()
        if current in processed:
            continue
        if depth > 0 or include_root:
            all_deps.append((current, depth))
        processed.add(current)

        deps = get_xml_dependencies(current)
        to_process.extend([(dep, depth + 1) for dep in deps])

    return all_deps


def get_all_xml_dependencies(xml_path: str, include_root: bool = True) -> List[str]:
    """Recursively collect all XML dependencies."""
    return [dep for dep, _ in get_all_xml_dependencies_depth(xml_path, include_root)]


def load_model() -> Union[Tuple[mujoco.MjModel, mujoco.MjData], None]:
    model = None
    try:
        model = mujoco.MjModel.from_xml_path(scene_file)
    except ValueError as e:
        con.print()
        printerr(f"Error loading model XML: {e}")
        # Extract the missing body name
        match = re.search(r"could not find body '([^']+)'", str(e))
        missing_body = None
        if match:
            missing_body = match.group(1)
            con.print(f"[DEBUG] The model is missing a body named: '{missing_body}'")
        else:
            con.print("[DEBUG] Could not extract missing body name from error message.")

        if missing_body:
            # Print all body references in the XML with line numbers and context
            con.print("All body references in the XML (with line numbers and context):")
            with open(scene_file, "r") as f:
                count = 0
                for i, line in enumerate(f, 1):
                    for m in re.finditer(r"body=\"([^\"]+)\"", line):
                        count += 1
                        con.print(
                            f'  #{count} Line {i}: body="{m.group(1)}" | {line.strip()}'
                        )
            printerr(f"XML Error: could not find body {missing_body}")
        printerr(e.args[0].replace("\n", " "))
        return None
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    return model, data


def file_watcher():
    """Watch files and set reload flag on changes."""
    global reload_flag, watched_files, debounce_secs

    while not stop_watcher.is_set():
        if not watched_files:
            time.sleep(0.1)
            continue

        try:
            for _ in watch(
                *watched_files,
                stop_event=stop_watcher,
                debounce=int(debounce_secs * 1000),
            ):
                con.print("[orange]Scene modified, reloading...")
                reload_flag = True
                break
        except Exception:
            pass


def update_watched_files():
    """Update the set of watched files."""
    global scene_file, watched_files

    # Watch all XML dependencies of a scene file
    watched_files = get_all_xml_dependencies(scene_file, include_root=True)


def run():
    global viewer, reload_flag, scene_file
    model: mujoco.MjModel = mujoco.MjSpec().compile()
    data: mujoco.MjData = mujoco.MjData(model)

    perturb_obj_id = -1

    def key_callback(keycode: int):
        global reload_flag
        key = chr(keycode)
        if key == " ":
            reload_flag = True

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False

    def reload_viewer():
        nonlocal model, data
        global viewer
        res = load_model()
        if not res:
            return
        model, data = res
        mujoco.mj_forward(model, data)
        if viewer:
            viewer._get_sim().load(model, data, scene_file)
        update_watched_files()

    def print_xml_dependencies():
        con.print("\n[orange][bold]XML Dependencies:")
        dependencies = get_all_xml_dependencies_depth(scene_file, include_root=True)
        for dep_path, depth in dependencies:
            con.print(f"[orange]  {'  ' * depth}{os.path.relpath(dep_path, '.')}")
        con.print()

    print_xml_dependencies()
    sync_interval = int((1 / model.opt.timestep) / 60)
    con.print("[orange][bold]Simulation Parameters:")
    con.print(f"[orange]  timestep:      {model.opt.timestep}s")
    con.print(f"[orange]  sync_interval: {sync_interval} ticks/s\n")

    con.print("[orange][bold]Controls:")
    con.print("[orange]  <space>: Reload scene\n")

    reload_viewer()

    ticks = 0
    while viewer.is_running() and not shutdown_flag:
        ticks += 1
        step_start = time.time()

        if reload_flag:
            reload_flag = False
            reload_viewer()

        if perturb_obj_id != viewer.perturb.select:
            perturb_obj_id = viewer.perturb.select
            perturb_obj_name = model.body(perturb_obj_id).name
            con.print(
                f"[bold]Selected[/]  {str(perturb_obj_id):<4}  {perturb_obj_name:<30} - xpos: {data.body(perturb_obj_id).xpos.tolist()}"
            )

        if not freeze:
            mujoco.mj_step(model, data)

        if ticks % sync_interval == 0:
            viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    if viewer:
        viewer.close()
        viewer = None


if not os.path.exists(scene_file):
    printerr("File does not exist.")
    exit()

try:
    update_watched_files()
    watcher_thread = threading.Thread(target=file_watcher, daemon=True)
    watcher_thread.start()
    run()
finally:
    con.print("\n[red bold]🛑 Stopping...")
    stop_watcher.set()
    if watcher_thread and watcher_thread.is_alive():
        watcher_thread.join(timeout=1.0)
    if viewer:
        viewer.close()
        viewer = None
