import json
import os
import pickle
import threading
import time
import re
import subprocess
import mujoco
import mujoco.viewer
import argparse
import signal
from rich.console import Console

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
)

con = Console(highlight=False)


def exiterr(*msg):
    con.print(f"[red][bold][ERROR]:[/bold] {''.join(msg)}")
    exit()


def run(experiment_dir: str, freeze: bool):
    con.print("[bold cyan]MuJoCo State Inspector 🔍")
    con.print("[cyan]  experiment_dir: ", experiment_dir)
    con.print("[cyan]  freeze:         ", freeze)

    con.print("\n[bold magenta]Controls:")
    con.print("  [bold][magenta]Space[/bold]: Toggle freeze")
    con.print("  [bold][magenta]R[/bold]: Reload state")
    con.print("  [bold][magenta]Left Arrow[/bold]: Previous state")
    con.print("  [bold][magenta]Right Arrow[/bold]: Next state\n")

    if experiment_dir is None:
        exiterr("Must provide --experiment_dir")

    shutdown_flag = False

    def signal_handler(sig, frame):
        nonlocal shutdown_flag
        shutdown_flag = True

    signal.signal(signal.SIGINT, signal_handler)

    # read all state_XX.pkl files from experiment_dir, sort by index
    experiment_info = None
    state_file = None
    state_dict = None
    state_files = []
    state_file_index = 0
    mj_lock = threading.Lock()
    if experiment_dir:
        state_file_pattern = re.compile(r"^state_(\d+)\.pkl$")
        for f in os.listdir(experiment_dir):
            if state_file_pattern.match(f):
                state_files.append(os.path.join(experiment_dir, f))
        state_files.sort(
            key=lambda x: int(state_file_pattern.match(os.path.basename(x)).group(1))
        )
        experiment_info_path = os.path.join(experiment_dir, "info_experiment.json")
        if os.path.exists(experiment_info_path):
            with open(experiment_info_path, "r") as f:
                experiment_info = json.load(f)

    if experiment_dir and len(state_files) == 0:
        exiterr(f"No state_XX.pkl files found in {experiment_dir}")

    con.print(
        f"[bold cyan]Experiment info:[/bold cyan] {json.dumps(experiment_info, indent=2)}"
    )

    def key_callback(keycode: int):
        nonlocal freeze, state_file_index, state_files
        key = chr(keycode)
        if key == " ":
            freeze = not freeze
            if freeze:
                con.print("[bold magenta]FROZEN")
                print_curr_state_info()
            else:
                con.print("[magenta]UNFROZEN")
        elif key == "R":
            con.print("[bold magenta]RELOAD STATE")
            load_curr_state()
        elif keycode == 263:  # left arrow
            state_file_index = (state_file_index - 1) % len(state_files)
            con.print(
                f"[bold magenta]PREV STATE: {state_file_index + 1}/{len(state_files)}"
            )
            load_curr_state()
        elif keycode == 262:  # right arrow
            state_file_index = (state_file_index + 1) % len(state_files)
            con.print(
                f"[bold magenta]NEXT STATE: {state_file_index + 1}/{len(state_files)}"
            )
            load_curr_state()

    model: mujoco.MjModel = mujoco.MjSpec().compile()
    data: mujoco.MjData = mujoco.MjData(model)

    perturb_obj_id = -1

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False

    scene_xml = None

    def print_curr_state_info():
        nonlocal model, data, experiment_info, state_dict, state_file
        con.print(f"[bold cyan]State info: {os.path.basename(state_file)}")

        con.print("[cyan]  Gripper collisions:")
        collisions_set = set()
        for g1, g2 in data.contact.geom:
            body1 = model.geom(g1).bodyid[0]
            body2 = model.geom(g2).bodyid[0]
            body1_name = model.body(body1).name
            body2_name = model.body(body2).name

            collision_pair = tuple(sorted([body1_name, body2_name]))
            if collision_pair in collisions_set:
                continue
            collisions_set.add(collision_pair)

            if "pad" in body1_name or "pad" in body2_name:
                con.print(f"    {body1_name} <-> {body2_name}")

        def og_gripper_collision_check():
            def has_ancestor(geom, name):
                parent = model.body(geom.bodyid[0])
                while parent.parentid:
                    if parent.name == name:
                        return True
                    parent = model.body(parent.parentid[0])
                return False

            gripper_geom_ids = set(
                filter(
                    lambda i: has_ancestor(model.geom(i), "robotiq_2f85"),
                    range(model.ngeom),
                )
            )

            grasping = set()
            for g1, g2 in data.contact.geom:
                gripisg1 = g1 in gripper_geom_ids
                gripisg2 = g2 in gripper_geom_ids
                if gripisg1 and not gripisg2:
                    objid = model.geom(g2).bodyid[0]
                    grasping.add(model.body(objid).name)
                if gripisg2 and not gripisg1:
                    objid = model.geom(g1).bodyid[0]
                    grasping.add(model.body(objid).name)
            grasping = list(grasping)

            con.print(f"[cyan]  Gripped objects (original check):[/cyan] {grasping}")
            if "target" in state_dict:
                target = model.body(int(state_dict["target"])).name
                success = len(grasping) == 1 and grasping[0] == target
                con.print(f"[cyan]    Target object:[/cyan] {target}")
                con.print(f"[cyan]    Success:[/cyan] {success}")

        def recursive_gripper_collision_check():
            # Check if any objects are colliding with the gripper, as well as recursively fetch objects that are colliding with these objects
            def has_ancestor(geom, name):
                parent = model.body(geom.bodyid[0])
                while parent.parentid:
                    if parent.name == name:
                        return True
                    parent = model.body(parent.parentid[0])
                return False

            gripper_geom_ids = set(
                filter(
                    lambda i: has_ancestor(model.geom(i), "robotiq_2f85"),
                    range(model.ngeom),
                )
            )

            grasping = set()
            visited_body_ids = set()
            current_geom_ids = gripper_geom_ids.copy()

            while current_geom_ids:
                next_geom_ids = set()
                for geom_1_id, geom_2_id in data.contact.geom:
                    g1_in = geom_1_id in current_geom_ids
                    g2_in = geom_2_id in current_geom_ids

                    if g2_in and not g1_in:
                        g1_in, g2_in = g2_in, g1_in
                        geom_1_id, geom_2_id = geom_2_id, geom_1_id
                    if g1_in and not g2_in:
                        body_id = model.geom(geom_2_id).bodyid[0]
                        body_name = model.body(body_id).name
                        if (
                            body_id not in visited_body_ids
                            and geom_2_id not in gripper_geom_ids
                            and body_name.startswith(
                                ("object_", "obj_", "0")
                            )  # only consider objects as grasped, ignore other potential collisions
                        ):
                            visited_body_ids.add(body_id)
                            grasping.add(body_name)

                            # Add all geoms of this body for next iteration
                            for geom_idx in range(model.ngeom):
                                if model.geom(geom_idx).bodyid[0] == body_id:
                                    next_geom_ids.add(geom_idx)

                current_geom_ids = next_geom_ids

            grasping = list(grasping)

            con.print(f"[cyan]  Gripped objects (recursive check):[/cyan] {grasping}")
            if "target" in state_dict:
                target = model.body(int(state_dict["target"])).name
                success = len(grasping) == 1 and grasping[0] == target
                con.print(f"[cyan]    Target object:[/cyan] {target}")
                con.print(f"[cyan]    Success:[/cyan] {success}")

        og_gripper_collision_check()
        recursive_gripper_collision_check()

        dropped = []
        for i in range(model.nbody):
            body = model.body(i)
            dbody = data.body(i)
            name = body.name
            if name[:7] == "object_" or name[:4] == "obj_" or name[0] == "0":
                if dbody.xpos[2] < 0.5:
                    dropped.append(name)
        con.print(f"[cyan]  Dropped objects:[/cyan] {dropped}")

    def load_curr_state():
        nonlocal model, data, viewer, scene_xml, state_dict, state_file
        with mj_lock:
            state_file = state_files[state_file_index]
            state_dict = None
            with open(state_file, "rb") as file:
                state_dict = pickle.load(file)
            if state_dict is None:
                return

            # convert path to use current repo's lab_vbnpm's directory
            new_scene_xml = os.path.join(
                ROOT_DIR, state_dict["scene_xml"].split("lab_vbnpm/")[-1]
            )

            if new_scene_xml != scene_xml:
                # trim all paths until lab_vbnpm
                model = mujoco.MjModel.from_xml_path(new_scene_xml)
                data = mujoco.MjData(model)

                mujoco.mj_forward(model, data)

            mujoco.mj_setState(
                model,
                data,
                state_dict["mujoco_state"],
                mujoco.mjtState.mjSTATE_INTEGRATION,
            )

            mujoco.mj_forward(model, data)

            print_curr_state_info()

        # Sync viewer if scene changed
        # Synce outside of lock, since viewer.sync() may potentially run key_callback which then attempts to acquire the mj_lock
        if new_scene_xml != scene_xml and viewer:
            scene_xml = new_scene_xml
            viewer._get_sim().load(model, data, scene_xml)
            viewer.sync()

    con.print(f"[bold magenta]CURR STATE: {state_file_index + 1}/{len(state_files)}")
    load_curr_state()

    while viewer.is_running() and not shutdown_flag:
        step_start = time.time()

        with mj_lock:
            if perturb_obj_id != viewer.perturb.select:
                perturb_obj_id = viewer.perturb.select
                perturb_obj_name = model.body(perturb_obj_id).name
                con.print(
                    f"[bold]Selected[/bold] {str(perturb_obj_id):<4} {perturb_obj_name:<30}"
                )

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        if not freeze:
            with mj_lock:
                mujoco.mj_step(model, data)

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    if viewer:
        viewer.close()
        viewer = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="mujoco_state_inspector.sh",
        description="Load and visualizes a mujoco state pickle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "experiment_dir",
        help="Experiment directory to read states from",
        type=str,
    )
    parser.add_argument(
        "-z",
        "--freeze",
        help="Whether to freeze the simulation (don't step forward in time).",
        action="store_true",
    )
    args = parser.parse_args()

    run(experiment_dir=args.experiment_dir, freeze=args.freeze)
