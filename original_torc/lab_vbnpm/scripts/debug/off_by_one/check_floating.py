from dataclasses import dataclass
import os
import pickle
import argparse
import subprocess
import re
import warnings
from multiprocessing import Pool, cpu_count
from rich.console import Console
from tqdm.rich import TqdmExperimentalWarning, tqdm

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

FLOATING_FILE_PATTERN = re.compile(r"^state_(\d+)_floating\.pkl$")

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
)

con = Console(highlight=False)


@dataclass
class StateFloatingInfo:
    """
    Data class to hold sets of floating objects for a state.
    """

    floating_objects: set
    floating_positions: dict
    state_file: str


def _format_floating_objects(state_floating_info: StateFloatingInfo) -> str:
    positions = state_floating_info.floating_positions or {}
    parts = []
    for name in sorted(state_floating_info.floating_objects):
        pos = positions.get(name)
        if pos is None or len(pos) < 3:
            parts.append(name)
        else:
            parts.append(f"{name}@({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    return ", ".join(parts)


def process_directory_state_files(directory: str) -> str:
    """
    Process all state files in a directory, loading the mujoco model once.
    Saves results to state_XXX_floating.pkl files.

    Args:
        directory: Directory path containing state_XXX.pkl files

    Returns:
        error_msg: str or None (None if successful)
    """
    try:
        import mujoco

        def check_floating_objects(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
            """
            Check for floating objects (objects not in collision with anything else).

            Returns:
                Mapping of floating object name to position
            """
            # Collect all objects involved in collisions
            colliding_bodies = set()

            for g1, g2 in data.contact.geom:
                body1_id = model.geom(g1).bodyid[0]
                body2_id = model.geom(g2).bodyid[0]
                body1_name = model.body(body1_id).name
                body2_name = model.body(body2_id).name

                colliding_bodies.add(body1_name)
                colliding_bodies.add(body2_name)

            # Find all object bodies (those starting with obj_ or object_)
            floating = {}
            for i in range(model.nbody):
                body = model.body(i)
                name = body.name
                if name.startswith("obj_") or name.startswith("object_"):
                    # Treat as floating only if not colliding, above height threshold, and is not infty or Nan.
                    if (
                        name not in colliding_bodies
                        and data.xpos[i][2] >= 0.5
                        and not any(
                            [float("inf") in data.xpos[i], float("nan") in data.xpos[i]]
                        )
                    ):
                        floating[name] = tuple(data.xpos[i])

            return floating

        # Find all state files in this directory
        state_files = []
        state_file_pattern = re.compile(r"^state_(\d+)\.pkl$")
        for f in os.listdir(directory):
            if state_file_pattern.match(f):
                state_files.append(os.path.join(directory, f))

        # Remove any cached floating results from previous runs
        for f in os.listdir(directory):
            if FLOATING_FILE_PATTERN.match(f):
                try:
                    os.remove(os.path.join(directory, f))
                except OSError:
                    pass

        if not state_files:
            return "No state files found"

        state_files.sort()

        # Load the first state file to get scene_xml
        with open(state_files[0], "rb") as f:
            first_state = pickle.load(f)

        if first_state is None:
            return "First state dict is None"

        scene_xml_path = first_state.get("scene_xml")
        if not scene_xml_path:
            return "No scene_xml in state"

        # Convert path to use current repo's directory
        new_scene_xml = os.path.join(
            ROOT_DIR, scene_xml_path.split("lab_vbnpm/")[-1]
        )

        if not os.path.exists(new_scene_xml):
            return f"Scene XML not found: {new_scene_xml}"

        # Load the model once for this directory
        model = mujoco.MjModel.from_xml_path(new_scene_xml)
        data = mujoco.MjData(model)

        # Process all state files in this directory
        for state_file in state_files:
            with open(state_file, "rb") as f:
                state_dict = pickle.load(f)

            if state_dict is None:
                continue

            # Restore the state
            mujoco.mj_setState(
                model,
                data,
                state_dict["mujoco_state"],
                mujoco.mjtState.mjSTATE_INTEGRATION,
            )

            # Step the simulation forward and require objects to float every step
            steps = 10
            floating = None
            for _ in range(steps):
                mujoco.mj_step(model, data)
                floating_step = check_floating_objects(model, data)
                if floating is None:
                    floating = floating_step
                else:
                    floating_keys = set(floating.keys()) & set(floating_step.keys())
                    floating = {name: floating_step[name] for name in floating_keys}

            if floating is None:
                floating = {}

            # Save results if this state file has floating objects
            if floating:
                floating_info = StateFloatingInfo(
                    floating_objects=set(floating.keys()),
                    floating_positions=floating,
                    state_file=state_file,
                )
                pickle.dump(
                    floating_info,
                    open(state_file.replace(".pkl", "_floating.pkl"), "wb"),
                )

        return None

    except Exception as e:
        return str(e)


def find_state_directories(start_dir: str) -> list:
    """
    Recursively find all directories containing state_XXX.pkl files.

    Args:
        start_dir: Root directory to search

    Returns:
        List of directory paths that contain state files
    """
    directories = set()
    state_file_pattern = re.compile(r"^state_(\d+)\.pkl$")
    for root, dirs, files in os.walk(start_dir):
        for f in files:
            if state_file_pattern.match(f):
                directories.add(root)

    return sorted(list(directories))


def _worker_process_directory(directory: str) -> tuple:
    """
    Worker function for multiprocessing pool.
    Processes all state files in a directory and returns results.

    Args:
        directory: Directory path containing state files

    Returns:
        (directory_path, error_msg: str or None)
    """
    error = process_directory_state_files(directory)
    return directory, error


def print_floating_info(runs_dir: str):
    """
    Read and display results from state_XXX_floating.pkl files.

    Args:
        runs_dir: Path to runs directory to search for floating pkl files
    """
    con.print("[bold cyan]Float object results from state_XXX_floating.pkl files 🔍")
    con.print(f"[cyan]Searching in: {runs_dir}\n")

    if not os.path.isdir(runs_dir):
        con.print(f"[red][bold]ERROR:[/bold] {runs_dir} is not a valid directory")
        return

    # Find all state_XXX_floating.pkl files
    floating_files = []
    for root, dirs, files in os.walk(runs_dir):
        for f in files:
            if FLOATING_FILE_PATTERN.match(f):
                floating_files.append(os.path.join(root, f))

    if not floating_files:
        con.print("[yellow]No floating object files found")
        return

    floating_files.sort()

    # Group by directory
    dirs_with_floating = {}
    for floating_file in floating_files:
        directory = os.path.dirname(floating_file)
        if directory not in dirs_with_floating:
            dirs_with_floating[directory] = []
        dirs_with_floating[directory].append(floating_file)

    # Display results
    problem_dirs = 0
    for directory in sorted(dirs_with_floating.keys()):
        has_content = False
        for floating_file in dirs_with_floating[directory]:
            try:
                with open(floating_file, "rb") as f:
                    state_floating_info: StateFloatingInfo = pickle.load(f)

                # Check if there are floating objects
                if state_floating_info and state_floating_info.floating_objects:
                    if not has_content:
                        con.print(f"  [blue]{directory}")
                        has_content = True
                        problem_dirs += 1

                    fname = os.path.basename(state_floating_info.state_file)
                    con.print(
                        f"    [cyan]{fname:<16} {_format_floating_objects(state_floating_info)}"
                    )

            except Exception as e:
                con.print(f"  [red]Error reading {floating_file}: {e}")
        if has_content:
            con.print()  # Add spacing between directories

    con.print(
        f"[yellow]Found [bold]{len(floating_files)}[/bold] floating object file(s)[/yellow]"
    )
    if problem_dirs == 0:
        con.print("[green]✓ No floating objects recorded in cached files")
    else:
        con.print(
            f"[yellow]Found [bold]{problem_dirs}[/bold] experiment directories with floating objects.[/yellow]"
        )


def run(runs_dir: str, verbose: bool = False, num_workers: int = None):
    """
    Recursively check all state files in runs directory for floating objects.

    Args:
        runs_dir: Path to runs directory
        verbose: Print detailed output
        num_workers: Number of worker processes (default: CPU count)
    """
    con.print("[bold cyan]Checking for floating objects in state files 🔍")
    con.print(f"[cyan]Searching in: {runs_dir}\n")

    if not os.path.isdir(runs_dir):
        con.print(f"[red][bold]ERROR:[/bold] {runs_dir} is not a valid directory")
        return

    # Find all directories containing state files
    directories = find_state_directories(runs_dir)

    if not directories:
        con.print("[yellow]No state files found")
        return

    con.print(
        f"[cyan]Found [bold]{len(directories)}[/bold] director(ies) with state files"
    )

    if num_workers is None:
        num_workers = max(cpu_count() - 1, 1)  # Leave one CPU free

    con.print(f"[cyan]Using {num_workers} worker process(es)\n")

    # Process directories in parallel
    with Pool(processes=num_workers) as pool:
        results = tqdm(
            pool.imap_unordered(_worker_process_directory, directories),
            total=len(directories),
            desc="Processing directories",
            unit="dir",
        )
        results = list(results)

    # Collect errors
    for directory, error in results:
        if error and verbose:
            con.print(f"[yellow]Warning ({directory}): {error}")

    con.print()  # Add spacing
    # Display results using print_floating_info to read the cached files
    print_floating_info(runs_dir)


def print_raw_floating_files(runs_dir: str):
    """
    Print only the file paths of state files that have floating objects.
    Reads from state_XXX_floating.pkl files.

    Args:
        runs_dir: Path to runs directory to search for floating pkl files
    """
    # Find all state_XXX_floating.pkl files
    floating_files = []
    for root, dirs, files in os.walk(runs_dir):
        for f in files:
            if FLOATING_FILE_PATTERN.match(f):
                floating_files.append(os.path.join(root, f))

    floating_files.sort()

    # Extract and print state file paths
    for floating_file in floating_files:
        try:
            with open(floating_file, "rb") as f:
                state_floating_info: StateFloatingInfo = pickle.load(f)

            if state_floating_info and state_floating_info.floating_objects:
                con.print(state_floating_info.state_file)
        except Exception as e:
            # Skip files that can't be read
            pass


def visualize_floating_experiments(runs_dir: str):
    """
    Loop through directories with floating objects and visualize them with mujoco_state_inspector.

    Args:
        runs_dir: Path to runs directory
    """
    # Find all state_XXX_floating.pkl files
    floating_files = []
    for root, dirs, files in os.walk(runs_dir):
        for f in files:
            if FLOATING_FILE_PATTERN.match(f):
                floating_files.append(os.path.join(root, f))

    if not floating_files:
        con.print("[yellow]No floating object files found")
        return

    # Group by directory
    dirs_with_floating = set()
    for floating_file in floating_files:
        directory = os.path.dirname(floating_file)
        dirs_with_floating.add(directory)

    try:
        # Loop through each directory and visualize
        for directory in sorted(dirs_with_floating):
            con.print(f"\n[bold blue]{directory}")
            state_floating_files = sorted(
                [
                    os.path.join(directory, f)
                    for f in os.listdir(directory)
                    if FLOATING_FILE_PATTERN.match(f)
                ]
            )

            for state_file in state_floating_files:
                try:
                    with open(state_file, "rb") as f:
                        state_floating_info: StateFloatingInfo = pickle.load(f)
                    if state_floating_info and state_floating_info.floating_objects:
                        fname = os.path.basename(state_file)
                        con.print(
                            f"  [cyan]{fname:<32} {_format_floating_objects(state_floating_info)}"
                        )
                except Exception as e:
                    con.print(f"  [red]Error reading {state_file}: {e}")

            con.print()

            # Get the path to mujoco_state_inspector.sh
            inspector_script = os.path.join(ROOT_DIR, "mujoco_state_inspector.sh")

            try:
                # Run mujoco_state_inspector with frozen flag
                subprocess.run([inspector_script, directory, "-z"], check=False)
            except Exception as e:
                con.print(f"[red]Error running inspector on {directory}: {e}")
    except KeyboardInterrupt:
        con.print("\n[red]🛑 Stopping")

    con.print(f"\n[green]Finished visualizing {len(dirs_with_floating)} director(ies)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="check_floating",
        description="Recursively check state files fo floating objects",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Command to execute", required=True
    )

    # Common arguments
    def add_common_args(subparser):
        subparser.add_argument(
            "-d",
            "--directory",
            type=str,
            default=os.path.join(ROOT_DIR, "experiments/runs"),
            help="Root directory to search for state files",
        )

    # Process command (scan state files and check for floating objects)
    process_parser = subparsers.add_parser(
        "process", help="Scan state files for floating objects (default)"
    )
    add_common_args(process_parser)
    process_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed output for each state file",
    )
    process_parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )

    # Print command (display cached results)
    print_parser = subparsers.add_parser(
        "print", help="Read and display cached state_XXX_floating.pkl files"
    )
    add_common_args(print_parser)

    # Print raw command (print only file paths)
    print_raw_parser = subparsers.add_parser(
        "print_raw",
        help="Print only the full paths of state files with floating objects",
    )
    add_common_args(print_raw_parser)

    # Visualize command (open state inspector for each directory)
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Visualize experiments with floating objects using mujoco_state_inspector",
    )
    add_common_args(visualize_parser)

    args = parser.parse_args()

    if args.command == "visualize":
        visualize_floating_experiments(runs_dir=args.directory)
    elif args.command == "print_raw":
        print_raw_floating_files(runs_dir=args.directory)
    elif args.command == "print":
        print_floating_info(runs_dir=args.directory)
    elif args.command == "process":
        run(
            runs_dir=args.directory,
            verbose=getattr(args, "verbose", False),
            num_workers=getattr(args, "workers", None),
        )
