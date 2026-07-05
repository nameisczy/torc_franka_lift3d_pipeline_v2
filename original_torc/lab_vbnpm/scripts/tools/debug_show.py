import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import subprocess
from pathlib import Path
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def show_files(file_paths: List[Path]):
    # Load dep graphs first since they take the longest
    for file_path in file_paths:
        extension = file_path.suffix.lower()
        if extension in [".depgraph"]:
            print(f"  {file_path.name} - dep graph")
            subprocess.Popen(
                f"{SCRIPT_DIR}/../../debug_depgraph.sh -f {file_path}",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    # Then load images
    for file_path in file_paths:
        extension = file_path.suffix.lower()
        if extension in [".jpg", ".png"]:
            print(f"  {file_path.name} - image")
            fig = plt.figure(file_path.stem)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(mpimg.imread(file_path))

    plt.show()


def run():
    parser = argparse.ArgumentParser(
        prog="debug_show.sh", epilog="Displays files under a directory."
    )
    subparsers = parser.add_subparsers(dest="cmd")

    dir_parser = subparsers.add_parser("dir")
    dir_parser.add_argument(
        "dir",
        type=str,
        help="Directory containing files to render.",
    )
    dir_parser.add_argument(
        "-r",
        "--reset",
        action="store_true",
        help="Clears the directory if --dir is used.",
    )
    dir_parser.add_argument(
        "-f",
        "--files",
        type=str,
        nargs="+",
        default=[],
        help="Files under directory to use.",
    )

    files_parser = subparsers.add_parser("files")
    files_parser.add_argument(
        "files", nargs="+", type=str, default=[], help="Files to use."
    )
    args = parser.parse_args()

    if hasattr(args, 'reset') and args.reset:
        print(f"📂 Resetting directory: {args.dir}")
        subprocess.run(f"rm -rf {args.dir}", shell=True, capture_output=True)
        subprocess.run(f"mkdir -p {args.dir}", shell=True, capture_output=True)
    else:
        try:
            file_paths: List[Path] = []
            if args.cmd == "dir":
                if not os.path.isdir(args.dir):
                    print(f"🛑 Error: Expected dir: '{args.dir}' to be a directory!")
                    return
                print(f"📷 Showing files under:\n  {args.dir}")
                files_set = set(args.files)
                if args.files:
                    print(f"Filtered files:\n  {args.files}")
                print()
                for filename in os.listdir(args.dir):
                    file_path = Path.joinpath(Path(args.dir), Path(filename))
                    if file_path.is_file() and (
                        filename in files_set or not args.files
                    ):
                        file_paths.append(file_path)
            elif args.cmd == "files":
                print(f"📷 Showing files:\n  {args.files}")
                for file_path_str in args.files:
                    file_path = Path(file_path_str)
                    if file_path.is_file():
                        file_paths.append(file_path)

            show_files(file_paths)
        except:
            pass
        print("🛑 Exiting...")


if __name__ == "__main__":
    run()
