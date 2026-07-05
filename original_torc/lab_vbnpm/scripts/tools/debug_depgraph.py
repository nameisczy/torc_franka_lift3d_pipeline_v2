import argparse
import pickle
import os
import matplotlib.pyplot as plt
from pathlib import Path
from task_planner.dep_graph import DepGraph


def run():
    parser = argparse.ArgumentParser(
        prog="debug_depgraph.sh", epilog="Displays a pickled DepGraph."
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="File containing the pickled DepGraph.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"🛑 Error: Expected file: '{args.file}' to be a file!")
        return

    try:
        print(f"🕸 Showing DepGraph: {args.file}")
        with open(args.file, "rb") as file:
            dep_graph = pickle.load(file)
            path = Path(args.file)
            if isinstance(dep_graph, DepGraph):
                plt.figure(path.stem)
                dep_graph.draw(to_show=True)
    except:
        pass
    print("🛑 Exiting...")


if __name__ == "__main__":
    run()
