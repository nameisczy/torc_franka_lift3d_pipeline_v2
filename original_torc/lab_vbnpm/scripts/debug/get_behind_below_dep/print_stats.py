import argparse
import os

from line_profiler import LineProfiler, LineStats


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STATS_PATH = os.path.join(
    SCRIPT_DIR, "get_behind_below_dependencies_results.lprof"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print line_profiler stats from a .lprof file."
    )
    parser.add_argument(
        "stats_path",
        nargs="?",
        default=DEFAULT_STATS_PATH,
        help="Path to the .lprof stats file.",
    )
    args = parser.parse_args()

    stats: LineStats = LineStats.from_files(args.stats_path)
    stats.print(rich=True, output_unit=0.001)


if __name__ == "__main__":
    main()
