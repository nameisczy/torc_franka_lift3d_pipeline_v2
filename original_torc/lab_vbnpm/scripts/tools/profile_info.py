import argparse
import pstats
from typing import Optional
from line_profiler import LineStats
from pathlib import Path
from dataclasses import dataclass

import line_profiler


def main():
    @dataclass
    class Args:
        experiment_directory: str
        output: Optional[str] = None

    parser = argparse.ArgumentParser(
        description="Parse .prof files from an experiment directory"
    )
    parser.add_argument(
        "experiment_directory",
        type=str,
        help="Path to the experiment directory containing .prof files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file for combined statistics (optional)",
    )

    args = Args(**parser.parse_args().__dict__)

    # Get all .prof files in directory
    prof_dir = Path(args.experiment_directory)
    if not prof_dir.exists():
        print(f"Error: Directory '{args.experiment_directory}' does not exist")
        return

    # glob *.prof or *.cprof
    def try_dump_cprof_stats():
        cprof_files = sorted(prof_dir.glob("{*.prof,*.cprof}"))
        if not cprof_files:
            return False

        print(f"Found {len(cprof_files)} .prof/.cprof files from cProfile")

        # Create combined stats object
        combined_stats = None
        for prof_file in cprof_files:
            print(f"Loading {prof_file.name}...")
            stats = pstats.Stats(str(prof_file))

            if combined_stats is None:
                combined_stats = stats
            else:
                combined_stats.add(stats)

        print("\n" + "=" * 80)
        print("Combined cProfile Statistics")
        print("=" * 80)
        combined_stats.sort_stats("cumulative")
        combined_stats.print_stats(20)
        combined_stats.print_callers(20)

        if args.output:
            combined_stats.dump_stats(args.output)
            print(f"\nStats saved to {args.output}")

        return True

    def try_dump_line_prof_stats():
        ln_prof_files = sorted(prof_dir.glob("*.ln_prof"))
        if not ln_prof_files:
            return False

        print(f"Found {len(ln_prof_files)} .ln_prof files from line_profiler")

        combined_stats: LineStats = LineStats.from_files(*ln_prof_files)

        print("\n" + "=" * 80)
        print("Combined Line Profiler Statistics")
        print("=" * 80)
        combined_stats.print(rich=True, output_unit=10**-3)

        if args.output:
            combined_stats.to_file(args.output)
            print(f"\nStats saved to {args.output}")

        return True

    try_dump_cprof_stats()
    try_dump_line_prof_stats()


if __name__ == "__main__":
    main()
