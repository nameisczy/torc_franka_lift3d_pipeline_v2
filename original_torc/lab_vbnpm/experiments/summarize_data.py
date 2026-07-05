import csv
import pandas as pd
import argparse
import os
from typing import Type, Any, List, Generator
import typing
import json
from colorama import Fore, Style

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/..")
RUNS_DIR = os.path.join(ROOT_DIR, "experiments/runs")

warnings = []


def exiterr(*msg):
    print("Error: ", *msg)
    exit(1)


class SummarizeError(Exception):
    pass


def warn(msg: str):
    global warnings
    text = f"{Fore.YELLOW}{Style.BRIGHT}{msg}{Style.RESET_ALL}"
    print(text)
    warnings.append(text)


def print_all_warnings():
    global warnings
    if not warnings:
        print("No warnings.")
        return
    print(f"{Fore.YELLOW}{Style.BRIGHT}⚠️  Warnings:{Style.RESET_ALL}")
    for text in warnings:
        print(text)


def process_output_summary_df(df: pd.DataFrame):
    # BUGFIX: Fixed Grasp Result's "grasp_choice_failed" accidentally saved as "g"
    df.replace({"Grasp Result": "g"}, "grasp_choice_failed", inplace=True)


def summarize_experiment(output_dir: str):
    print("Summarize experiment: ".ljust(12), output_dir)
    try:
        output_summary_df = pd.read_csv(f"{output_dir}/output_summary.csv")
    except Exception as e:
        raise SummarizeError(f"Could not read output_summary.csv: {e}")
    try:
        with open(f"{output_dir}/info_experiment.json", "r") as file:
            experiment_info = json.load(file)
    except Exception as e:
        raise SummarizeError(f"Could not read info_experiment.json: {e}")

    process_output_summary_df(output_summary_df)

    # print(output_summary_df)
    # pprint.pp(experiment_info)
    res = {}
    for column in output_summary_df.columns:
        res[column] = ",".join(output_summary_df[column].dropna().astype(str))

    def count_options(field: str, options: int):
        value_counts = output_summary_df[field].value_counts()
        for option in options:
            res[f"{field} {option} Count"] = value_counts.get(option, 0)

    def count(field: str, manual_name: str = None):
        _count = output_summary_df[field].count()
        if manual_name:
            res[manual_name] = _count
        else:
            res[f"{field} Count"] = _count

    count_options(
        "Grasp Result",
        [
            "finished",
            "timeout",
            "no_valid_grasps_found",
            "grasp_choice_failed",
            "no_valid_motion_plan_found",
        ],
    )
    count_options("Grasp Success", [True, False])
    count("Grasp Retracted Objects")
    count("Accidental Dropped Objects")

    res["Pick Count"] = output_summary_df["Pick Count"].dropna().iloc[-1]
    res["Scene"] = os.path.basename(experiment_info["args"]["scene"]).replace(
        ".xml", ""
    )
    res["Experiment Path"] = "/".join(
        experiment_info["args"]["data_dir"].split("/")[-2:]
    )
    res["Method"] = experiment_info["args"]["method"]
    res["Total Experiment Time"] = experiment_info.get("seconds_elapsed", 0)

    # pprint.pp(res)

    df = pd.DataFrame(res, index=[0])
    # print(df.to_string())
    df.to_csv(f"{output_dir}/experiment_summary.csv", index=False)


def summarize_output(output_csv: str):
    print("Summarize output: ".ljust(12), output_csv)
    try:
        with open(output_csv, "r") as file:
            rows = list(csv.reader(file))
        if len(rows) == 0:
            raise SummarizeError("output_csv is empty")
        last_row = rows[-1]
        if last_row[0] == "Unhandled Error":
            raise SummarizeError(
                "output_csv detected an unhandled error in the experiment!"
            )
        elif last_row[0] not in ["Retrieved Target Object", "Dropped Target Object"]:
            raise SummarizeError(
                "output_csv is incomplete. Last row is not 'Retrieved Target Object' or 'Dropped Target Object'."
            )
    except Exception as e:
        raise SummarizeError(f"Failed to read csv: {e}")

    def fix_second_to_last_bug(rows: List[List[str]]):
        picks = []
        curr_pick = {}
        for row in rows:
            if row[0] == "Pick Count":
                if curr_pick:
                    picks.append(curr_pick)
                curr_pick = {}
            curr_pick[row[0]] = row[1]
        meta_data = picks.pop(0)
        picks.append(curr_pick)

        try:
            # Check for graspping target object second-to-last but dropping
            if len(picks) >= 2:
                target_object = meta_data["Target Object"]
                second_last_grasped = str(picks[-2]["Grasped"]).split(",")
                second_last_dropped = str(picks[-2]["Dropped"]).split(",")
                last_dropped = str(picks[-1]["Dropped"]).split(",")
                if (
                    target_object in second_last_grasped
                    and target_object not in second_last_dropped
                    and target_object in last_dropped
                ):
                    warn(
                        f"    ⚠️  Detected second-to-last drop target object bug. Likely not an error @\n       {output_csv}"
                    )
                    # # We grabbed the object in the second to last grasp,
                    # # but it didn't register as a successful retraction and
                    # # did not drop to the ground.
                    # #
                    # # However, in the next grasp, it was detected on the group, and
                    # # the experimented ends with the target_object dropped.
                    # #
                    # # NOTE: This fix can potentially overwrite actual scenarios
                    # #       where the target object was dropped mid retrieval, and
                    # #       the object eventually rolled off of the table in the next
                    # #       pick.
                    # warn(
                    #     f"    ⚠️  Detected second-to-last drop target object bug. Applying fix @\n       {output_csv}"
                    # )
                    # second_last_pick_count = picks[-2]["Pick Count"]
                    # last_pick_count = picks[-1]["Pick Count"]
                    # last_pick_start_index = -1
                    # curr_pick_count = 0
                    # for i in range(len(rows)):
                    #     row = rows[i]
                    #     if row[0] == "Pick Count":
                    #         curr_pick_count = row[1]
                    #         if curr_pick_count == last_pick_count:
                    #             last_pick_start_index = i
                    #             break
                    #     if (
                    #         curr_pick_count == second_last_pick_count
                    #         and row[0] == "Retract Success"
                    #     ):
                    #         row[1] = "True"
                    # # Exclude the last pick info + result row
                    # rows = rows[:last_pick_start_index]
                    # # Mark the experiment as successful
                    # rows.append(["Retrieved Target Object", "True"])
                    # with open(
                    #     os.path.dirname(output_csv) + f"/output_fixed.csv", "w"
                    # ) as file:
                    #     writer = csv.writer(file)
                    #     writer.writerows(rows)
        except:
            pass
        return rows

    rows = fix_second_to_last_bug(rows)

    value_lists = {}
    for row in rows:
        if len(row) == 1:
            # Edge case where data is saved incorrectly:
            # Ex. "Error MotionGenStatus.TRAJOPT_FAIL" instead of "Error,MotionGenStatus.TRAJOPT_FAIL"
            row = row[0].split(" ", 1)
        if row[0] not in value_lists:
            value_lists[row[0]] = []
        value_lists[row[0]].append(row[1])
    output_dir = os.path.basename(os.path.dirname(output_csv))

    def get_typed_value(value: str):
        value_lower = value.lower()
        if value_lower == "true":
            return True
        elif value_lower == "false":
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        if "," in value:
            # Parse as list
            return [get_typed_value(value) for value in value.split(",")]
        return value

    def get_default_value(_type: Type) -> Any:
        if _type == str:
            return ""
        elif _type == int:
            return 0
        elif _type == float:
            return 0.0
        elif _type == bool:
            return False
        return None

    def force_typed_value(value: str, _type: Type, default: Any = None) -> Any:
        if default is None:
            default = get_default_value(_type)

        if _type == str:
            return value
        elif _type == int:
            try:
                return int(value)
            except:
                return default
        elif _type == float:
            try:
                return float(value)
            except:
                return default
        elif _type == bool:
            if value.lower() == "true":
                return True
            if value.lower() == "False":
                return False
            return default

        args = typing.get_args(_type)
        origin = typing.get_origin(_type)
        if origin == list:
            if len(value.strip()) == 0:
                return []
            list_type = args[0]
            return [force_typed_value(x, list_type) for x in value.split(",")]

        return None

    def flatten_recursive(l: list) -> Generator[Any, None, None]:
        for elem in l:
            if isinstance(elem, list):
                yield from flatten_recursive(elem)
            else:
                yield elem

    DEFINED_TYPES = {
        "Grasped": List[str],
        "Dropped": List[str],
        "Grasp Choice (Seg ID)": int,
        "Pick Count": int,
        "Retract Success": bool,
        "Grasp Success": bool,
        "Grasp Result": str,
        "Grasp Score": float,
        "Object Name": str,
    }
    typed_dict = {}
    for field, items in value_lists.items():
        if field in DEFINED_TYPES:
            typed_dict[field] = [
                force_typed_value(item, DEFINED_TYPES[field]) for item in items
            ]
        else:
            typed_dict[field] = [get_typed_value(item) for item in items]
    for _type in DEFINED_TYPES:
        if _type not in typed_dict:
            typed_dict[_type] = []

    typed_dict["Grasp Retracted Objects"] = set(
        flatten_recursive(
            [
                elem
                for elem, success in zip(
                    typed_dict["Grasped"], typed_dict["Retract Success"]
                )
                if success
            ]
        )
    )
    typed_dict["Dropped"] = set(flatten_recursive(typed_dict["Dropped"]))
    typed_dict["Accidental Dropped Objects"] = (
        set(typed_dict["Dropped"]) - typed_dict["Grasp Retracted Objects"]
    )
    # print("typed_lists")
    # pprint.pp(typed_dict)

    for field, value in typed_dict.items():
        if isinstance(value, set):
            typed_dict[field] = list(value)

    series_dict = {k: pd.Series(v) for k, v in typed_dict.items()}
    # print(series_dict)
    df = pd.DataFrame.from_dict(series_dict)

    output_dir = os.path.dirname(output_csv)
    summary_file = f"{output_dir}/output_summary.csv"
    df.to_csv(summary_file, index=False)
    print("   Summary: ".ljust(12), summary_file)


def merge_summaries(dir_path: str, summary_name: str = "output_summary.csv"):
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
    print("   Summary: ".ljust(12), summary_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="summarize_data.sh", epilog="Generates a summary of the data collected"
    )
    parser.add_argument(
        "-f", "--folder", default=RUNS_DIR, help="Folder to recursively process."
    )
    parser.add_argument(
        "-s",
        "--only-summary",
        action="store_true",
        help="Only generate summary csv files.",
    )
    parser.add_argument(
        "-m",
        "--only-merge",
        action="store_true",
        help="Only merge csv files in parent directories.",
    )

    args = parser.parse_args()

    merge = True
    summary = True
    if args.only_merge:
        merge = True
        summary = False
    if args.only_summary:
        merge = False
        summary = True

    for base_dir_path, sub_dirs, files in os.walk(args.folder, topdown=False):
        if summary:
            for file in files:
                file_path = os.path.join(base_dir_path, file)
                if file == "output.csv":
                    try:
                        summarize_output(file_path)
                        summarize_experiment(os.path.dirname(file_path))
                    except SummarizeError as e:
                        print("  Error:", e)
        if merge:
            merge_summaries(base_dir_path, "output_summary.csv")
            merge_summaries(base_dir_path, "experiment_summary.csv")

    print()
    print_all_warnings()
