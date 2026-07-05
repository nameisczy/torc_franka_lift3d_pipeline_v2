import line_profiler
import cProfile
import sys
import json
import time
import os
import argparse
import traceback
from enum import Enum
from typing import Optional, Any
from datetime import datetime
from dataclasses import dataclass
import zmq

if __name__ != "__main__":
    print("Error: This script must be run and not imported!")
    exit(1)


USER_PREFIX = ""
if "USER" in os.environ:
    USER_PREFIX = os.environ["USER"] + "_"


class Method(str, Enum):
    HUMAN = "human"
    VLM_ONLY = "vlm_only"
    DG_ONLY = "dg_only"
    DG_INTO_VLM = "dg_into_vlm"
    VLM_DG = "vlm_dg"
    RANDOM = "random"


class Profiler(str, Enum):
    CPROFILE = "c_profile"
    LINE_PROFILER = "line_profiler"


@dataclass
class Args:
    method: Method
    sim: bool = False
    ground_truth: bool = False
    rosbag: bool = False
    debug_show: bool = False
    vlm: bool = True
    in_file: str = None
    out_file: str = None
    target_object: Optional[str] = None
    target_object_id: Optional[int] = None
    pick_limit: int = 15
    server: bool = False
    server_address: Optional[str] = None
    profile_grasping: bool = False
    profiler: Profiler = Profiler.LINE_PROFILER


parser = argparse.ArgumentParser(epilog="Controls grasping with Curobo.")
parser.add_argument(
    "--method",
    default=Method.HUMAN,
    choices=list([method.value for method in Method]),
    help="Decision making method to use.",
)
parser.add_argument(
    "--sim", action="store_true", help="Uses a simulation to run the experiments."
)
parser.add_argument(
    "--ground-truth",
    action="store_true",
    help="Uses the ground truth segmentation data in simulation.",
)
parser.add_argument(
    "--rosbag",
    action="store_true",
    help="Collects rosbag data (takes 60 GB of space per bag).",
)
parser.add_argument(
    "--debug-show",
    action="store_true",
    help="Displays debug diagrams in-between decisions.",
)
parser.add_argument(
    "--vlm",
    action="store_true",
    help="Uses VLM for decision making (if applicable).",
)
parser.add_argument(
    "--in-file",
    type=str,
    help="File to use for human input. If none, then the CLI will prompt for input.",
)
parser.add_argument(
    "--out-file",
    type=str,
    help="File to store logging data. If none, then stderr is used.",
)
parser.add_argument(
    "--target-object",
    type=str,
    help="Name of object to retrieve.",
)
parser.add_argument(
    "--target-object-id",
    type=str,
    help="Id of object to retrieve.",
)
parser.add_argument(
    "--pick-limit",
    type=int,
    default=15,
    help="Maximum number of picks, before this process ends. The process will also end if the target object is picked. If = -1, then the picking loop is ran forever. If = 0, then no picking is done and the process ends after running sensing and grasping.",
)
parser.add_argument(
    "--server", action="store_true", help="Runs the Curobo control as a server."
)
parser.add_argument(
    "--server-address",
    type=str,
    default="tcp://*:5757",
    help="Address of socket to bind the server to. Ex. tcp://*:5757",
)
parser.add_argument(
    "--profile-grasping",
    action="store_true",
    help="Profiles the grasping choice function and logs the results.",
)
parser.add_argument(
    "--profiler",
    type=str,
    default=Profiler.LINE_PROFILER,
    choices=list([profiler.value for profiler in Profiler]),
    help="Profiler to use for grasping choice function.",
)
args = Args(**parser.parse_args().__dict__)

robot = None
perception = None
planner = None
grasp_planner = None
prompter = None


def run(args: Args) -> str:
    # region Arg Parsing
    init_t0 = time.time()

    if args.in_file and not os.path.exists(args.in_file):
        print(f"Error: in_file {args.in_file} does not exist!")
        return "error"
    if args.out_file and not args.out_file.endswith(".csv"):
        print("Error: out_file must end with .csv if it's set!")
        return "error"

    OUT_FILE_DIR = os.path.dirname(args.out_file) if args.out_file else None
    INFO_FILE = (
        os.path.join(OUT_FILE_DIR, "info_curobo_control.json") if OUT_FILE_DIR else None
    )

    def print_args():
        print("🤖 Curobo Runner")
        max_arg_len = max(len(s) for s in args.__dict__)
        for arg, value in args.__dict__.items():
            print(f"  {arg.ljust(max_arg_len)}  {value}")

    print_args()

    if INFO_FILE:
        with open(INFO_FILE, "w") as file:
            now = datetime.now()
            timestamp = now.strftime("%Y/%m/%d %H:%M:%S")

            info = {
                "cmd": " ".join(sys.argv),
                "args": args.__dict__,
                "unix_timestamp": now.timestamp(),
                "timestamp": timestamp,
            }
            json.dump(info, file, indent=2)
    # endregion

    # region Pre Init
    init_t1 = time.time()
    import signal
    import traceback
    import subprocess
    from pathlib import Path

    from rospkg import RosPack

    def load_process_pid(process_name: str) -> int:
        pid_file_path = f"/tmp/{USER_PREFIX}{process_name}.pid"
        try:
            with open(pid_file_path, "r") as f:
                return int(f.read())
        except (IOError, ValueError):
            return -1

    def set_process_pid(process_name: str, value: int):
        pid_file_path = f"/tmp/{USER_PREFIX}{process_name}.pid"
        with open(pid_file_path, "w") as f:
            f.write(str(value))

    def try_kill_process(process_name: str):
        pid = load_process_pid(process_name)
        if pid > -1:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        set_process_pid(process_name, -1)

    def try_kill_processes():
        try_kill_process("debug_show")
        try_kill_process("record_rosbag")

    def mark_unfinished():
        Path(f"/tmp/{USER_PREFIX}curobo_control_finished").unlink(missing_ok=True)

    def mark_finished():
        with open(f"/tmp/{USER_PREFIX}curobo_control_finished", "w") as file:
            file.write("True")

    def cleanup():
        print("Cleanup and exiting...")
        try_kill_processes()
        mark_finished()
        # Explicitly flush the output file
        if out_file:
            out_file.close()

    def handle_term(signum: int, frame):
        print(f"handle_term: {signum} {frame}")
        cleanup()
        exit()

    rp = RosPack()

    ROOT_DIR = os.path.abspath(rp.get_path("lab_vbnpm"))

    DEBUG_SHOW_DIR = f"/tmp/{USER_PREFIX}curobo_runner_debug"
    DEBUG_SHOW_SH = os.path.join(ROOT_DIR, "debug_show.sh")
    RECORD_ROSBAG_SH = os.path.join(ROOT_DIR, "record_rosbag.sh")

    PROMPT_PICK_OBJECT = ""
    PROMPT_PICK_OBJECT_DG = ""
    PROMPT_MAKE_DG = ""

    prompt_templates_dir = os.path.join(
        rp.get_path("lab_vbnpm"), "scripts/task_planner/prompt_templates"
    )
    with open(os.path.join(prompt_templates_dir, "pick_object.txt"), "r") as f:
        PROMPT_PICK_OBJECT = f.read()

    with open(os.path.join(prompt_templates_dir, "pick_object_dg_desc.txt"), "r") as f:
        PROMPT_PICK_OBJECT_DG = f.read()

    with open(os.path.join(prompt_templates_dir, "make_dg.txt"), "r") as f:
        PROMPT_MAKE_DG = f.read()

    ROBOT_RESET_JOINTS = {
        "torso_joint_b1": 0,
        "arm_left_joint_1_s": 1.75,
        "arm_left_joint_2_l": 0.8,
        "arm_left_joint_3_e": 0,
        "arm_left_joint_4_u": -0.66,
        "arm_left_joint_5_r": 0,
        "arm_left_joint_6_b": 0,
        "arm_left_joint_7_t": 0,
        # "arm_right_joint_1_s": 0.75,
        # "arm_right_joint_2_l": 0,
        # "arm_right_joint_3_e": -0.6,
        # "arm_right_joint_4_u": -1.15,
        # "arm_right_joint_5_r": 0,
        # "arm_right_joint_6_b": -1.3,
        # "arm_right_joint_7_t": 0.0,
        "arm_right_joint_1_s": 0.2,
        "arm_right_joint_2_l": -0.7,
        "arm_right_joint_3_e": 0.0,
        "arm_right_joint_4_u": -1.7,
        "arm_right_joint_5_r": 0,
        "arm_right_joint_6_b": -1.3,
        "arm_right_joint_7_t": 0.0,
    }

    in_file = open(args.in_file, "r") if args.in_file else sys.stdin
    out_file = open(args.out_file, "a") if args.out_file else sys.stderr

    signal.signal(signal.SIGQUIT, handle_term)
    signal.signal(signal.SIGINT, handle_term)
    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGHUP, handle_term)

    mark_unfinished()
    try_kill_processes()

    if args.debug_show:
        subprocess.run(
            f"{DEBUG_SHOW_SH} dir {DEBUG_SHOW_DIR} --reset",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    import re
    import pickle

    import cv2
    import numpy as np

    import rospy
    from cv_bridge import CvBridge
    from std_msgs.msg import Int32
    from geometry_msgs.msg import PointStamped
    from sensor_msgs.msg import JointState, Image
    from visualization_msgs.msg import MarkerArray, Marker

    from lab_vbnpm.msg import ObjectIdsToNames
    from task_planner.eutils import ee_open, execute
    from utils.visual_utils import decode_seg_img_rgb
    from grasp_planner.curobo_grasp_planner import GraspPlanner
    from task_planner.prompt import Prompter, create_labeled_image
    from task_planner.curobo_open_loop import open_loop_pick_or_place
    from task_planner.dep_graph import (
        DepGraph,
        get_behind_candidates,
        get_behind_below_dependencies,
    )

    rospy.init_node("curobo_control")
    bridge = CvBridge()

    ## init perception and planning interfaces ##
    init_t2 = time.time()

    def init_robot():
        # Only initialize robot related classes once, since we
        # get CUDA problems if we call init_motion_planner twice,
        # even if they are on separate MotomanSDA10F class instances.
        global robot, perception, planner, grasp_planner
        from task_planner.motoman import MotomanSDA10F

        robot = MotomanSDA10F(args.sim, args.ground_truth)
        perception = robot.init_perception_interface()
        planner = robot.init_motion_planner(planner="curobo")  # ,warmup=False)
        grasp_planner = GraspPlanner(
            robot.curobo_config,
            planner.static_world_config,
            robot.urdf,
            ignore_collision_ee_links=robot.ignore_collision_ee_links,
        )

    def init_vlm():
        global prompter
        if args.vlm:
            prompter = Prompter(
                url="https://generativelanguage.googleapis.com/v1beta/openai/",
                key=os.getenv("GEMINI_API_KEY"),
            )

    if robot is None:
        init_robot()
    else:
        print("******Resetting perception!******")
        perception.tsdf_vol.reset_all()

    if prompter is None:
        init_vlm()

    init_t3 = time.time()

    marker_pub = rospy.Publisher("/plot_grasps", MarkerArray, queue_size=5)
    # endregion

    # region Functions
    def select_place_point(height):
        print(
            """

            Please select a point to place the object
    using Rviz's point publisher tool.
    """
        )
        print(""" rospy.wait_for_message("/clicked_point", PointStamped) """)
        msg = rospy.wait_for_message("/clicked_point", PointStamped)
        print("  '- SUCCESS")
        x, y, z = msg.point.x, msg.point.y, msg.point.z
        z += height
        print(f"Clicked point: ({x}, {y}, {z})")
        marker = MarkerArray()
        marker.markers.append(Marker())
        marker.markers[-1].header.frame_id = "world"
        marker.markers[-1].header.stamp = rospy.Time()
        marker.markers[-1].ns = "clicked_points"
        marker.markers[-1].id = 0
        marker.markers[-1].type = Marker.SPHERE
        marker.markers[-1].action = Marker.ADD
        marker.markers[-1].pose.position.x = x
        marker.markers[-1].pose.position.y = y
        marker.markers[-1].pose.position.z = z
        marker.markers[-1].scale.x = 0.1
        marker.markers[-1].scale.y = 0.1
        marker.markers[-1].scale.z = 0.1
        marker.markers[-1].color.a = 1
        marker.markers[-1].color.r = 1
        marker.markers[-1].color.g = 1
        marker.markers[-1].color.b = 0
        marker_pub.publish(marker)
        return [x, y, z]

    def vlm_make_dg(img_labeled, obj_depths, dg_args):
        global prompter
        nonlocal pick_count
        if prompter is not None:
            # depth_str = ""
            # mx0 = max(obj_depths.values(), key=lambda x: x[1])
            # mx1 = max(obj_depths.values(), key=lambda x: x[0])
            # mn0 = min(obj_depths.values(), key=lambda x: x[1])
            # mn1 = min(obj_depths.values(), key=lambda x: x[0])
            # mx = max(mx0[1], mx1[1])
            # mn = min(mn0[0], mn1[0])
            # for k, v in obj_depths.items():
            #     min_d = 100 * (v[0] - mn) / (mx - mn)
            #     max_d = 100 * (v[1] - mn) / (mx - mn)
            #     depth_str += (
            #         f"  * Object {k}: min depth {min_d:.2f}, max depth {max_d:.2f}\n"
            #     )
            # prompt_text = PROMPT_MAKE_DG.replace("<object_depths>", depth_str)

            seg = perception.cur_mask
            dep = perception.cur_depth
            candidate_edges = get_behind_candidates(seg, dep, obj_depths.keys())
            candidates = ""
            for s, t in candidate_edges:
                candidates += f"  * Object {s} is behind Object {t}\n"
            prompt_text = PROMPT_MAKE_DG.replace("<candidates>", candidates)

            result = prompter.prompt_model(prompt_text, images=[img_labeled])
            print("\n\nVLM Reply:", result)
            try:
                str_json = re.findall("(?<=```json)[\s\S]*(?=```)", result)[0]
                edges_dict = json.loads(str_json)
                print("VLM Parsed JSON:", edges_dict)
            except:
                print("Error: Failed to parse JSON from VLM.")
                return None
            grasp_collisions = dg_args[0]
            obj_ids = dg_args[1]
            pickable = dg_args[2]
            vis_pts = dg_args[3]
            all_pts = dg_args[4]
            vis_mask = dg_args[5]
            all_mask = dg_args[6]
            dep_graph = DepGraph.from_edges(
                grasp_collisions,
                obj_ids,
                pickable,
                edges_dict,
                all_pts,
                all_mask,
            )
            dep_graph.keep_only(obj_depths.keys())
            dep_graph.normalize_edges()
            # visualize
            # dep_graph.draw(
            #     to_show=False, block=False, fname=f"{DEBUG_SHOW_DIR}/vlm_dep_graph.png"
            # )
            if args.debug_show:
                with open(f"{DEBUG_SHOW_DIR}/vlm_dep_graph.depgraph", "wb") as file:
                    pickle.dump(dep_graph, file)
            if OUT_FILE_DIR:
                with open(
                    f"{OUT_FILE_DIR}/vlm_dep_graph_{pick_count}.depgraph", "wb"
                ) as file:
                    pickle.dump(dep_graph, file)
            dep_graph.add_hidden_edges()
            return dep_graph
        else:
            print("Error: No API key provided.")
            return None

    def prepare_for_choice():
        nonlocal pick_count
        img = perception.cur_color
        seg = perception.cur_mask
        dep = perception.cur_depth
        # seg_filtered = np.zeros_like(seg, dtype=np.uint32)
        # for i in obj_ids:
        #     seg_filtered |= seg & (1 << i)
        # img_labeled, labels = create_labeled_image(img, seg_filtered)
        img_labeled, obj_depths = create_labeled_image(img, dep, seg)
        img_back = None
        if args.debug_show:
            cv2.imwrite(f"{DEBUG_SHOW_DIR}/img_labeled.png", img_labeled)
            # save back image if human running simulation
            if robot.is_sim and args.method == Method.HUMAN:
                try:
                    img_msg = rospy.wait_for_message(
                        "/camera2/color/image_raw", Image, timeout=10
                    )
                    dep_msg = rospy.wait_for_message(
                        "/camera2/aligned_depth_to_color/image_raw", Image, timeout=10
                    )
                    seg_msg = rospy.wait_for_message(
                        "/ground_truth/camera2/seg_image", Image, timeout=10
                    )
                    image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
                    depth = bridge.imgmsg_to_cv2(dep_msg, "32FC1")
                    segrw = bridge.imgmsg_to_cv2(seg_msg, "rgb8")
                    segmt = decode_seg_img_rgb(segrw)
                    mask = (segmt == int(target_object_id)).astype(np.uint32)
                    img_back, _obj_d = create_labeled_image(image, depth, mask)
                    cv2.imwrite(f"{DEBUG_SHOW_DIR}/back.png", img_back)
                except Exception as e:
                    print(e)
        if OUT_FILE_DIR:
            cv2.imwrite(f"{OUT_FILE_DIR}/img_labeled_{pick_count}.png", img_labeled)
            cv2.imwrite(f"{OUT_FILE_DIR}/color_{pick_count}.png", img)
            cv2.imwrite(
                f"{OUT_FILE_DIR}/depth_{pick_count}.png", (1000 * dep).astype(np.uint16)
            )
            np.save(f"{OUT_FILE_DIR}/mask_{pick_count}.npy", seg)
            if img_back is not None:
                cv2.imwrite(f"{OUT_FILE_DIR}/back.png", img_back)
        return img_labeled, obj_depths

    def vlm_grasp_choice(img_labeled):
        global prompter
        if prompter is not None:
            prompt_text = PROMPT_PICK_OBJECT.replace("<target>", "0")
            result = prompter.prompt_model(prompt_text, images=[img_labeled])
            print("\n\nVLM Reply:", result)
            try:
                str_json = re.findall("(?<=```json)[\s\S]*(?=```)", result)[0]
                out_json = json.loads(str_json)
                print("VLM Parsed JSON:", out_json)
                return int(out_json[0]["input"])
            except:
                print("Error: Failed to parse JSON from VLM.")
                return -1
        else:
            print("Error: No API key provided.")
            return -1

    def human_grasp_choice(graspable_ids):
        print(
            f"\n\nThe robot can grasp objects {graspable_ids}.\n",
            "Which will help towards retrieving object 0?",
            end="\n> ",
        )
        try:
            return int(in_file.readline().strip())
        except:
            print("Error: Expecting an integer input.")
            print(in_file.readable())
            return -1

    def dg_grasp_choice(obj_depths, dg_args):
        seg = perception.cur_mask
        dep = perception.cur_depth
        # dep_graph = DepGraph.from_geometry(*dg_args)
        cam_intr = np.array(list(perception.cam_info.values())[0].K).reshape((3, 3))
        edges_dict = get_behind_below_dependencies(
            seg, dep, cam_intr, obj_depths.keys()
        )
        grasp_collisions = dg_args[0]
        obj_ids = dg_args[1]
        pickable = dg_args[2]
        vis_pts = dg_args[3]
        all_pts = dg_args[4]
        vis_mask = dg_args[5]
        all_mask = dg_args[6]
        dep_graph = DepGraph.from_edges(
            grasp_collisions,
            obj_ids,
            pickable,
            edges_dict,
            all_pts,
            all_mask,
        )
        dep_graph.keep_only(obj_depths.keys())
        dep_graph.normalize_edges()
        if args.debug_show:
            with open(f"{DEBUG_SHOW_DIR}/dep_graph.depgraph", "wb") as file:
                pickle.dump(dep_graph, file)
        if OUT_FILE_DIR:
            with open(f"{OUT_FILE_DIR}/dep_graph_{pick_count}.depgraph", "wb") as file:
                pickle.dump(dep_graph, file)
        dep_graph.add_hidden_edges()
        return dep_graph

    def dg_get_choice(dep_graph):
        if dep_graph is None:
            return -1
        sinks, probs = dep_graph.sinks()
        print({s: p for s, p in zip(sinks, probs)})
        if len(sinks) > 0:
            return sinks[np.argmax(probs)]
        return -1

    def vlm_wdg_grasp_choice(img_labeled, obj_depths, dg_args):
        global prompter
        dep_graph = DepGraph.from_grasps(*dg_args[:3])
        dep_graph.keep_only(obj_depths.keys())
        dep_graph.normalize_edges()
        if args.debug_show:
            with open(f"{DEBUG_SHOW_DIR}/dep_graph_grasp_only.depgraph", "wb") as file:
                pickle.dump(dep_graph, file)
        if OUT_FILE_DIR:
            with open(
                f"{OUT_FILE_DIR}/dep_graph_grasp_only_{pick_count}.depgraph", "wb"
            ) as file:
                pickle.dump(dep_graph, file)

        prompt_text = PROMPT_PICK_OBJECT_DG.replace("<target>", "0")
        # G = copy.deepcopy(dep_graph.nx_graph)
        # for e in G.edges:
        #     del G.edges[e]['w']
        # graphml = DepGraph.gen_graphml(G)
        # prompt_text = prompt_text.replace('<graph>', graphml)
        description = dep_graph.describe()
        prompt_text = prompt_text.replace("-1", "0")
        prompt_text = prompt_text.replace("<scene description>", description)
        if prompter is not None:
            result = prompter.prompt_model(prompt_text, images=[img_labeled])
            print("\n\nVLM+DG Reply:", result)
            try:
                str_json = re.findall("(?<=```json)[\s\S]*(?=```)", result)[0]
                out_json = json.loads(str_json)
                print("VLM+DG Parsed JSON:", out_json)
                return int(out_json[0]["input"])
            except:
                print("Error: Failed to parse JSON from VLM.")
                return -1
        else:
            print("Error: No API key provided.")
            return -1

    def grasp_choice(obj_ids, dg_args):
        # No grasping is done if pick_limit == 0
        # we only want sensing
        if args.pick_limit == 0:
            # create DG and labeled image
            img_labeled, obj_depths = prepare_for_choice()
            return -1

        # Get next available numbered file name
        def get_next_file_name(prefix: str, suffix: str) -> str:
            i = 1
            while True:
                file_name = f"{OUT_FILE_DIR}/{prefix}{i}{suffix}"
                if not os.path.exists(file_name):
                    return file_name
                i += 1

        def try_choose_grasp() -> int:
            # create DG and labeled image
            img_labeled, obj_depths = prepare_for_choice()
            # dep graph choice
            dg_choice = -1
            if args.method == Method.DG_ONLY:
                dep_graph = dg_grasp_choice(obj_depths, dg_args)
                dg_choice = dg_get_choice(dep_graph)
            # vlm choice
            vlm_choice = -1
            if args.method == Method.VLM_ONLY:
                vlm_choice = vlm_grasp_choice(img_labeled)
            # vlm + dg choice
            vlm_wdg_choice = -1
            if args.method == Method.DG_INTO_VLM:
                vlm_wdg_choice = vlm_wdg_grasp_choice(img_labeled, obj_depths, dg_args)
            # dg by vlm choice
            vlm_dg_choice = -1
            if args.method == Method.VLM_DG:
                vlm_dep_graph = vlm_make_dg(img_labeled, obj_depths, dg_args)
                if vlm_dep_graph is not None:
                    vlm_dg_choice = dg_get_choice(vlm_dep_graph)
            # random choice
            if args.method == Method.RANDOM:
                options = set(dg_args[2].keys()) & set(obj_depths.keys())
                if len(options) == 0:
                    random_choice = -1
                else:
                    if 0 in options:
                        random_choice = 0
                    else:
                        random_choice = np.random.choice(list(options))

            # human choice
            # if args.method == Method.HUMAN:
            #     print(
            #         f'\n\n The DG choice: "{dg_choice}"',
            #         f'\n The VLM choice: "{vlm_choice}"',
            #         f'\n The DG->VLM choice: "{vlm_wdg_choice}"',
            #         f'\n The VLM->DG choice: "{vlm_dg_choice}"',
            #     )

            if args.debug_show:
                # Stop showing previous debug images
                try_kill_process("debug_show")

                # Show the debug images
                debug_show = subprocess.Popen(
                    f"{DEBUG_SHOW_SH} dir {DEBUG_SHOW_DIR}",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                print(
                    f"START DEBUG SHOW: debug_show.pid: {debug_show.pid} my pid: {os.getpid()} debug_show.grpid: {os.getpgid(debug_show.pid)} my grpid: {os.getpgid(os.getpid())}"
                )
                set_process_pid("debug_show", debug_show.pid)

            if args.method == Method.HUMAN:
                graspable_ids = set(dg_args[2].keys()) & set(obj_depths.keys())
                human_choice = human_grasp_choice(graspable_ids)

            if args.method == Method.HUMAN:
                chosen_id = human_choice
            elif args.method == Method.DG_ONLY:
                chosen_id = dg_choice
            elif args.method == Method.VLM_ONLY:
                chosen_id = vlm_choice
            elif args.method == Method.DG_INTO_VLM:
                chosen_id = vlm_wdg_choice
            elif args.method == Method.VLM_DG:
                chosen_id = vlm_dg_choice
            elif args.method == Method.RANDOM:
                chosen_id = random_choice

            print("Chosen: ", chosen_id)
            # input("Press Enter to confirm...")
            print("Grasp Choice (Seg ID)", chosen_id, sep=",", file=out_file)

            # Invalid object choice, choosing a number that
            # is not in the list of valid grasps
            if chosen_id not in obj_ids:
                print(f"Object {chosen_id} is not graspable, please choose again.")
                print(f"    obj_ids: {obj_ids}")
                return -1

            # Return the object we want to grasp
            return chosen_id

        # Try 3 times
        for i in range(3):
            grasp_t0 = time.time()

            if args.profile_grasping:
                if args.profiler == Profiler.CPROFILE:
                    profiler = cProfile.Profile()
                    profiler.enable()
                    chosen_id = try_choose_grasp()
                elif args.profiler == Profiler.LINE_PROFILER:
                    profiler = line_profiler.LineProfiler()
                    profiler.add_function(vlm_grasp_choice)
                    profiler.add_function(human_grasp_choice)
                    profiler.add_function(dg_grasp_choice)
                    profiler.add_function(dg_get_choice)
                    profiler.add_function(vlm_wdg_grasp_choice)
                    profiler.add_function(vlm_make_dg)
                    profiler.add_function(prepare_for_choice)
                    profiler.add_function(get_behind_below_dependencies)
                    chosen_id = profiler(try_choose_grasp)()
            else:
                chosen_id = try_choose_grasp()

            if args.profile_grasping:
                if OUT_FILE_DIR:
                    if args.profiler == Profiler.CPROFILE:
                        profiler.dump_stats(
                            get_next_file_name("grasp_choice_profile_", ".cprof")
                        )
                    elif args.profiler == Profiler.LINE_PROFILER:
                        profiler.dump_stats(
                            get_next_file_name("grasp_choice_profile_", ".ln_prof")
                        )
                else:
                    profiler.print_stats()

            grasp_t1 = time.time()
            print("Grasp Choice Duration", grasp_t1 - grasp_t0, sep=",", file=out_file)

            if chosen_id >= 0:
                return chosen_id

        print("Grasp choice failed...")
        return -1

    # endregion

    # region Post Init
    init_t4 = time.time()

    if robot.is_sim:
        print(
            """ rospy.wait_for_message("/ground_truth/object_ids_to_names", ObjectIdsToNames) """
        )
        _msg = rospy.wait_for_message(
            "/ground_truth/object_ids_to_names", ObjectIdsToNames
        )
        print("  '- SUCCESS: ", _msg)
        OBJECT_ID_TO_NAME = dict(zip(_msg.obj_ids, _msg.names))
        OBJECT_NAME_TO_ID = dict(zip(_msg.names, _msg.obj_ids))

        print("ARGS: ", args)

        if args.target_object:
            args.target_object_id = OBJECT_NAME_TO_ID[args.target_object]

        if args.target_object_id is None:
            # Prompt for target object
            if args.sim and in_file == sys.stdin:
                target_object_id = 0
                while target_object_id == 0:
                    print(
                        "\n\nDouble click the object you want to pick in mujoco viewer."
                    )
                    print("Once an object is chosen hit enter.")
                    in_file.readline()
                    target_object_id = rospy.wait_for_message(
                        "/ground_truth/selected_object_id", Int32
                    ).data
                target_object_id = str(target_object_id)
            else:
                print("\n\nWhat do you want to retrieve?", end="\n> ")
                target_object_id = in_file.readline().strip()
        else:
            # Use target object specified
            target_object_id = str(args.target_object_id)
        target_object_name = OBJECT_ID_TO_NAME[int(target_object_id)]
    else:
        if args.target_object is None:
            target_object_name = input("Which object do you want to pick?\n")
        else:
            target_object_name = args.target_object
        target_object_id = str(target_object_name)
    target_object_id = str(target_object_id).strip()
    # endregion

    # region Control Loop
    result = "error"
    try:
        init_t5 = time.time()

        print("Init Import Duration", init_t2 - init_t1, sep=",", file=out_file)
        print(
            "Init Perception Planning Duration",
            init_t3 - init_t2,
            sep=",",
            file=out_file,
        )
        print("Init Functions Duration", init_t4 - init_t3, sep=",", file=out_file)
        print("Init Post Duration", init_t5 - init_t4, sep=",", file=out_file)
        print("Total Init Duration", init_t5 - init_t0, sep=",", file=out_file)

        print(f"Total Init Duration: {init_t5 - init_t0}s")
        print("Start control loop:")
        print(f"  target_object:    {target_object_name}")
        print(f"  target_object_id: {target_object_id}")
        pick_count = 0
        if args.rosbag and OUT_FILE_DIR:
            rosbag_file = os.path.join(OUT_FILE_DIR, "rosbag")
            print("Starting ROS bag...")
            record_ros_bag = subprocess.Popen(
                f"{RECORD_ROSBAG_SH} {rosbag_file}",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            set_process_pid("record_rosbag", record_ros_bag.pid)
        print("Target Object", target_object_name, sep=",", file=out_file)
        while True:
            pick_count += 1

            # Sleep a little to let the robot settle into it's reseting position.
            max_steps = 100
            min_steps = 10
            c = 0
            total_steps = 0
            while c < min_steps and total_steps < max_steps:
                joint_state = rospy.wait_for_message("/joint_states_all", JointState)
                current_positions = joint_state.position
                desired_positions = [ROBOT_RESET_JOINTS[n] for n in joint_state.name]
                reached = np.allclose(current_positions, desired_positions, atol=0.001)
                if reached:
                    c += 1
                else:
                    c = 0
                total_steps += 1
                rospy.sleep(0.01)
            # is above necessary?
            if not robot.is_sim:
                input("\nReady to sense? Press Enter to continue...")

            print("Pick Count", pick_count, sep=",", file=out_file)

            perception.tsdf_vol.reset_visible()
            result = open_loop_pick_or_place(
                target_object_id,
                robot,
                perception,
                planner,
                grasp_planner,
                place=[0.1, -0.7, 1.2, 0.0, 1.0, 0.0, 0.0],
                num_place_rotations=16,
                lift_height=0.02,
                out_file=out_file,
                grasp_choice=grasp_choice,
            )

            if args.pick_limit == 0:
                break

            print("Grasp Result", result[0], sep=",", file=out_file)

            ee_open()
            joint_state = rospy.wait_for_message(
                "/joint_states_all", JointState, timeout=5
            )
            plan_reset = planner.joint_motion_plan(joint_state, ROBOT_RESET_JOINTS)
            MAX_PLAN_RESET_ATTEMPTS = 5
            plan_reset_attempt = 1
            while plan_reset is None:
                print(
                    f"Replanning reset {plan_reset_attempt}/{MAX_PLAN_RESET_ATTEMPTS}..."
                )
                plan_reset = planner.joint_motion_plan(joint_state, ROBOT_RESET_JOINTS)
                plan_reset_attempt += 1
                if plan_reset_attempt > MAX_PLAN_RESET_ATTEMPTS:
                    raise Exception("Failed to plan reset after multiple attempts.")

            if result[0] == "finished":
                print("Reseting to neutral pose.")
                if not robot.is_sim:
                    print("Press Enter to Execute.")
                    input()

                if robot.is_sim:
                    VEL0 = rospy.get_param("/robot/vel_ang_lim")
                    ACC0 = rospy.get_param("/robot/acc_ang_lim")
                    rospy.set_param("/robot/vel_ang_lim", 600)
                    rospy.set_param("/robot/acc_ang_lim", 8500)
                for plan in [plan_reset]:
                    plan.points[-1].time_from_start = plan.points[0].time_from_start
                    execute(plan, window=0, wait=True, retime=True)
                if robot.is_sim:
                    rospy.set_param("/robot/vel_ang_lim", VEL0)
                    rospy.set_param("/robot/acc_ang_lim", ACC0)

            print(f"pick_count: {pick_count} >= pick_limit: {args.pick_limit}")
            if args.pick_limit >= 1:
                if result[0] == "finished":
                    grasp_success = bool(result[1])
                    grasp_object_id = str(result[2]).strip()
                    target_object_id_no_punctuation = re.sub(
                        r"[^\w\s]", "", target_object_id
                    )
                    dropped_objects = result[3]
                    print(
                        f"Finished: grasp_success: {grasp_success} grasp_object_id: {grasp_object_id} target_object_id: {target_object_id_no_punctuation}"
                    )
                    print(f"  Dropped objects: {dropped_objects}")
                    if (
                        grasp_object_id == target_object_id_no_punctuation
                        and grasp_success
                    ):
                        result = "retrieve_success"
                        print("Retrieved Target Object", True, sep=",", file=out_file)
                        break
                    elif target_object_name in dropped_objects:
                        result = "target_dropped"
                        print("Dropped Target Object", True, sep=",", file=out_file)
                        break
                if pick_count >= args.pick_limit:
                    result = "retrieve_failed"
                    print("Retrieved Target Object", False, sep=",", file=out_file)
                    break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        result = "error"
        exception_str = traceback.format_exc()
        print("ERROR:")
        print(exception_str)
        escaped_str = exception_str.replace('"', '""')
        print("Unhandled Error", f'"{escaped_str}"', sep=",", file=out_file)
    finally:
        cleanup()

    return result
    # endregion


def run_server(address: str):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(address)

    def reply(_type: str, msg: Any = {}):
        socket.send_json({"type": _type, **msg})

    def reply_error(msg: str):
        reply("error", {"msg": str(msg)})

    print("Server intialized...")
    while True:
        req = socket.recv_json()
        cmd = req.get("cmd", "")
        print("Received request:\n", json.dumps(req, indent=2))
        if cmd == "run":
            args = Args(**req["args"])
            result = run(args)
            reply("result", {"result": result})
        elif cmd == "stop":
            reply("success")
            break
        else:
            reply_error(f"Unknown cmd: {cmd}")
            continue

    print("🛑 Server stopping...")


if args.server:
    run_server(args.server_address)
else:
    result = run(args)
    if result == "error":
        exit(1)
