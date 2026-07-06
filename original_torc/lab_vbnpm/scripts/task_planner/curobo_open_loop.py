import sys
import time
import os
import json
import traceback
import numpy as np
import transformations as tf
import matplotlib.pyplot as plt

import rospy
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from curobo.wrap.reacher.motion_gen import MotionGenStatus

from utils.print_color import *
from task_planner.dep_graph import DepGraph
from utils.visual_utils import from_color_map
from task_planner.prompt import create_labeled_image
from perception.perception_fast import PerceptionInterface
from grasp_planner.curobo_grasp_planner import GraspPlanner
from utils.conversions import pose_to_matrix, pose_to_list, list_to_pose
from task_planner.eutils import ee_open, ee_close, execute, get_experiment_result
import trimesh  # must import after perception to avoid conflict

# TODO: why is above true‽

from lab_vbnpm.msg import ObjectIdsToNames

FRANKA_GRASP_CONTRACT = (
    "CanonicalGrasp",
    "RobotAdapter",
    "RobotGraspCommand",
)

def _env_flag(name):
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def _stage_probe(stage, detail=""):
    path = os.environ.get("TORC_STAGE_LOG_FILE")
    message = f"[OPEN_LOOP_STAGE {time.time():.6f}] {stage}"
    if detail:
        message += f" | {detail}"
    print(message, flush=True)
    if not path:
        return
    try:
        with open(path, "a") as stage_file:
            print(message, file=stage_file, flush=True)
    except Exception:
        pass


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        return [float(value.x), float(value.y), float(value.z)]
    return value


def _matrix_from_pose_or_none(value):
    if value is None:
        return None
    if hasattr(value, "position"):
        return pose_to_matrix(value)
    return np.asarray(value, dtype=np.float64)


def _get_pick_count_from_out_file(out_file):
    try:
        path = getattr(out_file, "name", None)
        if not path or path in ("<stderr>", "<stdout>") or not os.path.exists(path):
            return None
        count = 0
        with open(path, "r") as f:
            for line in f:
                if line.startswith("Pick Count,"):
                    count = int(line.strip().split(",", 1)[1])
        return count or None
    except Exception:
        return None


def _write_selected_grasp_debug(
    out_file,
    pick_count,
    payload,
    arrays,
):
    def capture_log(message):
        print(f"[TORC_SELECTED_GRASP_CAPTURE] {message}", flush=True)
        out_path_for_log = getattr(out_file, "name", None)
        if out_path_for_log and out_path_for_log not in ("<stderr>", "<stdout>"):
            try:
                log_dir = os.path.join(
                    os.path.dirname(os.path.abspath(out_path_for_log)),
                    "selected_grasp_debug",
                )
                os.makedirs(log_dir, exist_ok=True)
                with open(os.path.join(log_dir, "capture_diagnostics.log"), "a") as f:
                    print(f"{time.time():.6f} {message}", file=f, flush=True)
            except Exception:
                pass

    capture_log(
        "writer entered "
        f"pid={os.getpid()} pick_count={pick_count} "
        f"flag={os.environ.get('TORC_CAPTURE_SELECTED_GRASP')!r}"
    )
    if not _env_flag("TORC_CAPTURE_SELECTED_GRASP"):
        capture_log("early return: TORC_CAPTURE_SELECTED_GRASP disabled")
        return None, None
    out_path = getattr(out_file, "name", None)
    if not out_path or out_path in ("<stderr>", "<stdout>"):
        capture_log(f"early return: invalid out_file.name={out_path!r}")
        return None, None
    exp_dir = os.path.dirname(os.path.abspath(out_path))
    debug_dir = os.path.join(exp_dir, "selected_grasp_debug")
    try:
        os.makedirs(debug_dir, exist_ok=True)
        capture_log(f"debug directory ready path={debug_dir}")
    except Exception:
        capture_log(
            "exception creating debug directory:\n" + traceback.format_exc().rstrip()
        )
        return None, None
    if pick_count is None:
        pick_count = _get_pick_count_from_out_file(out_file) or 0
        capture_log(f"resolved missing pick_count from out_file pick_count={pick_count}")
    prefix = os.path.join(debug_dir, f"pick_{int(pick_count):02d}_selected_grasp_debug")
    json_path = prefix + ".json"
    npz_path = prefix + ".npz"
    payload = dict(payload or {})
    arrays = dict(arrays or {})
    required_payload_keys = [
        "selected_object_segment_id",
        "selected_raw_cgn_index",
        "selected_score",
        "planning",
    ]
    missing_payload_keys = [k for k in required_payload_keys if k not in payload]
    capture_log(
        "writer preparing "
        f"json_path={json_path} npz_path={npz_path} "
        f"payload_keys={sorted(payload.keys())} "
        f"array_keys={sorted(arrays.keys())} "
        f"missing_payload_keys={missing_payload_keys}"
    )
    payload["json_path"] = json_path
    payload["npz_path"] = npz_path
    payload["written_unix_time"] = time.time()
    try:
        capture_log("writing NPZ begin")
        np.savez_compressed(npz_path, **arrays)
        capture_log("writing NPZ done")
        capture_log("writing JSON begin")
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2, default=_json_safe)
            f.write("\n")
        capture_log(f"writing JSON done path={json_path}")
    except Exception:
        capture_log("exception writing selected grasp debug:\n" + traceback.format_exc().rstrip())
        return None, None
    capture_log(f"writer success json={json_path} npz={npz_path}")
    return json_path, npz_path


def _write_original_allpts_capture(out_file, pick_count, payload, arrays, suffix):
    """Save exact original planner inputs for audit only.

    This is gated by TORC_CAPTURE_ORIGINAL_PICK1_ALLPTS and does not alter any
    planner state, candidate ordering, or execution behavior.
    """
    if not _env_flag("TORC_CAPTURE_ORIGINAL_PICK1_ALLPTS"):
        return None, None
    if pick_count not in (None, 1):
        return None, None
    out_path = getattr(out_file, "name", None)
    if not out_path or out_path in ("<stderr>", "<stdout>"):
        return None, None
    exp_dir = os.path.dirname(os.path.abspath(out_path))
    debug_dir = os.path.join(exp_dir, "original_planner_input_debug")
    os.makedirs(debug_dir, exist_ok=True)
    if pick_count is None:
        pick_count = _get_pick_count_from_out_file(out_file) or 1
    prefix = os.path.join(
        debug_dir,
        f"pick_{int(pick_count):02d}_original_planner_input_{suffix}",
    )
    json_path = prefix + ".json"
    npz_path = prefix + ".npz"
    payload = dict(payload or {})
    payload.update(
        {
            "capture_version": "stage3_1k_original_pick1_allpts_capture_v1",
            "behavior_preserving": True,
            "flag": "TORC_CAPTURE_ORIGINAL_PICK1_ALLPTS",
            "pick_count": int(pick_count),
            "json_path": json_path,
            "npz_path": npz_path,
            "written_unix_time": time.time(),
        }
    )
    np.savez_compressed(npz_path, **dict(arrays or {}))
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
        f.write("\n")
    print(f"[TORC_ORIGINAL_ALLPTS_CAPTURE] wrote {json_path} {npz_path}", flush=True)
    return json_path, npz_path


def grasp_object_ids_rviz(grasps, object_ids):
    gs = np.array(grasps)
    oids = np.array(object_ids)

    # marker to delete previous visualization
    marker_publisher = rospy.Publisher(
        "/plot_grasps",
        MarkerArray,
        latch=True,
        queue_size=10,
    )

    objs = set(oids)

    marker = MarkerArray()
    for i in objs:
        avg = np.mean(gs[oids == i, :3, 3], axis=0)

        marker.markers.append(Marker())
        marker.markers[-1].ns = "object"
        marker.markers[-1].id = i
        marker.markers[-1].action = Marker.ADD
        marker.markers[-1].header.frame_id = "world"
        marker.markers[-1].header.stamp = rospy.Time.now()
        marker.markers[-1].type = Marker.TEXT_VIEW_FACING
        marker.markers[-1].scale.z = 0.05
        marker.markers[-1].pose.position.x = avg[0]
        marker.markers[-1].pose.position.y = avg[1]
        marker.markers[-1].pose.position.z = avg[2] + 0.1
        marker.markers[-1].text = f"[{i}]"
        red, green, blue = from_color_map(i, 32)
        marker.markers[-1].color.r = red
        marker.markers[-1].color.g = green
        marker.markers[-1].color.b = blue
        marker.markers[-1].color.a = 1.0

    marker_publisher.publish(marker)


def update_and_get_points(
    object_to_grasp,
    robot,
    perception,
    filter_points=False,
    camera_inds=[0],
    save_debug=False,
    debug_number=0,
):
    t0 = time.time()
    # print('0', flush=True)
    debug_mask = [None] * (max(camera_inds) + 1)
    ground = bool(object_to_grasp)
    for i in camera_inds:
        debug_mask[i] = perception.updated_fused_points(
            robot.camera[i], object_to_grasp, ground=ground
        )
        ground = False
    if 0 in camera_inds and len(camera_inds) > 1:
        # repeat first image in case 2nd grounded
        debug_mask[0] = perception.updated_fused_points(
            robot.camera[0], object_to_grasp, ground=False
        )
    # print('2', flush=True)
    points, colors = perception.get_fused_point_cloud()
    # print('3', flush=True)
    # bg_points, bg_colors = perception.get_fused_bg_point_cloud()
    # print('4', flush=True)
    t2 = time.time()

    # perception.rays_from_occlusion()
    print("Rays Time:", time.time() - t2)

    # print('5', flush=True)
    for i in camera_inds:
        perception.update_occlusion(robot.camera[i])
    # tsdf_vol, occl_vol, color_vol, _m,_s = perception.tsdf_vol.get_volume()
    # print(occl_vol)
    # print(np.histogram(occl_vol.flatten()))
    # print('6', flush=True)
    occluded_points = perception.get_occlusion_points()
    occluded_colors = np.zeros((*occluded_points.shape[:-1], 3))
    occluded_colors[:] = [1, 0, 1]
    t3 = time.time()
    # print('7', flush=True)

    obj_mask = perception.get_object_instance_mask()
    occ_mask = perception.get_object_occlusion_mask()

    # filter combined point cloud
    all_pts = np.concatenate([points, occluded_points])
    all_rgb = np.concatenate([colors, occluded_colors])
    all_mask = np.concatenate([obj_mask, occ_mask])
    # tgt_pts, tgt_rgb = perception.get_fused_target_point_cloud()

    # t00 = time.time()
    # if filter_points:
    #     # print('f7', flush=True)

    #     tree = KDTree(tgt_pts)

    #     unexplored = set(range(len(tgt_pts)))
    #     compoments = []
    #     while len(unexplored) > 0:
    #         explored = set()
    #         frontier = [next(iter(unexplored))]
    #         while frontier is not None and len(frontier) != 0:
    #             close_pts = tree.query_ball_point(tgt_pts[frontier], 0.01)
    #             # print(close_pts)
    #             close_pts_flattened = [pt for pts in close_pts for pt in pts]
    #             explored.update(frontier)
    #             unexplored.difference_update(frontier)
    #             frontier = list(set(close_pts_flattened) - explored)
    #         compoments.append(explored)

    #     biggest = max(compoments, key=len)
    #     print('Before Size:', len(tgt_pts))
    #     tgt_pts = tgt_pts[list(biggest)]
    #     tgt_rgb = tgt_rgb[list(biggest)]
    #     print('After Size:', len(tgt_pts))

    #     # mask = tt.query_ball_point(tgt_pts,0.005,return_length=True)
    #     # print(np.histogram(mask))
    #     # mask = mask > 7
    #     # tgt_pts = tgt_pts[mask]
    #     # tgt_rgb = tgt_rgb[mask]
    #     # pcl = o3d.geometry.PointCloud()
    #     # pcl.points = o3d.utility.Vector3dVector(all_pts)
    #     # pcl.colors = o3d.utility.Vector3dVector(all_rgb)
    #     # pcl, ind = pcl.remove_radius_outlier(10, 0.004)
    #     # pcl, ind = pcl.remove_radius_outlier(10, 0.008)
    #     # pcl, ind = pcl.remove_statistical_outlier(20, 3)
    #     # all_pts = np.array(pcl.points)
    #     # all_rgb = np.array(pcl.colors)
    #     t01 = time.time()
    #     print('Filtering Time:', t01 - t00, flush=True)

    # down sample
    # t00 = time.time()
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(all_pts)
    # resolution = 0.1
    # downpcd = pcd.voxel_down_sample(resolution)
    # ds_all_pts = np.array(downpcd.points)

    t1 = time.time()
    # print('Downsample time:', t1 - t00, flush=True)

    if save_debug in ("video", "all"):
        cameras = [robot.camera[i] for i in camera_inds]
        perception.save_image("/tmp/recording_0000", cameras, debug_number, debug_mask)
    if save_debug in ("pcd", "all") and len(tgt_pts) > 0:
        perception.save_fusion(
            "/tmp/pcd_0000", tgt_pts, tgt_rgb, "target", debug_number
        )
        perception.save_fusion("/tmp/pcd_0000", points, colors, "surface", debug_number)
        perception.save_fusion("/tmp/pcd_0000", all_pts, all_rgb, "all", debug_number)

    print("Fusion Time:", t2 - t0, flush=True)
    print("Occlusion Time:", t3 - t2, flush=True)
    print("Total Perception Time:", t1 - t0, flush=True)
    return points, colors, obj_mask, all_pts, all_rgb, all_mask


def open_loop_pick_or_place(
    object_to_grasp,
    robot,
    perception,
    planner,
    grasp_planner,
    target_augment_scale=0,  # scale factor to augment convex hull of points
    target_augment_scale_dim=2,  # consider distane in 2 or 3 dimensions?
    place=None,
    num_place_rotations=8,
    lift_height=0.01,
    max_grasp_iterations=1,
    max_motion_plan_attempts=0,
    min_grasp_score=0.1,
    visualize=False,
    save_debug=False,
    out_file=sys.stderr,
    grasp_choice=lambda o, s, g: np.argmax(s),
    pick_count=None,
):
    debug_target_pub = rospy.Publisher(
        "/debug/target_points", PointCloud2, queue_size=3, latch=True
    )
    debug_full_pcd = rospy.Publisher(
        "/debug/full_pcd", PointCloud2, queue_size=3, latch=True
    )
    debug_surface_pcd = rospy.Publisher(
        "/debug/surface_pcd", PointCloud2, queue_size=3, latch=True
    )

    if robot.is_sim:
        obj_ids_to_names = rospy.wait_for_message(
            "/ground_truth/object_ids_to_names", ObjectIdsToNames
        )
        object_ids_to_names_dict = dict(
            zip(obj_ids_to_names.obj_ids, obj_ids_to_names.names)
        )

    # print('Staring!')
    comp_t0 = time.time()
    grasp_time = 0
    mplan_time = 0

    ## sense the environment ##
    result = update_and_get_points(object_to_grasp, robot, perception)
    pts, rgb, obj_mask, all_pts, all_rgb, all_mask = result

    tgt_pts = pts[(obj_mask & 1).astype(bool)]
    tgt_rgb = rgb[(obj_mask & 1).astype(bool)]

    target_msg = PerceptionInterface.create_cloud(
        tgt_pts,
        np.linspace([0, 0, 0], [255, 255, 255], len(tgt_pts)),
        255,
        "world",
        rospy.Time.now(),
    )
    debug_target_pub.publish(target_msg)
    surface_msg = PerceptionInterface.create_cloud(
        pts,
        np.linspace([0, 0, 0], [255, 255, 255], len(pts)),
        255,
        "world",
        rospy.Time.now(),
    )
    debug_surface_pcd.publish(surface_msg)
    full_msg = PerceptionInterface.create_cloud(
        all_pts,
        np.linspace([0, 0, 0], [255, 255, 255], len(all_pts)),
        255,
        "world",
        rospy.Time.now(),
    )
    debug_full_pcd.publish(full_msg)

    ## augment target points ##
    if target_augment_scale:
        aug_t0 = time.time()
        viz_t0 = time.time()
        if visualize:
            trimesh.points.PointCloud(tgt_pts, tgt_rgb).show()
        viz_t1 = time.time()
        tgt_mesh = perception.get_shape_estimate(
            tgt_pts,
            tgt_rgb,
            scale=target_augment_scale,
        )
        tgt_pts = trimesh.sample.sample_surface_even(tgt_mesh, 1000)[0]
        tgt_rgb = 100 * np.ones((tgt_pts.shape[0], 3))
        aug_t1 = time.time()
        print("Augmentation Time:", (aug_t1 - aug_t0) - (viz_t1 - viz_t0))
        print(
            "Augmentation Duration",
            (aug_t1 - aug_t0) - (viz_t1 - viz_t0),
            sep=",",
            file=out_file,
        )

    # img = perception.cur_color
    # seg = perception.cur_mask
    # dep = perception.cur_depth
    # img_labeled, visible = create_labeled_image(img, dep, seg)
    cpmask = perception.cur_pmask
    cpoints = perception.cur_points
    # trimesh.points.PointCloud(cpoints).show()
    # unique = list(sorted(set(cpmask)))
    # color_num = list(map(unique.index, cpmask))
    # colors = from_color_map(color_num, 32)
    # trimesh.points.PointCloud(cpoints, colors).show()

    viz_t0 = time.time()
    if visualize:
        # if len(tgt_pts) > 0:
        #     trimesh.points.PointCloud(tgt_pts, tgt_rgb).show()
        trimesh.points.PointCloud(all_pts, all_rgb).show()

        if False:
            res = perception.tsdf_vol.get_volume()
            tsdf_vol, occl_vol, color_vol, mask_vol = res
            occl_en = trimesh.voxel.encoding.DenseEncoding(
                (-100 < occl_vol) & (occl_vol < 0)
            )
            tsdf_en = trimesh.voxel.encoding.DenseEncoding(
                (tsdf_vol > -0.5) & (tsdf_vol < 0.9)
            )
            mask_en = trimesh.voxel.encoding.DenseEncoding(mask_vol != 0)
            # trimesh.voxel.base.VoxelGrid(tsdf_en).show()
            # trimesh.voxel.base.VoxelGrid(occl_en).show()
            # trimesh.voxel.base.VoxelGrid(mask_en).show()

            direct_occl = trimesh.voxel.encoding.DenseEncoding(
                (mask_vol != 0) & (occl_vol < 0)
            )
            trimesh.voxel.base.VoxelGrid(direct_occl).show()

        print(set(obj_mask), max(obj_mask))

        # unique = list(sorted(set(obj_mask)))
        # color_num = list(map(unique.index, obj_mask))
        # colors = from_color_map(color_num, 32)
        # trimesh.points.PointCloud(pts, colors).show()

        # unique = list(sorted(set(all_mask)))
        # color_num = list(map(unique.index, all_mask))
        # colors = from_color_map(color_num, 32)
        # trimesh.points.PointCloud(all_pts, colors).show()

        num_objs = len(np.binary_repr(max(all_mask)))
        geoms = []
        boxes = {}
        occ = {}
        vis = {}
        for i in range(num_objs):
            omask = (all_mask & (1 << i)).astype(bool)
            vmask = (obj_mask & (1 << i)).astype(bool)
            print(i, np.sum(vmask))
            if np.sum(vmask) > 0:
                opcd = trimesh.points.PointCloud(all_pts[omask])
                vpcd = trimesh.points.PointCloud(pts[vmask])
                occ[i] = opcd
                vis[i] = vpcd
                # debug
                box = opcd.convex_hull
                box.visual.face_colors = from_color_map(i, 32)
                vbox = vpcd.bounding_box
                vbox.visual.face_colors = from_color_map(i, 32)
                boxes[i] = np.eye(4)
                boxes[i][:3, 3] = vpcd.centroid
                # geoms.append(box)
                # geoms.append(vbox)
                debug_color = all_rgb[omask]
                debug_color[:, 2] = all_rgb[omask][:, 0]
                debug_color[:, 0] = all_rgb[omask][:, 2]
                debug_color[len(rgb[vmask]) :] = from_color_map(i, 32)
                geoms.append(trimesh.points.PointCloud(all_pts[omask], debug_color))

        from task_planner.dep_graph import vis_depends

        obj_deps = {}
        for i in vis.keys():
            if i not in obj_deps:
                obj_deps[i] = {}
            for j in vis.keys():
                if i != j:  # no self edges
                    print(i, j, ":")
                    behind, below = vis_depends(vis[i], occ[i], vis[j], occ[j])
                    if behind and below:
                        obj_deps[i][j] = {"d": "both"}
                    elif behind:
                        obj_deps[i][j] = {"d": "behind"}
                    elif below:
                        obj_deps[i][j] = {"d": "below"}
        import json

        print(json.dumps(obj_deps, indent=4))
        import networkx as nx

        G = nx.from_dict_of_dicts(obj_deps, create_using=nx.DiGraph)
        grasp_planner.plotter.draw_grasps()
        grasp_object_ids_rviz(
            np.array(list(boxes.values())), np.array(list(boxes.keys()))
        )
        DepGraph.draw_graph(G, block=False)
        input()
        trimesh.Scene(geoms).show()
        plt.close("all")
    viz_t1 = time.time()

    joint_state = rospy.wait_for_message("/joint_states_all", JointState, timeout=5)

    # ---------- Generate grasps ---------- #
    grasp = []
    grasp_iterations = 0
    while len(grasp) == 0:
        grasp_iterations += 1
        if grasp_iterations > max_grasp_iterations:
            return ("no_valid_grasps_found",)
        planner.set_planning_scene(
            all_pts,
            visualize=False,
        )
        # planner.visualize_rviz()
        grasp_t0 = time.time()
        ikres = grasp_planner.get_ik_grasps(
            pts,
            rgb,
            # pts,
            perception.cur_points,
            all_pts,
            # obj_mask,
            perception.cur_pmask,
            seg_label_to_obj_id=perception.seg_label_to_obj_id,
            visualize=True,
            # rviz_func=planner.visualize_spheres_rviz,
        )
        if len(ikres) != 7:
            continue
        ik_v_pose_t = ikres[0]
        ik_joints = ikres[1]
        ik_v_scores = ikres[2]
        ik_v_samples = ikres[3]
        ik_v_obj_ids = ikres[4]
        aug_pts = ikres[5]
        aug_mask = ikres[6]

        tggc0 = time.time()
        grasp_collisions = grasp_planner.get_grasp_collisions(
            ik_v_pose_t,
            ik_joints,
            ik_v_scores,
            ik_v_samples,
            ik_v_obj_ids,
            pts,
            all_pts,
            obj_mask,
            all_mask,
        )
        tggc1 = time.time()
        print("Time to get grasp collisions:", tggc1 - tggc0)

        gres = grasp_planner.validate_grasps(
            ik_v_pose_t,
            ik_joints,
            ik_v_scores,
            ik_v_samples,
            ik_v_obj_ids,
            aug_pts,
            all_pts,
            aug_mask,
            singularity_points=all_pts,
            singularity_mask=all_mask,
        )
        if len(gres) != 7:
            continue

        grasp, p_grasp, grasp_js, p_grasp_js, score, sample, obj_ids = gres

        # Convert grasp, p_grasp, grasp_js, p_grasp_js, score, and obj_ids
        # all into np arrays for easier manipulation
        grasp = np.array(grasp)
        p_grasp = np.array(p_grasp)
        grasp_js = np.array(grasp_js)
        p_grasp_js = np.array(p_grasp_js)
        score = np.array(score)
        obj_ids = np.array(obj_ids)

        # Filter out the grasps that have a score < 0.3
        (grasp_indices_to_keep,) = np.nonzero(score >= min_grasp_score)
        debug_valid_source_indices = np.asarray(
            getattr(grasp_planner, "_torc_selected_grasp_debug", {}).get(
                "validated_candidate_source_indices",
                np.arange(len(score)),
            ),
            dtype=np.int64,
        )
        # Sort grasp_indices to keep by their scores in descending order
        #
        # By sorting all grasp_indices, every time we extract the grasps
        # of a specific object_id, the grasps will automatically be in
        # sorted order from highest to lowest score.
        #
        # Note that [::-1] is used to reverse the sorted array from
        # ascending order to descending order
        print("grasp_indices_to_keep: ", grasp_indices_to_keep)
        grasp_indices_to_keep = grasp_indices_to_keep[
            np.argsort(score[grasp_indices_to_keep])[::-1]
        ]

        grasp = grasp[grasp_indices_to_keep]
        p_grasp = p_grasp[grasp_indices_to_keep]
        grasp_js = grasp_js[grasp_indices_to_keep]
        p_grasp_js = p_grasp_js[grasp_indices_to_keep]
        score = score[grasp_indices_to_keep]
        obj_ids = obj_ids[grasp_indices_to_keep]
        if len(grasp_indices_to_keep) and len(debug_valid_source_indices) >= int(np.max(grasp_indices_to_keep)) + 1:
            debug_sorted_source_indices = debug_valid_source_indices[
                grasp_indices_to_keep
            ]
        else:
            debug_sorted_source_indices = np.full(
                len(grasp),
                -1,
                dtype=np.int64,
            )

        pickable = {}
        for oid in obj_ids:
            pickable[oid] = pickable.get(oid, 0) + 1
        G_args = (
            grasp_collisions,
            ik_v_obj_ids,
            pickable,
            pts,
            all_pts,
            obj_mask,
            all_mask,
        )
        grasp_planner.plotter.draw_grasps()
        grasp_object_ids_rviz(ik_v_pose_t, ik_v_obj_ids)
        grasp_planner.plotter.draw_grasps(grasp, obj_ids)
        # input('Continue?')
        # grasp_planner.plotter.draw_grasps(grasp, score)
        # g_color = np.divide(obj_ids, max(*obj_ids, 1))
        # g_color[np.argmax(score)] = -1.
        # grasp_planner.plotter.draw_grasps(p_grasp, g_color)

        print(f"grasps: ({len(grasp)})")
        print("  score (sorted): ", score)
        print("  obj_ids: ", obj_ids)
        print("  obj_ids (unique): ", set(obj_ids))

        grasp_t1 = time.time()
        grasp_time += grasp_t1 - grasp_t0
        print("Grasp Planning Time:", grasp_t1 - grasp_t0)

    # ---------- Choose object to grasp ---------- #
    # Get object that the user wants to grasp.
    chosen_obj_id = grasp_choice(obj_ids, G_args)
    if chosen_obj_id < 0:
        return ("grasp_choice_failed",)
    # Get the grasp indicies that correspond with that object.
    # This is in ascending score order, since obj_ids is already
    # sorted by score in ascending order.
    (chosen_obj_grasp_indices,) = np.nonzero(obj_ids == chosen_obj_id)

    # ---------- Find a valid motion plan for the object ---------- #
    # Attempt to find a motion plan for this object from
    # all of its valid grasps.
    #
    # We want to try the grasps from highest score to lowest score
    sorted_indices = chosen_obj_grasp_indices.tolist()
    # If we are limiting number of motion plans, then we
    # limit the ind_queue to at most max_motion_plan_attempts elements
    if max_motion_plan_attempts > 0:
        sorted_indices = sorted_indices[:max_motion_plan_attempts]
    plans = []
    selected_debug_paths = (None, None)
    selected_debug_payload = None
    selected_debug_arrays = None
    _stage_probe(
        "candidate motion planning loop begin",
        f"chosen_obj_id={int(chosen_obj_id)} sorted_indices={sorted_indices}",
    )
    for ind in sorted_indices:
        plan_t0 = time.time()
        _stage_probe(
            "candidate begin",
            f"ind={int(ind)} rank={sorted_indices.index(ind)} score={float(score[ind])}",
        )
        ## plan to target ##
        allpts_capture_payload = {
            "stage": "before_plan1_set_planning_scene",
            "scene_name": os.environ.get("TORC_SCENE_NAME"),
            "scene_path": os.environ.get("TORC_SCENE_PATH"),
            "target_object": os.environ.get("TORC_TARGET_OBJECT"),
            "method": os.environ.get("TORC_METHOD"),
            "chosen_segment_id": int(chosen_obj_id),
            "candidate_index_in_sorted_validated_candidates": int(ind),
            "candidate_rank_within_chosen_segment": int(sorted_indices.index(ind)),
            "num_candidates_for_chosen_segment": int(len(chosen_obj_grasp_indices)),
            "num_validated_candidates_after_score_filter": int(len(grasp)),
            "candidate_object_segment_id": int(obj_ids[ind]),
            "candidate_score": float(score[ind]),
            "candidate_raw_cgn_index": int(debug_sorted_source_indices[ind])
            if ind < len(debug_sorted_source_indices)
            else -1,
            "planning_scene_call": "planner.set_planning_scene(all_pts, visualize=False)",
            "pose_motion_plan_call": "planner.pose_motion_plan(joint_state, selected_canonical_command)",
            "all_pts_shape": list(np.asarray(all_pts).shape),
            "pts_shape": list(np.asarray(pts).shape),
            "obj_mask_shape": list(np.asarray(obj_mask).shape),
            "all_mask_shape": list(np.asarray(all_mask).shape),
            "frame_assumption": "same frame used by original TORC CuroboPlanner.set_planning_scene and grasp poses",
            "scientific_semantics": {
                "original_torc_selection_path": True,
                "gt_grasp_used": False,
                "fake_grasp_used": False,
                "collision_checking_disabled": False,
                "planning_skipped": False,
                "franka_used": os.environ.get("TORC_ROBOT", "").strip().lower()
                in ("franka", "panda"),
            },
        }
        allpts_capture_arrays = {
            "all_pts": np.asarray(all_pts, dtype=np.float64),
            "pts": np.asarray(pts, dtype=np.float64),
            "obj_mask": np.asarray(obj_mask),
            "all_mask": np.asarray(all_mask),
            "selected_pregrasp_matrix": np.asarray(
                _matrix_from_pose_or_none(p_grasp[ind]),
                dtype=np.float64,
            ),
            "selected_grasp_matrix": np.asarray(
                _matrix_from_pose_or_none(grasp[ind]),
                dtype=np.float64,
            ),
            "validated_pregrasp_matrices_sorted": np.asarray(
                [_matrix_from_pose_or_none(g) for g in p_grasp],
                dtype=np.float64,
            ),
            "validated_grasp_matrices_sorted": np.asarray(
                [_matrix_from_pose_or_none(g) for g in grasp],
                dtype=np.float64,
            ),
            "validated_scores_sorted": np.asarray(score, dtype=np.float64),
            "validated_object_ids_sorted": np.asarray(obj_ids, dtype=np.int64),
            "validated_source_indices_sorted": np.asarray(
                debug_sorted_source_indices,
                dtype=np.int64,
            ),
        }
        _write_original_allpts_capture(
            out_file,
            pick_count or _get_pick_count_from_out_file(out_file),
            allpts_capture_payload,
            allpts_capture_arrays,
            "before_plan1",
        )
        _stage_probe(
            "plan1 set_planning_scene begin",
            f"ind={int(ind)} all_pts_shape={list(np.asarray(all_pts).shape)}",
        )
        planner.set_planning_scene(
            all_pts,
            visualize=False,
        )
        _stage_probe("plan1 set_planning_scene done", f"ind={int(ind)}")
        # planner.visualize_rviz()
        plan_t1 = time.time()
        print("Plan1 Scene Time:", plan_t1 - plan_t0)

        gmatrix = [pose_to_matrix(g) for g in grasp]
        # grasp_planner.plotter.draw_grasps()
        # grasp_object_ids_rviz(gmatrix, obj_ids)
        grasp_planner.plotter.draw_grasps()
        grasp_planner.plotter.draw_grasps(grasp, score)
        g_color = [0.0] * len(score)
        g_color[ind] = -1.0
        selected_canonical_command = p_grasp[ind]
        grasp_planner.plotter.draw_grasps(p_grasp, g_color)
        is_franka_robot = os.environ.get("TORC_ROBOT", "").strip().lower() in (
            "franka",
            "panda",
        )
        _stage_probe("plan1 pose_motion_plan begin", f"ind={int(ind)}")
        plan1, plan1_result = planner.pose_motion_plan(
            joint_state,
            selected_canonical_command,
            return_all=True,
            # custom_bias_state=JointState.from_numpy(
            #     planner.motion_gen.joint_names,
            #     # np.zeros(len(motion_gen.joint_names)),
            #     np.array([-2.9, -3.1, -1.8, -2.9, -2.3, -3.1, -1.8, -3.1])
            #     # velocities,
            #     # accelerations,
            # )
        )
        _stage_probe(
            "plan1 motion_plan done",
            f"ind={int(ind)} success={plan1 is not None} "
            f"status={getattr(plan1_result, 'status', None)} "
            f"valid_query={getattr(plan1_result, 'valid_query', None)}",
        )
        plan_t2 = time.time()
        print("Plan1 Time:", plan_t2 - plan_t1)
        allpts_capture_payload_after = dict(allpts_capture_payload)
        allpts_capture_payload_after.update(
            {
                "stage": "after_plan1_pose_motion_plan",
                "plan1_success": plan1 is not None,
                "plan1_elapsed_s": float(plan_t2 - plan_t1),
                "plan1_scene_elapsed_s": float(plan_t1 - plan_t0),
            }
        )
        _write_original_allpts_capture(
            out_file,
            pick_count or _get_pick_count_from_out_file(out_file),
            allpts_capture_payload_after,
            allpts_capture_arrays,
            "after_plan1",
        )
        if plan1 is None:
            # js = JointState()
            # js.name = planner.motion_gen.joint_names
            # js.position = p_grasp_js[ind]
            # planner.visualize_spheres_rviz(js)
            # input('Continue?')
            grasp_t1 = time.time()
            mplan_time += grasp_t1 - plan_t0
            continue

        ## get estimated target mesh ##

        # print(obj_ids)
        # print(obj_ids[ind])
        # print('ids', set(obj_ids))
        # print('masks', set(obj_mask))
        # input('Continue?')
        obj_id = 1 << obj_ids[ind]
        # tgt_pts = pts[(obj_mask & obj_id).astype(bool)]
        tgt_pts = perception.cur_points[(perception.cur_pmask & obj_id).astype(bool)]
        # tgt_rgb = rgb[(obj_mask & obj_id).astype(bool)]
        tgt_mesh = PerceptionInterface.get_shape_estimate(
            tgt_pts,
            # tgt_rgb,
        )

        ## plan approach ##
        joint_state2 = JointState()
        joint_state2.name = plan1.joint_names
        joint_state2.position = plan1.points[-1].positions
        # grasp_position = pose_to_matrix(grasps[ind])[:3, 3]
        # pre_grasp_pose = pose_to_matrix(pre_grasps[ind])[:3, 3]
        # dist = np.linalg.norm(grasp_position - pre_grasp_pose)
        # print('New Pre-grasp Distance:', dist)

        # planner.set_planning_scene(
        #     None,
        #     tgt_mesh,
        #     grasp_js[ind],
        #     attach_zoffset=0.01,
        # )
        # planner.visualize_spheres_rviz(grasp_js[ind])
        # input('Continue?')

        # planner.visualize_rviz()
        # plan2 = planner.joint_motion_plan(
        #     joint_state2,
        #     grasp_js[ind],
        #     path_constraint=[1, 1, 1, 1, 1, 0],
        #     constraint_in_goal_frame=True,
        # )
        # plan2 = planner.plan_to_selected_contact(
        #     joint_state2,
        #     grasp[ind],
        #     path_constraint=[0.9, 0.9, 0.9, 0.9, 0.9, 0],
        #     # path_constraint=[1, 1, 1, 1, 1, 0],
        #     constraint_in_goal_frame=True,
        # )

        _stage_probe("plan2 pink_cartesian_motion begin", f"ind={int(ind)}")
        plan2, success = planner.pink_cartesian_motion(
            joint_state2,
            grasp[ind],
            # offset=[0, 0, 0.005, 1, 0, 0, 0],
            return_all=True,
        )
        _stage_probe(
            "plan2 motion_plan done",
            f"ind={int(ind)} plan_is_none={plan2 is None} success={bool(success)}"
        )
        plan2_success = success
        # planner.visualize_traj_rviz(plan2)
        plan_t3 = time.time()
        print("Plan2 Time:", plan_t3 - plan_t2)
        # input('Continue?')
        if plan2 is None or not success:
            grasp_t1 = time.time()
            mplan_time += grasp_t1 - plan_t0
            continue

        ## plan lift ##
        planner.set_planning_scene(None)
        joint_state3 = JointState()
        joint_state3.name = plan2.joint_names
        joint_state3.position = plan2.points[-1].positions
        grasp_state = joint_state3
        # plan3 = planner.pose_motion_plan(
        #     joint_state3,
        #     grasp[ind],
        #     path_constraint=[0.9, 0.9, 0.9, 0.9, 0.9, 0],
        #     # path_constraint=[1, 1, 1, 1, 1, 0],
        #     constraint_in_goal_frame=False,
        #     offset=[0, 0, lift_height, 1, 0, 0, 0],
        #     visualize=False,
        # )
        if len(tgt_pts) == 0:
            tgt_min = np.min(all_pts[:, 2])
        else:
            tgt_min = np.min(tgt_pts[:, 2])
        height = 1.3 - grasp[ind].position.z
        # height = grasp[ind].position.z - tgt_min
        # height += np.max(all_pts[:, 2]) - np.min(all_pts[:, 2])
        # height += lift_height
        _stage_probe(
            "plan3 pink_cartesian_motion begin",
            f"ind={int(ind)} height={float(height)}",
        )
        plan3, success = planner.pink_cartesian_motion(
            joint_state3,
            grasp[ind],
            offset=[0, 0, height, 1, 0, 0, 0],
            constraint_in_goal_frame=False,
            return_all=True,
        )
        _stage_probe(
            "plan3 motion_plan done",
            f"ind={int(ind)} plan_is_none={plan3 is None} success={bool(success)}",
        )
        plan3_success = success
        # plan3 = copy.deepcopy(plan2)
        # plan3.points = list(reversed(plan3.points))
        # planner.visualize_traj_rviz(plan3)
        # input('paused...')
        plan_t4 = time.time()
        print("Plan3 Time:", plan_t4 - plan_t3)
        if plan3 is None:  # or not success:
            grasp_t1 = time.time()
            mplan_time += grasp_t1 - plan_t0
            continue

        grasp_t1 = time.time()
        mplan_time += grasp_t1 - plan_t0
        if place is not None:
            plan_t0 = time.time()
            ## plan to place ##
            joint_state4 = JointState()
            joint_state4.name = plan3.joint_names
            joint_state4.position = plan3.points[-1].positions

            # filter all_pts by robot and attach object
            planner.set_planning_scene(
                all_pts,
                tgt_mesh,
                grasp_state,
                attach_zoffset=0,
                filter_js=joint_state4,
                visualize=False,
            )
            # input('Pause')
            plan_t1 = time.time()
            print("Plan4 Scene Time:", plan_t1 - plan_t0)
            planner.visualize_rviz()
            planner.visualize_spheres_rviz(joint_state4)

            grasp_list = pose_to_list(grasp[ind])
            grasp_pos = grasp_list[:3]
            grasp_quat = grasp_list[3:7]

            loop = True
            while loop:
                if not callable(place):
                    place_target = place
                    loop = False
                else:
                    height = grasp_pos[2] - np.min(tgt_pts[:, 2])
                    planner.visualize_rviz()
                    planner.visualize_spheres_rviz(joint_state4)
                    place_target = place(height + 0.02)

                if type(place_target) == list:
                    if len(place_target) == 7:
                        # grasp_rot = pose_to_matrix(grasp[ind])[:3, :3]
                        # place_rot0 = tf.quaternion_matrix(place_target[3:7])[:3, :3]
                        # place_alt = np.multiply([1, 1, 1, 1, 1, -1, -1], place_target)
                        # place_rot1 = tf.quaternion_matrix(place_alt[3:7])[:3, :3]
                        # norm0 = np.linalg.norm(grasp_rot - place_rot0)
                        # norm1 = np.linalg.norm(grasp_rot - place_rot1)
                        # print('Dist to', place_target, ' :', norm0)
                        # print('Dist to', place_alt, ' :', norm1)
                        # min_ind = np.argmin([norm0, norm1])
                        # place_target = np.array([place_target, place_alt])[min_ind]
                        rotated_placements = [place_target]
                    elif len(place_target) == 3:
                        rotated_placements = [
                            place_target
                            + tf.quaternion_multiply(
                                tf.quaternion_about_axis(a, [0, 0, 1]),
                                grasp_quat,
                            ).tolist()
                            for a in np.linspace(
                                0,
                                2 * np.pi,
                                num_place_rotations,
                                endpoint=False,
                            )
                        ]
                    else:
                        print("Invalid place target:", place_target)
                        break

                    grasp_planner.plotter.draw_grasps()
                    grasp_planner.plotter.draw_grasps(
                        [list_to_pose(p) for p in rotated_placements],
                        [1] * len(rotated_placements),
                    )
                    # input('Continue?')

                    plan4, res = planner.pose_motion_plan(
                        joint_state4,
                        rotated_placements,
                        return_all=True,
                    )
                elif type(place_target) is dict:
                    plan4, res = planner.joint_motion_plan(
                        joint_state4,
                        place_target,
                        return_all=True,
                    )

                loop &= plan4 is None
                if (
                    res is not None
                    and res.status == MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS
                ):
                    break

            plan_t2 = time.time()
            print("Plan4 Time:", plan_t2 - plan_t1)
            if plan4 is None:
                with open("/tmp/grasp_fail.txt", "w") as f:
                    print(grasp[ind], file=f)
                with open("/tmp/pre_grasp_fail.txt", "w") as f:
                    print(p_grasp[ind], file=f)
                status = getattr(res, "status", "FRANKA_PLACE_IK_FAIL")
                print("Error", status, sep=",", file=out_file)
                plan_t3 = time.time()
                mplan_time += plan_t3 - plan_t0
                continue

            plan_t4 = time.time()
            mplan_time += plan_t4 - plan_t0
            plans = [plan1, plan2, plan3, plan4]
        else:
            plans = [plan1, plan2, plan3]
        debug_state = getattr(grasp_planner, "_torc_selected_grasp_debug", {})
        selected_source_index = (
            int(debug_sorted_source_indices[ind])
            if ind < len(debug_sorted_source_indices)
            else -1
        )
        raw_mats = np.asarray(
            debug_state.get("raw_cgn_grasp_matrices", np.empty((0, 4, 4))),
            dtype=np.float64,
        )
        selected_raw_matrix = (
            raw_mats[selected_source_index]
            if 0 <= selected_source_index < len(raw_mats)
            else np.full((4, 4), np.nan)
        )
        selected_debug_payload = {
            "capture_version": "f2_6_selected_grasp_capture_v1",
            "behavior_preserving": True,
            "scene_name": os.environ.get("TORC_SCENE_NAME"),
            "scene_path": os.environ.get("TORC_SCENE_PATH"),
            "target_object": os.environ.get("TORC_TARGET_OBJECT"),
            "method": os.environ.get("TORC_METHOD"),
            "pick_count": pick_count or _get_pick_count_from_out_file(out_file),
            "selected_object_segment_id": int(obj_ids[ind]),
            "dg_chosen_segment_id": int(chosen_obj_id),
            "selected_index_in_sorted_validated_candidates": int(ind),
            "selected_raw_cgn_index": selected_source_index,
            "selected_score": float(score[ind]),
            "num_raw_cgn_grasps": int(debug_state.get("raw_cgn_count", -1)),
            "num_ik_candidates": int(debug_state.get("ik_candidate_count", -1)),
            "num_validated_candidates_before_score_sort": int(
                debug_state.get("validated_candidate_count", -1)
            ),
            "num_validated_candidates_after_score_filter": int(len(grasp)),
            "num_candidates_for_chosen_segment": int(len(chosen_obj_grasp_indices)),
            "frame_labels": debug_state.get(
                "frame_labels",
                {
                    "selected_raw_cgn_matrix": "T_world_cgn_raw",
                    "selected_transformed_grasp_matrix": "T_world_motoman_ee_goal_after_plus_hand_depth_local_z",
                    "selected_pregrasp_matrix": "T_world_motoman_ee_pregrasp",
                },
            ),
            "hand_depth_m": debug_state.get("hand_depth_m"),
            "grasp_planner": debug_state.get("grasp_planner"),
                "planning": {
                    "plan1_pregrasp_success": plan1 is not None,
                    "plan2_grasp_approach_success": plan2 is not None
                    and plan2_success,
                    "plan3_lift_success": plan3 is not None and plan3_success,
                "place_requested": place is not None,
                "plan_count": len(plans),
            },
            "execution_result": None,
            "missing_fields": {
                "selected_postgrasp_lift_pose_matrix": (
                    "not directly represented as a single pose before execution; lift is encoded as plan3 trajectory"
                ),
                "raw_cgn_index": (
                    None
                    if selected_source_index >= 0
                    else "source index unavailable because debug source-index array length mismatch"
                ),
            },
            "scientific_semantics": {
                "real_cgn_predictions": True,
                "original_torc_selection_path": True,
                "gt_grasp_used": False,
                "fake_grasp_used": False,
                "collision_checking_disabled": False,
                "planning_skipped": False,
                "franka_used": os.environ.get("TORC_ROBOT", "").strip().lower()
                in ("franka", "panda"),
            },
        }
        selected_debug_arrays = {
            "raw_cgn_grasp_matrices": np.asarray(
                debug_state.get("raw_cgn_grasp_matrices", np.empty((0, 4, 4))),
                dtype=np.float64,
            ),
            "raw_cgn_scores": np.asarray(
                debug_state.get("raw_cgn_scores", np.empty((0,))),
                dtype=np.float64,
            ),
            "raw_cgn_object_ids": np.asarray(
                debug_state.get("raw_cgn_object_ids", np.empty((0,))),
                dtype=np.int64,
            ),
            "raw_cgn_source_indices": np.asarray(
                debug_state.get("raw_cgn_source_indices", np.empty((0,))),
                dtype=np.int64,
            ),
            "ik_candidate_matrices": np.asarray(
                debug_state.get("ik_candidate_matrices", np.empty((0, 4, 4))),
                dtype=np.float64,
            ),
            "ik_candidate_scores": np.asarray(
                debug_state.get("ik_candidate_scores", np.empty((0,))),
                dtype=np.float64,
            ),
            "ik_candidate_object_ids": np.asarray(
                debug_state.get("ik_candidate_object_ids", np.empty((0,))),
                dtype=np.int64,
            ),
            "ik_candidate_source_indices": np.asarray(
                debug_state.get("ik_candidate_source_indices", np.empty((0,))),
                dtype=np.int64,
            ),
            "validated_candidate_matrices_sorted": np.asarray(
                [_matrix_from_pose_or_none(g) for g in grasp],
                dtype=np.float64,
            ),
            "validated_pregrasp_matrices_sorted": np.asarray(
                [_matrix_from_pose_or_none(g) for g in p_grasp],
                dtype=np.float64,
            ),
            "validated_scores_sorted": np.asarray(score, dtype=np.float64),
            "validated_object_ids_sorted": np.asarray(obj_ids, dtype=np.int64),
            "validated_source_indices_sorted": np.asarray(
                debug_sorted_source_indices,
                dtype=np.int64,
            ),
            "selected_raw_cgn_matrix": np.asarray(
                selected_raw_matrix,
                dtype=np.float64,
            ),
            "selected_transformed_grasp_matrix": np.asarray(
                _matrix_from_pose_or_none(grasp[ind]),
                dtype=np.float64,
            ),
            "selected_pregrasp_matrix": np.asarray(
                _matrix_from_pose_or_none(p_grasp[ind]),
                dtype=np.float64,
            ),
            "selected_grasp_joint_values": np.asarray(
                grasp_js[ind],
                dtype=np.float64,
            ),
            "selected_pregrasp_joint_values": np.asarray(
                p_grasp_js[ind],
                dtype=np.float64,
            ),
        }
        selected_debug_paths = _write_selected_grasp_debug(
            out_file,
            selected_debug_payload["pick_count"],
            selected_debug_payload,
            selected_debug_arrays,
        )
        break

    if not plans:
        return ("no_valid_motion_plan_found",)

    comp_t1 = time.time()
    print("Chosen Grasp Score:", score[ind])
    print("Total Grasp Planning Time:", grasp_time)
    print("Total Motion Planning Time:", mplan_time)
    print("Total Computation Time:", (comp_t1 - comp_t0) - (viz_t1 - viz_t0))
    # print('Time till execution start', t1 - t0, sep=',', file=out_file)

    print("Grasp Score", score[ind], sep=",", file=out_file)
    print(
        "Total Computation Duration",
        (comp_t1 - comp_t0) - (viz_t1 - viz_t0),
        sep=",",
        file=out_file,
    )
    print("Total Grasp Planning Duration", grasp_time, sep=",", file=out_file)
    print("Total Motion Planning Duration", mplan_time, sep=",", file=out_file)
    if sys.stderr.name != out_file.name:
        np.savetxt(out_file.name.replace(".csv", "_target_points.txt"), tgt_pts)

    planner.set_planning_scene(
        all_pts,
        visualize=False,
        save_scene_file="/tmp/world_config.pth",
    )
    planner.visualize_rviz()
    # planner.visualize_traj_rviz(plan1)
    if not robot.is_sim:
        # print("Press Enter to Execute.")
        obj_name = input(
            f"\n\nChose object_id: {obj_ids[ind]}. What is the English name of this object?\n"
        )
        # obj_name = object_to_grasp
    else:
        # obj_name = 35 if type(object_to_grasp) is str else object_to_grasp
        obj_name = perception.seg_label_to_obj_id[obj_ids[ind]]
    if selected_debug_payload is not None:
        selected_debug_payload["intended_object_id"] = int(obj_name)
        if robot.is_sim:
            selected_debug_payload["intended_object_name"] = object_ids_to_names_dict.get(
                int(obj_name),
                str(obj_name),
            )
        _write_selected_grasp_debug(
            out_file,
            selected_debug_payload["pick_count"],
            selected_debug_payload,
            selected_debug_arrays,
        )

    if robot.is_sim:
        VEL0 = rospy.get_param("/robot/vel_ang_lim")
        ACC0 = rospy.get_param("/robot/acc_ang_lim")

    exec_t0 = time.time()
    for i, plan in enumerate(plans):
        _stage_probe("execute plan segment begin", f"segment={i}")
        if i == 0:
            ee_open()
        elif i == 2:
            ee_close()

        if type(plan) is JointTrajectory:
            # print('Plan success!')
            planner.visualize_traj_rviz(plan)
            # input('run?')
            # move as fast as possible
            if robot.is_sim:
                # if i == 0:
                if i == -1:
                    rospy.set_param("/robot/vel_ang_lim", 600)
                    rospy.set_param("/robot/acc_ang_lim", 8500)
                else:
                    rospy.set_param("/robot/vel_ang_lim", VEL0)
                    rospy.set_param("/robot/acc_ang_lim", ACC0)
            plan.points[-1].time_from_start = plan.points[0].time_from_start
            execute(plan, window=0, wait=True, retime=True)
            _stage_probe("execute plan segment done", f"segment={i}")
            # execute(plan, window=0)
        else:
            # print('Plan', i, 'error:', plan)
            _stage_probe("execute plan segment skipped", f"segment={i} type={type(plan)}")
            break

        if i == 0:
            grasp_t1 = time.time()
            # print(
            #     'Time execution start to pre-grasp',
            #     t2 - t0,
            #     sep=',',
            #     file=out_file
            # )
        elif i == 2:
            s, g, d = get_experiment_result(obj_name, sim=robot.is_sim)
            print(
                "Object Name",
                object_ids_to_names_dict[int(obj_name)] if robot.is_sim else obj_name,
                sep=",",
                file=out_file,
                flush=True,
            )
            print("Grasp Success", s, sep=",", file=out_file, flush=True)
            if len(g) == 0:
                g = [""]
            gs = '"' + ",".join(g) + '"'
            print("Grasped", gs, sep=",", file=out_file, flush=True)

    exec_t1 = time.time()
    print("Total Execution Time:", exec_t1 - exec_t0)
    print("Total Execution Duration", exec_t1 - exec_t0, sep=",", file=out_file)
    # print('Time pre-grasp to retract', t1 - t2, sep=',', file=out_file)

    s, g, d = get_experiment_result(obj_name, sim=robot.is_sim)
    print("Retract Success", s, sep=",", file=out_file, flush=True)
    if len(d) == 0:
        d = [""]
    ds = '"' + ",".join(d) + '"'
    print("Dropped", ds, sep=",", file=out_file, flush=True)
    if selected_debug_payload is not None:
        selected_debug_payload["execution_result"] = {
            "retract_success": bool(s),
            "grasped": list(g),
            "dropped": list(d),
        }
        _write_selected_grasp_debug(
            out_file,
            selected_debug_payload["pick_count"],
            selected_debug_payload,
            selected_debug_arrays,
        )

    return "finished", s, str(obj_name), d


if __name__ == "__main__":
    is_sim = (
        sys.argv[1][0]
        not in (
            "0",
            "r",
            "R",
            "n",
            "N",
        )
        if len(sys.argv) > 1
        else True
    )
    gt = (
        sys.argv[2][0]
        not in (
            "0",
            "r",
            "R",
            "n",
            "N",
        )
        if len(sys.argv) > 2
        else True
    )
    ## select object from argument ##
    object_to_grasp = sys.argv[3] if len(sys.argv) > 3 else "35"
    place = (
        None
        if len(sys.argv) > 4
        and sys.argv[4][0]
        in (
            "n",
            "N",
            "f",
            "F",
            "0",
        )
        else [0.48, 0, 1.25, 0.5, -0.5, 0.5, 0.5]
    )
    # place = [0.6, 0, 1.1] if len(sys.argv) > 4 else None
    out_file = open(sys.argv[5], "a") if len(sys.argv) > 5 else sys.stderr

    print("is_sim: ", is_sim)
    print("gt: ", gt)
    ## init perception and planning interfaces ##
    rospy.init_node("planning")
    t0 = time.time()
    from task_planner.motoman import MotomanSDA10F

    robot = MotomanSDA10F(is_sim, gt)
    perception = robot.init_perception_interface()
    planner = robot.init_motion_planner(planner="curobo")  # ,warmup=False)
    grasp_planner = GraspPlanner(
        robot.curobo_config,
        planner.static_world_config,
        robot.urdf,
        ignore_collision_ee_links=robot.ignore_collision_ee_links,
    )
    t1 = time.time()
    print("Init Time:", t1 - t0)

    result = open_loop_pick_or_place(
        object_to_grasp,
        robot,
        perception,
        planner,
        grasp_planner,
        place=place,
        visualize=True,
        out_file=out_file,
    )

    if result == "timeout":
        print("Timeout", True, sep=",", file=out_file)
    else:
        print("Timeout", False, sep=",", file=out_file)
