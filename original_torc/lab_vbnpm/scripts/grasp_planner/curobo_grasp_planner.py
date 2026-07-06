import time
import copy
import json
import os
import torch
import numpy as np
import open3d as o3d
import trimesh as tm
import transformations as tf
from scipy.spatial import KDTree
from tracikpy import TracIKSolver, MultiTracIKSolver

# from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.base import TensorDeviceType
from curobo.geom.types import WorldConfig, Mesh
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

import rospy
import rosservice

from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pcl2
from geometry_msgs.msg import Point, Pose
from std_msgs.msg import ColorRGBA, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import JointState as JointState_MSG

from utils.conversions import pose_to_matrix, matrix_to_pose
from perception.perception_fast import PerceptionInterface
from grasp_planner.grasp_plotter import GraspPlotter

try:
    from grasp_representation.lowlevel_grasp import lowlevel_grasps_from_arrays, lowlevel_to_canonical_grasp
    from robot_interface.franka_adapter import RobotAdapter
except Exception:
    lowlevel_grasps_from_arrays = None
    lowlevel_to_canonical_grasp = None
    RobotAdapter = None

# GRASP_PLANNER = 'cgn'
# GRASP_PLANNER = 'cgn_pytorch'
# GRASP_PLANNER = 'gpd'
# GRASP_PLANNER = 'gpg'
# GRASP_PLANNER = 'graspnet'
GRASP_PLANNER = "ground_truth"
if os.environ.get("TORC_USE_CGN_ZMQ") == "1":
    GRASP_PLANNER = os.environ.get("TORC_GRASP_PLANNER", "cgn")
elif os.environ.get("TORC_DUMP_CGN_CALLSITE") == "1" or os.environ.get("TORC_CGN_CAPTURE_ONLY") == "1":
    GRASP_PLANNER = os.environ.get("TORC_GRASP_PLANNER", "cgn")
else:
    GRASP_PLANNER = os.environ.get("TORC_GRASP_PLANNER", GRASP_PLANNER)


class CgnCallsiteCaptureOnly(RuntimeError):
    pass


def _env_flag(name):
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def _stage_probe(stage, detail=""):
    path = os.environ.get("TORC_STAGE_LOG_FILE")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[GRASP_PLANNER_STAGE {time.time():.6f}] {stage}")
            if detail:
                f.write(f" | {detail}")
            f.write("\n")
    except Exception:
        pass


def _safe_name(value, default="unknown"):
    value = str(value or default)
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _dump_cgn_callsite(
    visible_points,
    mask,
    target_points=None,
    collision_points=None,
    vol_bnds=None,
    z_min=None,
    srv_name=None,
):
    experiment_dir = os.environ.get("TORC_EXPERIMENT_DIR")
    if experiment_dir:
        output_dir = os.path.join(experiment_dir, "cgn_callsite_captures")
    else:
        output_dir = os.environ.get(
            "TORC_CGN_CALLSITE_CAPTURE_DIR",
            "/mnt/ssd/ziyaochen/torc_franka_lift3d_results_20260622_093234/cgn_callsite_captures",
        )
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    scene = _safe_name(os.environ.get("TORC_SCENE_NAME") or os.environ.get("TORC_SCENE_PATH"))
    target = _safe_name(os.environ.get("TORC_TARGET_OBJECT"))
    method = _safe_name(os.environ.get("TORC_METHOD"))
    path = os.path.join(
        output_dir,
        f"cgn_callsite_scene_{scene}_target_{target}_method_{method}_{timestamp}.npz",
    )

    points_np = np.asarray(visible_points, dtype=np.float32).reshape(-1, 3)
    mask_np = np.asarray(mask, dtype=np.uint32).reshape(-1)
    if len(points_np) != len(mask_np):
        raise RuntimeError(
            f"CGN callsite capture points/mask length mismatch: {len(points_np)} vs {len(mask_np)}"
        )

    metadata = {
        "scene_path": os.environ.get("TORC_SCENE_PATH"),
        "scene_name": os.environ.get("TORC_SCENE_NAME"),
        "target_object": os.environ.get("TORC_TARGET_OBJECT"),
        "method": os.environ.get("TORC_METHOD"),
        "experiment_dir": experiment_dir,
        "grasp_planner": GRASP_PLANNER,
        "srv_name": srv_name,
        "z_min": None if z_min is None else float(z_min),
        "num_points": int(len(points_np)),
        "num_mask": int(len(mask_np)),
        "num_nonzero_mask": int(np.count_nonzero(mask_np)),
        "unique_mask_values": [int(v) for v in np.unique(mask_np)[:256]],
        "capture_only": _env_flag("TORC_CGN_CAPTURE_ONLY"),
    }
    arrays = {
        "points": points_np,
        "mask": mask_np,
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }
    if target_points is not None:
        arrays["target_points"] = np.asarray(target_points, dtype=np.float32).reshape(-1, 3)
    if collision_points is not None:
        arrays["collision_points"] = np.asarray(collision_points, dtype=np.float32).reshape(-1, 3)
    if vol_bnds is not None:
        arrays["vol_bnds"] = np.asarray(vol_bnds, dtype=np.float32)

    np.savez_compressed(path, **arrays)
    print(f"[TORC_CGN_CALLSITE_CAPTURE] wrote {path}", flush=True)
    print(
        "[TORC_CGN_CALLSITE_CAPTURE] "
        f"points={points_np.shape} mask={mask_np.shape} "
        f"nonzero_mask={np.count_nonzero(mask_np)} unique_mask_values={len(np.unique(mask_np))}",
        flush=True,
    )
    return path


def _pose_from_zmq_list(values):
    if len(values) != 7:
        raise RuntimeError(f"CGN ZMQ pose must have 7 values, got {len(values)}")
    pose = Pose()
    pose.position.x = float(values[0])
    pose.position.y = float(values[1])
    pose.position.z = float(values[2])
    pose.orientation.x = float(values[3])
    pose.orientation.y = float(values[4])
    pose.orientation.z = float(values[5])
    pose.orientation.w = float(values[6])
    return pose


def _point_from_zmq_list(values):
    if len(values) != 3:
        raise RuntimeError(f"CGN ZMQ sample point must have 3 values, got {len(values)}")
    return Point(float(values[0]), float(values[1]), float(values[2]))


def _torc_object_ids_from_mask_labels(labels):
    object_ids = []
    for label in np.asarray(labels, dtype=np.uint32).reshape(-1):
        label = int(label)
        if label <= 0:
            continue
        for bit in range(label.bit_length()):
            if label & (1 << bit):
                object_ids.append(bit)
    return object_ids


def _assign_lowlevel_object_ids_from_mask(contact_points, scene_points, scene_mask, k=32):
    contacts = np.asarray(contact_points, dtype=np.float64).reshape(-1, 3)
    points = np.asarray(scene_points, dtype=np.float64).reshape(-1, 3)
    mask = np.asarray(scene_mask, dtype=np.uint32).reshape(-1)
    if len(contacts) == 0:
        return np.zeros(0, dtype=np.int32)
    if len(points) == 0 or len(mask) != len(points):
        return np.zeros(len(contacts), dtype=np.int32)

    tree = KDTree(points)
    query_k = min(max(1, int(k)), len(points))
    _dist, inds = tree.query(contacts, k=query_k)
    inds = np.asarray(inds)
    if inds.ndim == 1:
        inds = inds[:, None]

    assigned = []
    for row in inds:
        ordered_ids = _torc_object_ids_from_mask_labels(mask[row])
        if not ordered_ids:
            assigned.append(0)
            continue
        vals, counts = np.unique(np.asarray(ordered_ids, dtype=np.int32), return_counts=True)
        max_count = int(np.max(counts))
        tied = set(int(v) for v in vals[counts == max_count])
        chosen = next(obj_id for obj_id in ordered_ids if obj_id in tied)
        assigned.append(int(chosen))
    return np.asarray(assigned, dtype=np.int32)


def _call_cgn_zmq(visible_points, mask):
    import zmq

    address = os.environ.get("TORC_CGN_ZMQ_ADDRESS", "tcp://127.0.0.1:6007")
    timeout_ms = int(os.environ.get("TORC_CGN_ZMQ_TIMEOUT_MS", "180000"))
    points_np = np.asarray(visible_points, dtype=np.float32).reshape(-1, 3)
    mask_np = np.asarray(mask, dtype=np.uint32).reshape(-1)
    if len(points_np) != len(mask_np):
        raise RuntimeError(
            f"CGN ZMQ points/mask length mismatch: {len(points_np)} vs {len(mask_np)}"
        )
    points_np, mask_np = _limit_cgn_zmq_input(points_np, mask_np)

    print(
        "[TORC_CGN_ZMQ] sending infer request "
        f"address={address} timeout_ms={timeout_ms} "
        f"points={points_np.shape} mask={mask_np.shape} "
        f"nonzero_mask={np.count_nonzero(mask_np)}",
        flush=True,
    )
    t0 = time.time()
    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(address)
        cmd = "infer_lowlevel" if os.environ.get("TORC_ROBOT", "motoman").strip().lower() in ("franka", "panda") else "infer"
        socket.send_pyobj(
            {
                "cmd": cmd,
                "points": points_np,
                "mask": mask_np,
                "source": "torc_ros_callsite",
            }
        )
        response = socket.recv_pyobj()
    except zmq.error.Again as exc:
        raise RuntimeError(
            f"Timed out waiting for CGN ZMQ server at {address} after {timeout_ms} ms"
        ) from exc
    finally:
        socket.close()

    elapsed = time.time() - t0
    if not isinstance(response, dict):
        raise RuntimeError(f"CGN ZMQ response must be a dict, got {type(response)}")
    if response.get("type") != "success":
        message = response.get("message", "unknown error")
        traceback_text = response.get("traceback")
        if traceback_text:
            message += "\n" + traceback_text[-2000:]
        raise RuntimeError(f"CGN ZMQ infer failed: {message}")

    if response.get("cmd") == "infer_lowlevel":
        if any(x is None for x in (lowlevel_grasps_from_arrays, lowlevel_to_canonical_grasp, RobotAdapter)):
            raise RuntimeError("Franka lowlevel grasp adapter modules are unavailable")
        arrays = {
            "contact_point_world": np.asarray(response.get("contact_point_world", []), dtype=np.float64),
            "approach_dir_world": np.asarray(response.get("approach_dir_world", []), dtype=np.float64),
            "base_dir_world": np.asarray(response.get("base_dir_world", []), dtype=np.float64),
            "offset_bin_value": np.asarray(response.get("offset_bin_value", response.get("gripper_opening", [])), dtype=np.float64),
            "offset_bin_index": np.asarray(response.get("offset_bin_index", []), dtype=np.int32),
            "gripper_opening": np.asarray(response.get("gripper_opening", []), dtype=np.float64),
            "score": np.asarray(response.get("score", []), dtype=np.float64),
            "object_id": np.asarray(response.get("object_id", []), dtype=np.int32),
            "source_index": np.asarray(response.get("source_index", []), dtype=np.int32),
            "selection_index": np.asarray(response.get("selection_index", []), dtype=np.int32),
        }
        if len(arrays["offset_bin_index"]) == 0:
            arrays["offset_bin_index"] = np.full(len(arrays["contact_point_world"]), -1, dtype=np.int32)
        arrays["object_id"] = _assign_lowlevel_object_ids_from_mask(
            arrays["contact_point_world"],
            points_np,
            mask_np,
        )
        response_object_ids = np.asarray(response.get("object_id", []), dtype=np.int32)
        _stage_probe(
            "cgn lowlevel object id remap",
            "response_unique="
            f"{np.unique(response_object_ids).astype(int).tolist() if len(response_object_ids) else []} "
            f"mask_unique={np.unique(arrays['object_id']).astype(int).tolist()} "
            f"count={len(arrays['object_id'])}",
        )
        print(
            "[TORC_CGN_ZMQ] remapped lowlevel object ids from TORC mask "
            f"unique={np.unique(arrays['object_id']).tolist()}",
            flush=True,
        )
        adapter = RobotAdapter()
        poses_raw = []
        scores_raw = []
        samples_raw = []
        object_ids_raw = []
        for lowlevel in lowlevel_grasps_from_arrays(arrays):
            canonical = lowlevel_to_canonical_grasp(lowlevel)
            command = adapter.adapt(canonical)
            poses_raw.append(matrix_to_pose(command.tcp_contact_pose_world))
            scores_raw.append(float(lowlevel.score))
            samples_raw.append([float(v) for v in np.asarray(lowlevel.contact_point_world, dtype=np.float64).reshape(3)])
            object_ids_raw.append(int(lowlevel.object_id))
    else:
        poses_raw = response.get("poses", [])
        scores_raw = response.get("scores", [])
        samples_raw = response.get("samples", [])
        object_ids_raw = response.get("object_ids", [])
    lengths = [len(poses_raw), len(scores_raw), len(samples_raw), len(object_ids_raw)]
    if len(set(lengths)) != 1:
        raise RuntimeError(
            "CGN ZMQ response length mismatch: "
            f"poses/scores/samples/object_ids={lengths}"
        )

    poses = [p if hasattr(p, "position") else _pose_from_zmq_list(p) for p in poses_raw]
    scores = [float(s) for s in scores_raw]
    samples = [_point_from_zmq_list(s) for s in samples_raw]
    object_ids = [int(i) for i in object_ids_raw]
    _stage_probe(
        "cgn zmq response converted",
        f"num_grasps={len(poses)} object_unique={sorted(set(object_ids)) if object_ids else []}",
    )
    print(
        "[TORC_CGN_ZMQ] received infer response "
        f"num_grasps={len(poses)} client_elapsed_sec={elapsed:.3f} "
        f"server_elapsed_sec={response.get('elapsed_sec')}",
        flush=True,
    )
    return poses, scores, samples, object_ids


def _limit_cgn_zmq_input(points_np, mask_np):
    max_points = int(os.environ.get("TORC_CGN_ZMQ_MAX_POINTS", "12000"))
    if max_points <= 0 or len(points_np) <= max_points:
        return points_np, mask_np

    rng = np.random.default_rng(int(os.environ.get("TORC_CGN_ZMQ_SAMPLE_SEED", "0")))
    labels = [int(v) for v in np.unique(mask_np) if int(v) != 0]
    object_budget = min(
        max_points,
        int(os.environ.get("TORC_CGN_ZMQ_OBJECT_POINT_BUDGET", str(int(max_points * 0.75)))),
    )
    bg_budget = max_points - object_budget

    chosen = []
    if labels:
        per_label = max(1, object_budget // len(labels))
        for label in labels:
            idx = np.flatnonzero(mask_np == label)
            if len(idx) > per_label:
                idx = rng.choice(idx, size=per_label, replace=False)
            chosen.append(idx)

    chosen_count = int(sum(len(idx) for idx in chosen))
    remaining = max_points - chosen_count
    if remaining > 0:
        bg_idx = np.flatnonzero(mask_np == 0)
        if len(bg_idx) > remaining:
            bg_idx = rng.choice(bg_idx, size=remaining, replace=False)
        chosen.append(bg_idx)

    if not chosen:
        idx = rng.choice(len(points_np), size=max_points, replace=False)
    else:
        idx = np.concatenate(chosen)
        if len(idx) > max_points:
            idx = rng.choice(idx, size=max_points, replace=False)
        rng.shuffle(idx)

    limited_points = points_np[idx]
    limited_mask = mask_np[idx]
    print(
        "[TORC_CGN_ZMQ] downsampled request "
        f"points={points_np.shape}->{limited_points.shape} "
        f"nonzero_mask={np.count_nonzero(mask_np)}->{np.count_nonzero(limited_mask)} "
        f"labels={labels}",
        flush=True,
    )
    return limited_points, limited_mask


def _pose_matrix_or_none(value):
    if value is None:
        return None
    if hasattr(value, "position"):
        return pose_to_matrix(value)
    return np.asarray(value, dtype=np.float64)


def _point_xyz_or_none(value):
    if value is None:
        return None
    if hasattr(value, "x"):
        return [float(value.x), float(value.y), float(value.z)]
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(arr) >= 3:
        return arr[:3].tolist()
    return None


if GRASP_PLANNER[0:3] == "cgn":
    from cgn_ros.srv import GetGrasps
elif GRASP_PLANNER == "gpd":
    from gpd_docker.srv import GetGrasps
elif GRASP_PLANNER == "gpg":
    from gpg_ros.srv import GetGrasps
elif GRASP_PLANNER == "graspnet":
    from graspnet_ros.srv import GetGrasps
elif GRASP_PLANNER == "ground_truth":
    from rospkg import RosPack
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
    from utils.visual_utils import decode_seg_img_rgb
    from lab_vbnpm.msg import ObjectPoses, ObjectIdsToNames


class GraspPlanner:

    def __init__(
        self,
        curobo_config,
        world_config,
        urdf_file,
        resolution=0.005,
        ignore_collision_ee_links=None,
    ):
        self.grasp_srvs = sorted(
            filter(lambda x: x.find("get_grasps") >= 0, rosservice.get_service_list())
        )

        # distance to motoman_right_ee not to tip as in cfg
        # TODO verify this is correct
        if GRASP_PLANNER == "gpd":
            self.hand_depth = 0.045  # 55 / 2
        elif GRASP_PLANNER == "gpg":
            self.hand_depth = 0.035  # 55 / 2
        elif GRASP_PLANNER[0:3] == "cgn":
            if os.environ.get("TORC_ROBOT", "motoman").strip().lower() in ("franka", "panda") and _env_flag("TORC_USE_CGN_ZMQ"):
                self.hand_depth = 0.0
            else:
                self.hand_depth = float(os.environ.get("TORC_LEGACY_GRASP_DEPTH_M", "0.1034"))
        elif GRASP_PLANNER == "ground_truth":
            self.hand_depth = 0.0
            self.bridge = CvBridge()
            saved_grasps = (
                RosPack().get_path("lab_vbnpm")
                + "/scripts/grasp_planner/gc6d_grasp_poses.npy"
            )
            self.name2grasp = np.load(saved_grasps, allow_pickle=True).item()

        self.resolution = resolution
        self.curobo_config = curobo_config
        self.world_config = world_config
        self.tensor_args = TensorDeviceType()
        self.disable_ee_links = list(ignore_collision_ee_links or [])

        robot_config = RobotWorldConfig.load_from_config(
            self.curobo_config,
            self.world_config,
            collision_activation_distance=0.0,
        )
        base_link = robot_config.kinematics.base_link
        ee_link = robot_config.kinematics.ee_link
        self.ee_link = ee_link
        if ee_link not in self.disable_ee_links:
            self.disable_ee_links.append(ee_link)
        self.ik_base_pose_world = np.eye(4, dtype=np.float64)
        if os.environ.get("TORC_ROBOT", "motoman").strip().lower() in ("franka", "panda"):
            raw_base = os.environ.get("TORC_FRANKA_MUJOCO_BASE_POSE_WORLD", "0,0,0.86")
            values = [float(v) for v in raw_base.replace(";", ",").split(",") if v.strip()]
            if len(values) >= 3:
                self.ik_base_pose_world[:3, 3] = values[:3]
        self.ik_world_to_base = np.linalg.inv(self.ik_base_pose_world)

        # init global structures
        self.robot_world = RobotWorld(robot_config)
        self.kinematics_config = self.robot_world.kinematics.kinematics_config
        # input(self.robot_world.kinematics.robot_spheres[-2-max_s:])
        self.ik_solver = TracIKSolver(urdf_file, base_link, ee_link)
        # self.ik_solver = MultiTracIKSolver(
        #     urdf_file, base_link, ee_link, timeout=0.01, num_workers=16
        # )
        self.extra_sphere_link = self.ee_link

        # init plotter
        self.plotter = GraspPlotter()
        self._torc_selected_grasp_debug = {}
        self.is_franka_robot = (
            os.environ.get("TORC_ROBOT", "motoman").strip().lower()
            in ("franka", "panda")
        )

        pose_x, pose_y, pose_z = rospy.get_param("/workspace/pose")
        size_x, size_y, size_z = rospy.get_param("/workspace/size")
        offsetxF = 0.1
        offsetxB = 0.01
        offsetyLR = 0.3
        offsetz = 0.05
        pose_x -= offsetxF
        pose_y -= size_y - offsetyLR / 2
        pose_z += offsetz
        size_x += offsetxF - offsetxB
        size_y -= offsetyLR
        size_z -= 0.5 * offsetz
        pos_a = [pose_x, pose_y, pose_z]
        pos_b = [pose_x + size_x, pose_y + size_y, pose_z + size_z]
        self.vol_bnds = np.zeros((3, 2))
        self.vol_bnds[:, 0] = np.minimum(pos_a, pos_b)
        self.vol_bnds[:, 1] = np.maximum(pos_a, pos_b)

    def _pose_for_tracik(self, pose_world):
        return self.ik_world_to_base @ pose_world

    def _points_for_collision_world(self, points):
        if points is None or not self.is_franka_robot:
            return points
        pts = np.asarray(points, dtype=np.float64)
        original_shape = pts.shape
        if pts.size == 0:
            return pts.reshape(original_shape)
        pts = pts.reshape(-1, 3)
        pts_h = np.concatenate(
            [pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)],
            axis=1,
        )
        pts = (pts_h @ self.ik_world_to_base.T)[:, :3]
        return pts.reshape(original_shape)

    def _franka_single_object_collides_with(self, poses_world, visible_points, mask):
        """Franka equivalent of TORC's Robotiq proxy-sphere singularity gate.

        The TORC rule is kept unchanged: a candidate is singular when its
        gripper contact proxy intersects exactly one segmented object.  Only
        the robot-specific proxy geometry changes here.  For Franka, the proxy
        is a row of small spheres along the Panda closing axis at the adapted
        contact depth, mirroring TORC's attach_extra_spheres check without
        depending on Robotiq geometry.
        """
        points = np.asarray(visible_points, dtype=np.float64).reshape(-1, 3)
        labels = np.asarray(mask, dtype=np.uint32).reshape(-1)
        nonzero = labels != 0
        points = points[nonzero]
        labels = labels[nonzero]
        if len(points) == 0:
            return [set() for _ in range(len(poses_world))]

        per_object_budget = int(os.environ.get("TORC_FRANKA_SINGULARITY_POINTS_PER_OBJECT", "2500"))
        obj_points = []
        obj_indices = []
        num_objs = len(np.binary_repr(int(np.max(labels))))
        for obj_idx in range(num_objs):
            obj_mask = (labels & (1 << obj_idx)).astype(bool)
            pts = points[obj_mask]
            if len(pts) == 0:
                continue
            if len(pts) > per_object_budget:
                step = max(1, len(pts) // per_object_budget)
                pts = pts[::step][:per_object_budget]
            obj_points.append(pts)
            obj_indices.append(obj_idx)

        collides_with = []
        half_span = float(os.environ.get("TORC_FRANKA_PROXY_HALF_SPAN_M", "0.040"))
        radius = float(os.environ.get("TORC_FRANKA_PROXY_RADIUS_M", "0.008"))
        default_contact_depth = "0.0044"
        if RobotAdapter is not None:
            try:
                default_contact_depth = str(RobotAdapter.derive_tcp_pad_front_m())
            except Exception:
                default_contact_depth = "0.0044"
        contact_depth = float(os.environ.get("TORC_FRANKA_PROXY_CONTACT_DEPTH_M", default_contact_depth))
        sphere_count = int(os.environ.get("TORC_FRANKA_PROXY_SPHERE_COUNT", "5"))
        min_points = int(os.environ.get("TORC_FRANKA_PROXY_MIN_POINTS", "1"))
        sphere_xs = np.linspace(-half_span, half_span, max(1, sphere_count))
        sphere_centers = np.stack(
            [
                sphere_xs,
                np.zeros_like(sphere_xs),
                np.full_like(sphere_xs, contact_depth),
            ],
            axis=1,
        )

        for pose in poses_world:
            T = pose_to_matrix(pose) if hasattr(pose, "position") else np.asarray(pose, dtype=np.float64)
            R = T[:3, :3]
            t = T[:3, 3]
            hit = set()
            for obj_idx, pts in zip(obj_indices, obj_points):
                local = (pts - t) @ R
                delta = local[:, None, :] - sphere_centers[None, :, :]
                inside = np.any(np.sum(delta * delta, axis=2) <= radius * radius, axis=1)
                if int(np.count_nonzero(inside)) >= min_points:
                    hit.add(obj_idx)
            collides_with.append(hit)
        return collides_with

    def _franka_pad_penetrates_object(self, poses_world, visible_points, mask, object_ids):
        """Reject Franka candidates whose real pad boxes already contain target points.

        TORC's original singularity test uses a Robotiq-specific proxy.  The
        Franka branch keeps that target-selection rule, but its Panda pads are
        thicker and must be checked with the current gripper box geometry so a
        candidate cannot pass simply because the proxy touched exactly one
        object while one finger pad is already inside that object.
        """
        points = np.asarray(visible_points, dtype=np.float64).reshape(-1, 3)
        labels = np.asarray(mask, dtype=np.uint32).reshape(-1)
        object_ids = np.asarray(object_ids, dtype=np.int32).reshape(-1)
        if len(points) == 0 or len(labels) != len(points):
            return np.zeros(len(poses_world), dtype=bool)

        default_pad_front = "0.0044"
        if RobotAdapter is not None:
            try:
                default_pad_front = str(RobotAdapter.derive_tcp_pad_front_m())
            except Exception:
                default_pad_front = "0.0044"

        opening_half = float(os.environ.get("TORC_FRANKA_PAD_OPENING_HALF_M", "0.040"))
        center_extra_x = float(os.environ.get("TORC_FRANKA_PAD_CENTER_EXTRA_X_M", "0.0035"))
        half_x = float(os.environ.get("TORC_FRANKA_PAD_HALF_X_M", "0.0040"))
        half_y = float(os.environ.get("TORC_FRANKA_PAD_HALF_Y_M", "0.0080"))
        half_z = float(os.environ.get("TORC_FRANKA_PAD_HALF_Z_M", "0.0080"))
        z_front = float(os.environ.get("TORC_FRANKA_PAD_FRONT_Z_M", default_pad_front))
        margin = float(os.environ.get("TORC_FRANKA_PAD_PENETRATION_MARGIN_M", "0.0010"))
        z_center = z_front - half_z
        centers = np.array(
            [
                [-(opening_half + center_extra_x), 0.0, z_center],
                [opening_half + center_extra_x, 0.0, z_center],
            ],
            dtype=np.float64,
        )
        half_extents = np.array([half_x, half_y, half_z], dtype=np.float64) + margin

        penetrates = []
        for pose, obj_id in zip(poses_world, object_ids):
            if obj_id < 0:
                penetrates.append(False)
                continue
            obj_mask = (labels & (1 << int(obj_id))).astype(bool)
            pts = points[obj_mask]
            if len(pts) == 0:
                penetrates.append(False)
                continue

            T = pose_to_matrix(pose) if hasattr(pose, "position") else np.asarray(pose, dtype=np.float64)
            R = T[:3, :3]
            t = T[:3, 3]
            local = (pts - t) @ R
            in_any_pad = np.zeros(len(local), dtype=bool)
            for center in centers:
                in_box = np.all(np.abs(local - center) <= half_extents, axis=1)
                in_any_pad |= in_box
            penetrates.append(bool(np.any(in_any_pad)))
        return np.asarray(penetrates, dtype=bool)

    def attach_extra_spheres(self):
        # add sphere to end-effector link for pre-grasp collision checking
        ee_link = self.extra_sphere_link
        kconf = self.kinematics_config
        max_s = kconf.get_number_of_spheres(ee_link)
        # print(robot_config.kinematics.robot_spheres[-2-max_s:])
        sphere_tensor = torch.zeros((max_s, 4))
        sphere_tensor[:, 3] = -10.0
        # print(sphere_tensor)
        for i in range(min(5, max_s)):
            sphere_tensor[i, 1] = -0.04 + 0.02 * i
            sphere_tensor[i, 3] = 0.012
        # print(sphere_tensor)
        kconf.attach_object(sphere_tensor=sphere_tensor, link_name=ee_link)

    def detach_extra_spheres(self):
        self.kinematics_config.disable_link_spheres(self.extra_sphere_link)
        self.kinematics_config.enable_link_spheres(self.extra_sphere_link)

    def toggle_link_collision(self, collision_link_names, enable_flag):
        if len(collision_link_names) > 0:
            if enable_flag:
                for k in collision_link_names:
                    self.kinematics_config.enable_link_spheres(k)
            else:
                for k in collision_link_names:
                    self.kinematics_config.disable_link_spheres(k)

    def set_collision_scene(self, points, world_dict=None, filter_pts=None):
        # init world config
        if world_dict is None:
            world_config = WorldConfig.from_dict(self.world_config)
        else:
            world_config = WorldConfig.from_dict(world_dict)

        # filter collision points by proximity to filter points
        if filter_pts is not None:
            # kdtree = KDTree(filter_pts[:, :2])
            # dist, ind = kdtree.query(points[:, :2])
            surface, find = tm.sample.sample_surface_even(
                tm.convex.convex_hull(filter_pts), 10000
            )
            kdtree = KDTree(surface)
            dist, ind = kdtree.query(points)
            points = points[dist > 0.05]
        points = self._points_for_collision_world(points)

        # convert point cloud to mesh
        scene_mesh = Mesh.from_pointcloud(points, self.resolution, "world")

        # add to world config
        world_config.add_obstacle(scene_mesh)

        # update collision world
        self.robot_world.clear_world_cache()
        self.robot_world.update_world(world_config)

    def get_joint_tensor_from_list(self, joints):
        joint_state_tensor = JointState.from_position(
            torch.tensor(np.array(joints), dtype=torch.float32).cuda(),
            joint_names=list(self.ik_solver.joint_names),
        )
        joint_state_tensor = self.robot_world.get_active_js(joint_state_tensor)
        q = joint_state_tensor.position
        return q

    def get_active_joint_list_from_tracik(self, joints):
        return (
            self.get_joint_tensor_from_list(joints)
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

    @staticmethod
    def normalize_score(scores):
        return 1 / (1 + np.exp(-np.divide(scores, 2000)))

    def is_grasp_above_table(
        self,
        pose,
        table_height,
        safety_margin=0.001,
        visualize=False,
    ):
        """
        Check if a single grasp pose would collide with table surface.
        Uses EXACT gripper geometry from GraspPlotter to calculate fingertip positions.

        Args:
            pose: Single Pose object or 4x4 matrix
            table_height: Height of table surface
            safety_margin: Additional clearance above table (default 0.01m)
            visualize: If True, visualize the grasp with collision info

        Returns:
            bool: True if grasp is valid (no table collision), False if rejected
        """
        threshold_height = table_height + safety_margin

        # Gripper geometry parameters (from GraspPlotter defaults)
        finger_width = self.plotter.finger_width  # 0.0065
        outer_diameter = self.plotter.outer_diameter  # 0.098
        hand_depth = self.plotter.hand_depth  # 0.055
        # hand_height = self.plotter.hand_height  # 0.025

        # Calculate half-width of gripper (EXACT match to grasp_plotter.py)
        hw = 0.5 * outer_diameter - 0.5 * finger_width  # 0.04575

        # Convert pose to matrix
        if hasattr(pose, "position"):  # ROS Pose message
            pose_matrix = pose_to_matrix(pose)
        else:  # assume it's already a matrix
            pose_matrix = pose

        # Extract grasp frame vectors (EXACT match to grasp_plotter.py)
        grasp_approach = pose_matrix[:3, 2]  # z-axis (approach direction)
        grasp_binormal = -pose_matrix[:3, 1]  # -y-axis (finger closing direction)
        grasp_bottom = pose_matrix[:3, 3]  # grasp center position

        left_bottom = grasp_bottom - hw * grasp_binormal
        right_bottom = grasp_bottom + hw * grasp_binormal
        left_bottom -= hand_depth * grasp_approach
        right_bottom -= hand_depth * grasp_approach
        left_top = left_bottom + hand_depth * grasp_approach
        right_top = right_bottom + hand_depth * grasp_approach
        left_center = left_bottom + 0.5 * (left_top - left_bottom)
        right_center = right_bottom + 0.5 * (right_top - right_bottom)

        left_finger_tip = left_center + 0.75 * hand_depth * grasp_approach
        right_finger_tip = right_center + 0.75 * hand_depth * grasp_approach
        left_finger_top = left_center - 0.75 * hand_depth * grasp_approach
        right_finger_top = right_center - 0.75 * hand_depth * grasp_approach

        # Find minimum Z coordinate among all fingertip points
        min_fingertip_z = min(
            left_finger_tip[2],
            right_finger_tip[2],
            left_finger_top[2],
            right_finger_top[2],
        )

        # Check if any fingertip would be below threshold
        is_valid = min_fingertip_z >= threshold_height

        if visualize:
            # Visualize grasp with appropriate color
            score = 1.0 if is_valid else -1.0  # green if valid, red if invalid
            self.plotter.draw_grasps()
            self.plotter.draw_grasps([pose], [score])

            # visualize finger tips in rviz using sphere markers
            markers = MarkerArray()
            for i, fingertip in enumerate(
                [left_finger_tip, right_finger_tip, left_finger_top, right_finger_top]
            ):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.id = i
                marker.ns = "fingertips"
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.scale.x = 0.01
                marker.scale.y = 0.01
                marker.scale.z = 0.01
                marker.color.a = 1.0
                if is_valid:
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                else:
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                marker.pose.position.x = fingertip[0]
                marker.pose.position.y = fingertip[1]
                marker.pose.position.z = fingertip[2]
                markers.markers.append(marker)
            self.plotter.rviz_pub.publish(markers)

            status = "VALID" if is_valid else "REJECTED"
            print(
                f"{status}: Min fingertip Z: {min_fingertip_z:.3f}m vs threshold: {threshold_height:.3f}m"
            )
            print(
                f"  Table height: {table_height:.3f}m + safety margin: {safety_margin:.3f}m"
            )
            print(
                f"  Left finger bottom: {left_finger_tip[2]:.3f}m, Right finger bottom: {right_finger_tip[2]:.3f}m"
            )
            print(f"  Grasp center Z: {grasp_bottom[2]:.3f}m")
            print(
                f"  Left/Right bottom (before hand_height offset): {left_bottom[2]:.3f}m, {right_bottom[2]:.3f}m"
            )
            input("Press Enter to continue...")

        return is_valid

    def get_ik_grasps(
        self,
        target_points,
        target_colors,
        visible_points,
        collision_points,
        mask,
        seg_label_to_obj_id=None,
        filter_outliers=False,
        visualize=False,
        rviz_func=None,
    ):
        if len(target_points) == 0:
            return "NO_GRASPS", [], [], []

        # ensure points are correct format
        t0 = time.time()
        if type(target_points[0]) is Point:
            target_points = [[p.x, p.y, p.z] for p in target_points]
        t1 = time.time()
        # print('Time to prepare points:', t1 - t0, flush=True)

        # target point cloud?
        self._torc_selected_grasp_debug = {}
        if filter_outliers or visualize:
            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(target_points)
            pcl.colors = o3d.utility.Vector3dVector(target_colors)
            if filter_outliers:
                pcl, _removed = pcl.remove_statistical_outlier(*filter_outliers)
            target_points = pcl.points
            target_colors = pcl.colors
        target_points = np.array(target_points)

        # sample surface plane points of workspace
        add_bottom_else_remove = True
        z_min = np.min(collision_points[:, 2])
        if add_bottom_else_remove:
            # lx, ly = np.min(visible_points[:, 0]), np.min(visible_points[:, 1])
            # hx, hy = np.max(visible_points[:, 0]), np.max(visible_points[:, 1])
            # print(lx, ly, hx, hy)
            lx, ly = self.vol_bnds[0, 0], self.vol_bnds[1, 0]
            hx, hy = self.vol_bnds[0, 1], self.vol_bnds[1, 1]
            # print(lx, ly, hx, hy)
            surface_plane = np.random.uniform(
                [lx, ly, z_min], [hx, hy, z_min], size=(200000, 3)
            )
            visible_points = np.concatenate([visible_points, surface_plane], axis=0)
            mask = np.concatenate([mask, np.zeros(len(surface_plane))])
        else:
            mask = np.array(mask)[visible_points[:, 2] > z_min + 0.01]
            visible_points = visible_points[visible_points[:, 2] > z_min + 0.01]
        mask = mask.astype(np.uint32)
        # print('Time to prepare visible points:', time.time() - t1, flush=True)

        if visualize:
            np.save("/tmp/target_points.npy", target_points)
            np.save("/tmp/visible_points.npy", visible_points)
            np.save("/tmp/object_mask.npy", mask)
            # v_pcl = o3d.geometry.PointCloud()
            # v_pcl.points = o3d.utility.Vector3dVector(visible_points)
            # o3d.visualization.draw_geometries([pcl, v_pcl])

        ## init grasp service request
        t0 = time.time()
        if GRASP_PLANNER == "graspnet":
            ccolors = [ColorRGBA(r, g, b, 0) for r, g, b in target_colors]
        elif GRASP_PLANNER == "gpd":
            header = Header()
            header.frame_id = "world"
            cloud = pcl2.create_cloud_xyz32(header, visible_points)
            # camera_position = Point(0.14, 0, 1.4)
            camera_position = Point(0, 0, 0)
        elif GRASP_PLANNER == "cgn_pytorch":
            pts_arr = Float32MultiArray(data=visible_points.flatten().tolist())
        elif GRASP_PLANNER == "cgn":
            pts_arr = visible_points.flatten().tolist()
            mask_arr = mask.tolist()
            if _env_flag("TORC_DUMP_CGN_CALLSITE"):
                capture_path = _dump_cgn_callsite(
                    visible_points,
                    mask,
                    target_points=target_points,
                    collision_points=collision_points,
                    vol_bnds=self.vol_bnds,
                    z_min=z_min,
                )
                if _env_flag("TORC_CGN_CAPTURE_ONLY"):
                    raise CgnCallsiteCaptureOnly(
                        "TORC_CGN_CAPTURE_ONLY=1 captured CGN input and stopped before "
                        f"waiting for old ROS CGN service: {capture_path}"
                    )
            # tgt_arr = Float32MultiArray(data=target_points.flatten().tolist())
            # # low_z = np.percentile(target_points[:, 2], 10)
            # # low_v_pts = visible_points[visible_points[:, 2] < low_z]
            # new_vis_pts = np.vstack([target_points, surface_plane])
            # pts_arr = Float32MultiArray(data=new_vis_pts.flatten().tolist())
            # # pts_arr = Float32MultiArray(data=visible_points.flatten().tolist())
        elif GRASP_PLANNER == "ground_truth":
            pass
        print("Time to prepare service request:", time.time() - t0, flush=True)
        print("Number of points:", len(visible_points), flush=True)
        print("Number of target points:", len(target_points), flush=True)

        ## call service
        poses = []
        scores = []
        samples = []
        object_ids = []
        # print(self.grasp_srvs, flush=True)

        if GRASP_PLANNER == "ground_truth":
            obj_ids_to_names = rospy.wait_for_message(
                "/ground_truth/object_ids_to_names", ObjectIdsToNames
            )
            oid2name = dict(zip(obj_ids_to_names.obj_ids, obj_ids_to_names.names))
            # seg_img = rospy.wait_for_message(
            #     "/ground_truth/camera0/seg_image", Image
            # )
            # seg_raw = self.bridge.imgmsg_to_cv2(seg_img, 'rgb8')
            # seg = decode_seg_img_rgb(seg_raw)
            # ids = set(seg.flatten())
            obj_poses = rospy.wait_for_message(
                "/ground_truth/object_poses", ObjectPoses
            )
            name2pose = dict(zip(obj_poses.name, obj_poses.pose))
            i = 1
            aid2name = {}
            for i in range(32):
                tgt_mask = ((1 << i) & mask).astype(bool)
                tgt_pts = visible_points[tgt_mask]
                if len(tgt_pts) == 0:
                    continue
                obj_id = seg_label_to_obj_id[i]
                obj_name = oid2name[obj_id]
                aid2name[i] = obj_name

                obj_name_adj = "_".join(obj_name.split("_")[:-1])
                if obj_name_adj not in self.name2grasp:
                    print(f"No grasps for {obj_name}", flush=True)
                    continue
                grasps = self.name2grasp[obj_name_adj]
                # if len(grasps) > 500:
                #     grasps = grasps[:-4:4]
                # if len(grasps) > 250:
                #     grasps = grasps[:-2:2]
                # num_samples = min(500, len(grasps))
                # grasps = np.array(grasps)[np.random.choice(
                #     range(len(grasps)), num_samples, replace=False
                # )]

                obj_pose = name2pose[obj_name]
                obj_matrix = pose_to_matrix(obj_pose)
                for g in grasps:
                    # swap x and z axes
                    grasp = g @ np.array(
                        [
                            [0.0, 0.0, 1.0, 0.0],
                            [0, -1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    )
                    grasp_matrix = np.matmul(obj_matrix, grasp)
                    v_dir = grasp_matrix[:3, 2]
                    approach = v_dir / np.linalg.norm(v_dir)
                    # for score, displace in zip([1.0, 0.75, 0.5],
                    #                            [0.00, 0.015, 0.025]):
                    #     grasp_t = copy.deepcopy(grasp_matrix)
                    #     grasp_t[:3, 3] -= displace * approach
                    grasp_matrix[:3, 3] += 0.035 * approach
                    valid = self.is_grasp_above_table(
                        grasp_matrix, z_min, visualize=False
                    )
                    for attempt in range(3):
                        if valid:
                            break
                        grasp_matrix[:3, 3] -= 0.012 * approach
                        valid = self.is_grasp_above_table(
                            grasp_matrix, z_min, visualize=False
                        )
                    if not valid:
                        continue
                    score = (3 - attempt) / 3.0
                    grasp_pose = matrix_to_pose(grasp_matrix)
                    poses.append(grasp_pose)
                    scores.append(score)
                    samples.append(None)
                    object_ids.append(i)
        elif GRASP_PLANNER == "cgn" and _env_flag("TORC_USE_CGN_ZMQ"):
            zmq_poses, zmq_scores, zmq_samples, zmq_object_ids = _call_cgn_zmq(
                visible_points,
                mask,
            )
            poses.extend(zmq_poses)
            scores.extend(zmq_scores)
            samples.extend(zmq_samples)
            object_ids.extend(zmq_object_ids)
        else:
            for srv_name in self.grasp_srvs:
                rospy.wait_for_service(srv_name, timeout=10)
                try:
                    service = rospy.ServiceProxy(srv_name, GetGrasps)

                    if GRASP_PLANNER == "cgn":
                        # for i in range(32):
                        #     tgt_mask = ((1 << i) & mask).astype(bool)
                        #     tgt_pts = visible_points[tgt_mask]
                        #     if len(tgt_pts) == 0:
                        #         continue
                        #     m_arr = [1 << i] * len(tgt_pts)
                        #     back_pts = visible_points[~mask.astype(bool)]
                        #     tgt_pts = np.vstack([tgt_pts, back_pts])
                        #     m_arr.extend([0] * len(back_pts))
                        #     tgt_arr = tgt_pts.flatten().tolist()
                        #     resp = service(tgt_arr, m_arr)
                        resp = service(pts_arr, mask_arr)
                        poses.extend(resp.grasps.poses)
                        scores.extend(resp.grasps.scores)
                        samples.extend(resp.grasps.samples)
                        object_ids.extend(resp.grasps.object_ids)
                    else:
                        for i in range(32):
                            tgt_mask = ((1 << i) & mask).astype(bool)
                            tgt_pts = visible_points[tgt_mask]
                            if len(tgt_pts) == 0:
                                continue
                            ppoints = [Point(x, y, z) for x, y, z in tgt_pts]

                            resp_scores = None
                            if GRASP_PLANNER == "cgn_pytorch":
                                resp = service(pts_arr, tgt_mask, 0.9)
                            elif GRASP_PLANNER == "gpd":
                                resp = service(ppoints, cloud, camera_position)
                                resp_scores = self.normalize_score(resp.grasps.scores)
                            elif GRASP_PLANNER == "gpg":
                                # m = PerceptionInterface.get_shape_estimate(tgt_pts, scale=0.25)
                                # tm.Scene([m, tm.PointCloud(tgt_pts)]).show()
                                # cvx_pts = tm.sample.sample_surface(m, 10000)[0]
                                # tgt_arr = cvx_pts.flatten().tolist()
                                tgt_arr = tgt_pts.flatten().tolist()
                                resp = service(tgt_arr, 1000)
                                resp_scores = [1.0] * len(resp.grasps.poses)
                                # self.plotter.draw_grasps()
                                # self.plotter.draw_grasps(resp.grasps.poses, [i] * len(resp.grasps.poses))
                                # input('Continue...')
                            elif GRASP_PLANNER == "graspnet":
                                resp = service(ppoints, ccolors)

                            if len(resp.grasps.poses) == 0:
                                continue

                            poses.extend(resp.grasps.poses)
                            if resp_scores is None:
                                resp_scores = resp.grasps.scores
                            scores.extend(resp_scores)
                            samples.extend(resp.grasps.samples)
                            object_ids.extend([i] * len(resp.grasps.poses))

                except rospy.ServiceException as e:
                    print("Service call failed:", e, flush=True)
                    # return 'NO_GRASPS', [], [], []

        if len(poses) == 0:
            print("No grasps found!", flush=True)
            return "NO_GRASPS", [], [], []
        if len(object_ids) == 0:
            object_ids = [0] * len(poses)
        _stage_probe(
            "raw grasp ids",
            f"count={len(object_ids)} unique={sorted(set(int(v) for v in object_ids)) if object_ids else []}",
        )
        raw_source_indices = list(range(len(poses)))
        if _env_flag("TORC_CAPTURE_SELECTED_GRASP"):
            self._torc_selected_grasp_debug.update(
                {
                    "raw_cgn_grasp_matrices": np.asarray(
                        [_pose_matrix_or_none(p) for p in poses],
                        dtype=np.float64,
                    ),
                    "raw_cgn_scores": np.asarray(scores, dtype=np.float64),
                    "raw_cgn_samples": np.asarray(
                        [
                            _point_xyz_or_none(s)
                            if _point_xyz_or_none(s) is not None
                            else [np.nan, np.nan, np.nan]
                            for s in samples
                        ],
                        dtype=np.float64,
                    ),
                    "raw_cgn_object_ids": np.asarray(object_ids, dtype=np.int64),
                    "raw_cgn_source_indices": np.asarray(
                        raw_source_indices,
                        dtype=np.int64,
                    ),
                    "raw_cgn_count": int(len(poses)),
                    "hand_depth_m": float(self.hand_depth),
                    "grasp_planner": GRASP_PLANNER,
                    "frame_labels": {
                        "raw_cgn_grasp_matrices": "T_world_cgn_raw",
                        "ik_candidate_matrices": "T_world_motoman_ee_goal_after_plus_hand_depth_local_z",
                        "validated_candidate_matrices": "T_world_motoman_ee_goal_after_all_grasp_filters",
                        "validated_pregrasp_matrices": "T_world_motoman_ee_pregrasp",
                    },
                }
            )

        t1 = time.time()
        print("Grasp service time:", t1 - t0, flush=True)
        print(len(poses), "grasps recieved.", flush=True)

        # if visualize:
        #     self.plotter.draw_grasps(poses, scores)

        ## ik filter ##
        t0 = time.time()
        backup_poses = []
        ik_v_pose_t = []
        ik_v_scores = []
        ik_v_samples = []
        ik_joints = []
        ik_v_obj_ids = []

        ik_source_indices = []
        zipped = zip(poses, scores, samples, object_ids, raw_source_indices)
        for pose, score, sample, obj_id, source_index in zipped:
            # reject score threshold
            # if score < 0.5:
            #     continue

            # adjusted grasp
            pose_t = pose_to_matrix(pose)
            approach_t = pose_t[:3, 2] / np.linalg.norm(pose_t[:3, 2])
            displace_t = self.hand_depth
            pose_t[:3, 3] = pose_t[:3, 3] + displace_t * approach_t
            backup_poses.append(pose_t)

            js = self.ik_solver.ik(self._pose_for_tracik(pose_t))
            if js is not None:
                ik_joints.append(js)
                ik_v_pose_t.append(pose_t)
                ik_v_scores.append(score)
                ik_v_samples.append(sample)
                ik_v_obj_ids.append(obj_id)
                ik_source_indices.append(source_index)
        t1 = time.time()
        print("IK time:", t1 - t0, flush=True)
        print(len(ik_v_pose_t), "grasps after IK filtering.", flush=True)
        _stage_probe(
            "ik grasp ids",
            f"count={len(ik_v_obj_ids)} unique={sorted(set(int(v) for v in ik_v_obj_ids)) if len(ik_v_obj_ids) else []}",
        )
        if len(ik_v_pose_t) == 0:
            return "IK_INFEASIBLE", backup_poses, [], scores
        if _env_flag("TORC_CAPTURE_SELECTED_GRASP"):
            self._torc_selected_grasp_debug.update(
                {
                    "ik_candidate_matrices": np.asarray(
                        ik_v_pose_t,
                        dtype=np.float64,
                    ),
                    "ik_candidate_scores": np.asarray(
                        ik_v_scores,
                        dtype=np.float64,
                    ),
                    "ik_candidate_object_ids": np.asarray(
                        ik_v_obj_ids,
                        dtype=np.int64,
                    ),
                    "ik_candidate_source_indices": np.asarray(
                        ik_source_indices,
                        dtype=np.int64,
                    ),
                    "ik_candidate_count": int(len(ik_v_pose_t)),
                }
            )

        if visualize:
            filtered_pose = []
            filtered_id = []
            c = {}
            zipped = sorted(
                zip(ik_v_pose_t, ik_v_obj_ids, ik_v_scores), key=lambda x: -x[2]
            )
            for pose, oid, score in zipped:
                count = c.get(oid, 0)
                if count < 10:
                    filtered_pose.append(pose)
                    filtered_id.append(oid)
                c[oid] = count + 1
            self.plotter.draw_grasps()
            self.plotter.draw_grasps(filtered_pose, filtered_id)
            # self.plotter.draw_grasps(ik_v_pose_t, ik_v_obj_ids)
            # input('Continue...')

        ## debug visualization ##
        if visualize:
            # wm = self.robot_world.world_model.world_model
            # wc = wm.get_collision_check_world()
            # WorldConfig.get_scene_graph(wc).to_mesh().show()
            # max_grasp = matrix_to_pose(ik_v_pose_t[np.argmax(ik_v_scores)])
            # self.plotter.draw_grasps([max_grasp], [1])
            # self.plotter.draw_grasps(
            #     [matrix_to_pose(p) for p in ik_v_pose_t],
            #     ik_v_scores,
            # )
            if rviz_func is not None:
                joint_state0 = rospy.wait_for_message(
                    "/joint_states_all", JointState_MSG, timeout=5
                )
                i = 0
                for pose, jnt_vals, score, obj_id in zip(
                    ik_v_pose_t, ik_joints, ik_v_scores, ik_v_obj_ids
                ):
                    i += 1
                    if i % 10 != 0:
                        continue
                    inds = list(
                        map(joint_state0.name.index, self.ik_solver.joint_names)
                    )
                    vis_state = JointState_MSG()
                    vis_state.name = joint_state0.name
                    vis_state.position = np.array(joint_state0.position)
                    vis_state.position[inds] = jnt_vals
                    rviz_func(vis_state, self.robot_world.kinematics)
                    self.plotter.draw_grasps()
                    self.plotter.draw_grasps([pose], [score])
                    if GRASP_PLANNER == "ground_truth":
                        print(f"Grasping object: {aid2name[obj_id]}")
                    input(f"{score} next?")
                    self.is_grasp_above_table(pose, z_min, visualize=True)

        return (
            ik_v_pose_t,
            ik_joints,
            ik_v_scores,
            ik_v_samples,
            ik_v_obj_ids,
            visible_points,
            mask,
        )

    def validate_grasps(
        self,
        ik_v_pose_t,
        ik_joints,
        ik_v_scores,
        ik_v_samples,
        ik_v_obj_ids,
        visible_points,
        collision_points,
        mask=None,
        singularity_points=None,
        singularity_mask=None,
    ):
        backup_poses = ik_v_pose_t
        scores = ik_v_scores
        debug_source_indices = np.asarray(
            self._torc_selected_grasp_debug.get(
                "ik_candidate_source_indices",
                np.arange(len(ik_v_pose_t)),
            ),
            dtype=np.int64,
        )

        ## set collision scene ##
        t0 = time.time()
        self.set_collision_scene(visible_points)
        t1 = time.time()
        print("Set collision scene time:", t1 - t0, flush=True)

        ## collision filter against target ##
        t0 = time.time()
        if self.is_franka_robot:
            self.toggle_link_collision(self.disable_ee_links, False)
        else:
            self.toggle_link_collision(self.disable_ee_links, True)
        # order joints
        q = self.get_joint_tensor_from_list(ik_joints)
        # collision check
        res = self.robot_world.get_world_self_collision_distance_from_joints(q)
        d_world, d_self = res
        print("Penetration Depth: ", np.histogram(d_world.cpu().numpy(), bins=10))
        valid = ((d_world <= 0.01) & (d_self <= 0)).cpu().numpy()
        c_v_pose_t = np.array(ik_v_pose_t)[valid]
        c_v_scores = np.array(ik_v_scores)[valid]
        c_v_samples = np.array(ik_v_samples)[valid]
        c_v_obj_ids = np.array(ik_v_obj_ids)[valid]
        c_source_indices = debug_source_indices[valid]
        c_joints = q[valid]
        t1 = time.time()
        print("Collision time:", t1 - t0, flush=True)
        print(len(c_v_pose_t), "grasps after target collision filtering.", flush=True)
        _stage_probe(
            "target collision grasp ids",
            "count={} unique={}".format(
                len(c_v_obj_ids),
                sorted(set(int(v) for v in c_v_obj_ids)) if len(c_v_obj_ids) else [],
            ),
        )
        if len(c_v_pose_t) == 0:
            # return 'IN_COLLISION_TARGET', ik_v_pose_t, ik_joints, ik_v_scores
            return "IN_COLLISION_TARGET", backup_poses, [], scores

        ## ensure only grasping one object ##
        if mask is not None:
            # order joints
            q = c_joints
            if self.is_franka_robot:
                t0 = time.time()
                proxy_points = visible_points
                proxy_mask = mask
                if singularity_points is not None and singularity_mask is not None:
                    proxy_points = singularity_points
                    proxy_mask = singularity_mask
                collides_with = self._franka_single_object_collides_with(
                    c_v_pose_t, proxy_points, proxy_mask
                )
                t1 = time.time()
                print("Franka capture-window singularity time:", t1 - t0, flush=True)
            else:
                self.attach_extra_spheres()
                num_objs = len(np.binary_repr(max(mask)))
                # print('num_objs', num_objs, set(c_v_obj_ids))
                rw = self.robot_world
                collides_with = [set() for _ in range(len(c_v_pose_t))]
                for i in range(num_objs):
                    obj_vis_mask = (mask & (1 << i)).astype(bool)
                    vis_points = visible_points[obj_vis_mask]
                    if len(vis_points) == 0:
                        continue

                    ## set collision scene ##
                    t0 = time.time()
                    self.set_collision_scene(vis_points)
                    t1 = time.time()
                    print("Set collision scene time:", t1 - t0, flush=True)
                    # collision check
                    res = rw.get_world_self_collision_distance_from_joints(q)
                    d_world, d_self = res
                    # print(
                    #     "Penetration Depth: ", np.histogram(d_world.cpu().numpy(), bins=10)
                    # )
                    colliding = (d_world > 0).cpu().numpy()
                    t2 = time.time()
                    print("Collision time:", t2 - t1, flush=True)
                    for j, coll in enumerate(colliding):
                        if coll:
                            collides_with[j].add(i)

            # check grasp object singularity
            if self.is_franka_robot:
                hit_sizes = {}
                for obj_id, hit in zip(c_v_obj_ids, collides_with):
                    key = (int(obj_id), len(hit))
                    hit_sizes[key] = hit_sizes.get(key, 0) + 1
                _stage_probe(
                    "franka singularity proxy hit sizes",
                    " ".join(
                        f"obj={obj}:hits={hits}:count={count}"
                        for (obj, hits), count in sorted(hit_sizes.items())
                    ),
                )
            valid = np.array([len(s) == 1 for s in collides_with])
            if self.is_franka_robot:
                penetrates_target = self._franka_pad_penetrates_object(
                    c_v_pose_t, proxy_points, proxy_mask, c_v_obj_ids
                )
                if np.any(penetrates_target):
                    _stage_probe(
                        "franka pad penetration rejection",
                        f"rejected={int(np.count_nonzero(penetrates_target))} "
                        f"remaining_before={len(valid)}",
                    )
                valid = valid & (~penetrates_target)
            c_v_pose_t = np.array(c_v_pose_t)[valid]
            c_v_scores = np.array(c_v_scores)[valid]
            c_v_samples = np.array(c_v_samples)[valid]
            c_v_obj_ids = np.array(c_v_obj_ids)[valid]
            c_source_indices = np.array(c_source_indices)[valid]
            c_joints = q[valid]
            # t1 = time.time()
            # print('Collision time:', t1 - t0, flush=True)
            print(
                len(c_v_pose_t),
                "grasps after target singularity filtering.",
                flush=True,
            )
            _stage_probe(
                "target singularity grasp ids",
                f"count={len(c_v_obj_ids)} unique={sorted(set(int(v) for v in c_v_obj_ids)) if len(c_v_obj_ids) else []}",
            )
            if len(c_v_pose_t) == 0:
                # return 'NO_SINGULAR_GRASPS', ik_v_pose_t, ik_joints, ik_v_scores
                return "NO_SINGULAR_GRASPS", backup_poses, [], scores

        ## set collision scene ##
        t0 = time.time()
        self.set_collision_scene(collision_points)
        t1 = time.time()
        print("Set collision scene time:", t1 - t0, flush=True)

        ## collision filter against the scene ##
        t0 = time.time()
        self.toggle_link_collision(self.disable_ee_links, False)
        # order joints
        q = c_joints
        # collision check
        res = self.robot_world.get_world_self_collision_distance_from_joints(q)
        d_world, d_self = res
        valid = ((d_world <= 0) & (d_self <= 0)).cpu().numpy()
        c_v_pose_t = np.array(c_v_pose_t)[valid]
        c_v_scores = np.array(c_v_scores)[valid]
        c_v_samples = np.array(c_v_samples)[valid]
        c_v_obj_ids = np.array(c_v_obj_ids)[valid]
        c_source_indices = np.array(c_source_indices)[valid]
        c_joints = q[valid]
        t1 = time.time()
        print("Collision time:", t1 - t0, flush=True)
        print(len(c_v_pose_t), "grasps after scene collision filtering.", flush=True)
        _stage_probe(
            "scene collision grasp ids",
            f"count={len(c_v_obj_ids)} unique={sorted(set(int(v) for v in c_v_obj_ids)) if len(c_v_obj_ids) else []}",
        )
        if len(c_v_pose_t) == 0:
            # return 'IN_COLLISION_SCENE', ik_v_pose_t, ik_joints, ik_v_scores
            return "IN_COLLISION_SCENE", backup_poses, [], scores

        ## set collision scene ##
        # t0 = time.time()
        # self.set_collision_scene(collision_points)
        # t1 = time.time()
        # print('Set collision scene time:', t1 - t0, flush=True)

        ## generate pre grasps ##
        t0 = time.time()
        result = self.find_nearest_grasp_retraction(
            c_v_pose_t,
            c_v_scores,
            step=0.02,
            maxIters=3,
            return_mask=True,
        )
        p_v_pre_pose_t, p_pre_joints, p_v_scores, valid = result

        p_v_pose_t = [matrix_to_pose(m) for m in np.array(c_v_pose_t)[valid]]
        p_v_samples = np.array(c_v_samples)[valid].tolist()
        p_v_obj_ids = np.array(c_v_obj_ids)[valid].tolist()
        p_source_indices = np.asarray(c_source_indices)[valid]
        p_joints = c_joints[valid].cpu().tolist()
        t1 = time.time()
        print(len(p_v_pose_t), "grasps after pre-grasp filtering.", flush=True)
        print("Pre-grasp total time:", t1 - t0, flush=True)
        _stage_probe(
            "pregrasp grasp ids",
            f"count={len(p_v_obj_ids)} unique={sorted(set(int(v) for v in p_v_obj_ids)) if len(p_v_obj_ids) else []}",
        )
        if len(p_v_pose_t) == 0:
            # return 'IN_COLLISION_PRE_GRASPS', c_v_pose_t, c_joints, c_v_scores
            return "IN_COLLISION_PRE_GRASPS", backup_poses, [], scores
        if _env_flag("TORC_CAPTURE_SELECTED_GRASP"):
            self._torc_selected_grasp_debug.update(
                {
                    "validated_candidate_matrices": np.asarray(
                        [pose_to_matrix(p) for p in p_v_pose_t],
                        dtype=np.float64,
                    ),
                    "validated_pregrasp_matrices": np.asarray(
                        [pose_to_matrix(p) for p in p_v_pre_pose_t],
                        dtype=np.float64,
                    ),
                    "validated_candidate_scores": np.asarray(
                        p_v_scores,
                        dtype=np.float64,
                    ),
                    "validated_candidate_object_ids": np.asarray(
                        p_v_obj_ids,
                        dtype=np.int64,
                    ),
                    "validated_candidate_source_indices": np.asarray(
                        p_source_indices,
                        dtype=np.int64,
                    ),
                    "validated_candidate_count": int(len(p_v_pose_t)),
                }
            )

        # self.plotter.draw_grasps(p_v_pose_t, p_v_scores)
        # self.plotter.draw_grasps(p_v_pre_pose_t, [0] * len(p_v_pre_pose_t))

        return (
            p_v_pose_t,
            p_v_pre_pose_t,
            p_joints,
            p_pre_joints,
            p_v_scores,
            p_v_samples,
            p_v_obj_ids,
        )

    def find_nearest_grasp_retraction(
        self,
        grasps,
        scores,
        step=0.1,
        maxIters=4,
        return_mask=False,
        visualize=False,
        rviz_func=None,
    ):
        ## set collision scene ##
        # t0 = time.time()
        # self.set_collision_scene(collision_points)
        # t1 = time.time()
        # print('Set collision scene time:', t1 - t0, flush=True)

        t0 = time.time()

        self.toggle_link_collision(self.disable_ee_links, True)

        valid = np.zeros(len(grasps), dtype=bool)
        i = 0
        pre_grasps = np.array([None] * len(grasps))
        pre_grasp_joints = np.array([None] * len(grasps))
        pre_grasp_scores = np.array([None] * len(grasps))
        while i < maxIters and np.count_nonzero(valid) < 5:
            i += 1
            for j, pose_t, score in zip(range(len(grasps)), grasps, scores):
                # adjusted grasp
                if type(pose_t) is Pose:
                    p_pose_t = pose_to_matrix(pose_t)
                else:
                    p_pose_t = copy.deepcopy(pose_t)
                approach_t = p_pose_t[:3, 2] / np.linalg.norm(p_pose_t[:3, 2])
                displace_t = -step * (i + 1)
                p_pose_t[:3, 3] = p_pose_t[:3, 3] + displace_t * approach_t

                js = self.ik_solver.ik(self._pose_for_tracik(p_pose_t))
                if js is not None and not valid[j]:
                    valid[j] = True
                    pre_grasps[j] = matrix_to_pose(p_pose_t)
                    pre_grasp_joints[j] = js
                    pre_grasp_scores[j] = score

            if not np.any(valid):
                continue

            ## debug visualization ##
            if visualize:
                non_null = pre_grasps[valid].tolist()
                color = [0] * len(non_null)
                self.plotter.draw_grasps(non_null, color)
                # g = matrix_to_pose(grasps[np.argmax(pre_grasp_scores)])
                # self.plotter.draw_grasps([g], [1])
                if rviz_func:
                    joint_state0 = rospy.wait_for_message(
                        "/joint_states_all", JointState_MSG, timeout=5
                    )
                    for jnt_vals, score in zip(pre_grasp_joints, pre_grasp_scores):
                        inds = list(
                            map(joint_state0.name.index, self.ik_solver.joint_names)
                        )
                        vis_state = JointState_MSG()
                        vis_state.name = joint_state0.name
                        vis_state.position = np.array(joint_state0.position)
                        vis_state.position[inds] = jnt_vals
                        rviz_func(vis_state, self.robot_world.kinematics)
                        input(f"{score} next?")

            # order joints
            non_null_js = np.stack(pre_grasp_joints[valid])
            q = self.get_joint_tensor_from_list(non_null_js)

            # collision check
            rw = self.robot_world
            res = rw.get_world_self_collision_distance_from_joints(q)
            d_world, d_self = res
            # print(d_self, d_world)
            valid[valid] = ((d_world <= 0) & (d_self <= 0)).cpu().numpy()

        if np.any(valid):
            pre_grasps = pre_grasps[valid].tolist()
            pre_grasp_scores = np.stack(pre_grasp_scores[valid]).tolist()
            pre_grasp_joints = self.get_active_joint_list_from_tracik(
                np.stack(pre_grasp_joints[valid])
            )
        else:
            pre_grasps = []
            pre_grasp_scores = []
            pre_grasp_joints = []

        t1 = time.time()
        print("Retracted grasp search time:", t1 - t0, flush=True)
        print("Retracted grasp search iters:", i, flush=True)
        print(len(pre_grasps), "found.", flush=True)

        if return_mask:
            return pre_grasps, pre_grasp_joints, pre_grasp_scores, valid
        return pre_grasps, pre_grasp_joints, pre_grasp_scores

    def get_grasp_poses(
        self,
        target_points,
        target_colors,
        visible_points,
        collision_points,
        mask=None,
        seg_label_to_obj_id=None,
        filter_outliers=False,
        visualize=False,
        rviz_func=None,
    ):

        result = self.get_ik_grasps(
            target_points,
            target_colors,
            visible_points,
            collision_points,
            mask,
            seg_label_to_obj_id,
            filter_outliers,
            visualize,
            rviz_func,
        )
        if len(result) != 7:
            return result
        ik_v_pose_t = result[0]
        ik_joints = result[1]
        ik_v_scores = result[2]
        ik_v_samples = result[3]
        ik_v_obj_ids = result[4]
        aug_visible_points = result[5]
        aug_mask = result[6]
        return self.validate_grasps(
            ik_v_pose_t,
            ik_joints,
            ik_v_scores,
            ik_v_samples,
            ik_v_obj_ids,
            aug_visible_points,
            collision_points,
            aug_mask,
        )

    def get_grasp_collisions(
        self,
        ik_v_pose_t,
        ik_joints,
        ik_v_scores,
        ik_v_samples,
        ik_v_obj_ids,
        visible_points,
        collision_points,
        vmask,
        cmask,
    ):
        rw = self.robot_world
        collides_with = [set() for _ in range(len(ik_v_pose_t))]
        num_objs = len(np.binary_repr(max(vmask)))
        for i in range(num_objs):
            obj_vis_mask = (vmask & (1 << i)).astype(bool)
            vis_points = visible_points[obj_vis_mask]
            obj_cmask = (cmask & (1 << i)).astype(bool)
            col_points = collision_points[obj_cmask]

            if len(vis_points) == 0:
                continue
            ## set collision scene ##
            t0 = time.time()
            self.set_collision_scene(vis_points, world_dict={})
            ## collision filter against target ##
            self.toggle_link_collision(self.disable_ee_links, False)
            t1 = time.time()
            print("Set collision scene time:", t1 - t0, flush=True)

            # order joints
            q = self.get_joint_tensor_from_list(ik_joints)
            # collision check
            res = rw.get_world_self_collision_distance_from_joints(q)
            d_world, d_self = res
            colliding = (d_world > 0.01).cpu().numpy()
            t2 = time.time()
            print("Collision time:", t2 - t1, flush=True)
            for j, coll in enumerate(colliding):
                if coll:
                    collides_with[j].add(i)

            if len(col_points) == 0:
                continue
            ## set collision scene ##
            t0 = time.time()
            self.set_collision_scene(col_points, world_dict={})
            ## collision filter against the scene ##
            self.toggle_link_collision(self.disable_ee_links, False)
            t1 = time.time()
            print("Set collision scene time:", t1 - t0, flush=True)

            # collision check
            res = rw.get_world_self_collision_distance_from_joints(q)
            d_world, d_self = res
            colliding = (d_world > 0).cpu().numpy()
            t2 = time.time()
            print("Collision time:", t2 - t1, flush=True)
            for j, coll in enumerate(colliding):
                if coll:
                    collides_with[j].add(i)

        return collides_with

    def valid_pose(self, pose, qinit=None):

        if not isinstance(pose, np.ndarray):
            pose = pose_to_matrix(pose)
        tracik_pose = self._pose_for_tracik(pose)
        js = self.ik_solver.ik(tracik_pose, qinit=qinit)
        if js is None:
            for i in range(3):
                js = self.ik_solver.ik(tracik_pose)
            if js is None:
                return False

        joint_state_tensor = JointState.from_position(
            torch.tensor(np.array(js)[np.newaxis, :], dtype=torch.float32).cuda(),
            joint_names=list(self.ik_solver.joint_names),
        )
        joint_state_tensor = self.robot_world.get_active_js(joint_state_tensor)
        q = joint_state_tensor.position
        # collision check
        res = self.robot_world.get_world_self_collision_distance_from_joints(q)
        d_world, d_self = res
        # print(d_self, d_world)
        valid = ((d_world <= 0) & (d_self <= 0)).cpu().numpy()

        # print("[k_user] ================= valid_pose", valid)
        return valid[0]
