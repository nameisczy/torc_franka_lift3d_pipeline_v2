from dataclasses import dataclass
import os

import numpy as np

from utils.conversions import pose_to_matrix

try:
    from robot_interface.franka_adapter import RobotAdapter
except Exception:
    RobotAdapter = None


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.lower() not in ("0", "false", "no", "off")


@dataclass(frozen=True)
class FrankaGraspGeometryConfig:
    pre_filter_offset_scan: bool = True
    scan_min_object_points: int = 20
    scan_points_per_object: int = 2500
    scan_dx_extent_m: float = 0.012
    scan_dy_extent_m: float = 0.008
    scan_forward_extent_m: float = 0.020
    scan_backward_extent_m: float = 0.002
    scan_step_m: float = 0.004
    scan_target_clearance_m: float = 0.001
    pad_opening_half_m: float = 0.040
    pad_half_x_m: float = 0.0040
    pad_half_y_m: float = 0.0080
    pad_half_z_m: float = 0.0080
    pad_center_extra_x_m: float = 0.0035
    pad_penetration_margin_m: float = 0.0010
    singularity_points_per_object: int = 2500
    proxy_half_span_m: float = 0.040
    proxy_radius_m: float = 0.008
    proxy_sphere_count: int = 5
    proxy_min_points: int = 1
    enclosure_min_cross_m: float = 0.006
    enclosure_max_center_m: float = 0.022
    enclosure_min_points: int = 20
    enclosure_score_weight: float = 0.35

    @classmethod
    def from_env(cls):
        return cls(
            pre_filter_offset_scan=_env_bool("TORC_FRANKA_PRE_FILTER_OFFSET_SCAN", cls.pre_filter_offset_scan),
            scan_min_object_points=_env_int("TORC_FRANKA_SCAN_MIN_OBJECT_POINTS", cls.scan_min_object_points),
            scan_points_per_object=_env_int("TORC_FRANKA_SCAN_POINTS_PER_OBJECT", cls.scan_points_per_object),
            scan_dx_extent_m=_env_float("TORC_FRANKA_SCAN_DX_EXTENT_M", cls.scan_dx_extent_m),
            scan_dy_extent_m=_env_float("TORC_FRANKA_SCAN_DY_EXTENT_M", cls.scan_dy_extent_m),
            scan_forward_extent_m=_env_float("TORC_FRANKA_SCAN_FORWARD_EXTENT_M", cls.scan_forward_extent_m),
            scan_backward_extent_m=_env_float("TORC_FRANKA_SCAN_BACKWARD_EXTENT_M", cls.scan_backward_extent_m),
            scan_step_m=_env_float("TORC_FRANKA_SCAN_STEP_M", cls.scan_step_m),
            scan_target_clearance_m=_env_float("TORC_FRANKA_SCAN_TARGET_CLEARANCE_M", cls.scan_target_clearance_m),
            pad_opening_half_m=_env_float("TORC_FRANKA_PAD_OPENING_HALF_M", cls.pad_opening_half_m),
            pad_half_x_m=_env_float("TORC_FRANKA_PAD_HALF_X_M", cls.pad_half_x_m),
            pad_half_y_m=_env_float("TORC_FRANKA_PAD_HALF_Y_M", cls.pad_half_y_m),
            pad_half_z_m=_env_float("TORC_FRANKA_PAD_HALF_Z_M", cls.pad_half_z_m),
            pad_center_extra_x_m=_env_float("TORC_FRANKA_PAD_CENTER_EXTRA_X_M", cls.pad_center_extra_x_m),
            pad_penetration_margin_m=_env_float("TORC_FRANKA_PAD_PENETRATION_MARGIN_M", cls.pad_penetration_margin_m),
            singularity_points_per_object=_env_int(
                "TORC_FRANKA_SINGULARITY_POINTS_PER_OBJECT",
                cls.singularity_points_per_object,
            ),
            proxy_half_span_m=_env_float("TORC_FRANKA_PROXY_HALF_SPAN_M", cls.proxy_half_span_m),
            proxy_radius_m=_env_float("TORC_FRANKA_PROXY_RADIUS_M", cls.proxy_radius_m),
            proxy_sphere_count=_env_int("TORC_FRANKA_PROXY_SPHERE_COUNT", cls.proxy_sphere_count),
            proxy_min_points=_env_int("TORC_FRANKA_PROXY_MIN_POINTS", cls.proxy_min_points),
            enclosure_min_cross_m=_env_float("TORC_FRANKA_ENCLOSURE_MIN_CROSS_M", cls.enclosure_min_cross_m),
            enclosure_max_center_m=_env_float("TORC_FRANKA_ENCLOSURE_MAX_CENTER_M", cls.enclosure_max_center_m),
            enclosure_min_points=_env_int("TORC_FRANKA_ENCLOSURE_MIN_POINTS", cls.enclosure_min_points),
            enclosure_score_weight=_env_float("TORC_FRANKA_ENCLOSURE_SCORE_WEIGHT", cls.enclosure_score_weight),
        )


class FrankaGraspAdapterScorer:
    """Franka-specific grasp geometry checks below the TORC planner boundary.

    TORC owns the pipeline stages and object ordering.  This class owns only
    Panda pad/TCP geometry used to refine or score a candidate already produced
    by the canonical grasp adapter.
    """

    def __init__(self, config=None):
        self.config = config or FrankaGraspGeometryConfig.from_env()
        self._object_cache_key = None
        self._object_cache = {}

    @staticmethod
    def _pose_matrix(pose):
        return pose_to_matrix(pose) if hasattr(pose, "position") else np.asarray(pose, dtype=np.float64)

    def object_points(self, visible_points, mask, object_id, max_points=None):
        if mask is None:
            return np.empty((0, 3), dtype=np.float64)
        points = np.asarray(visible_points, dtype=np.float64).reshape(-1, 3)
        labels = np.asarray(mask, dtype=np.uint32).reshape(-1)
        if len(points) == 0 or len(labels) != len(points):
            return np.empty((0, 3), dtype=np.float64)
        obj_id = int(object_id)
        if obj_id < 0:
            return np.empty((0, 3), dtype=np.float64)
        key = (id(visible_points), id(mask), points.shape, labels.shape)
        if key != self._object_cache_key:
            self._object_cache_key = key
            self._object_cache = {}
        if obj_id not in self._object_cache:
            obj_mask = (labels & (1 << obj_id)).astype(bool)
            self._object_cache[obj_id] = points[obj_mask]
        pts = self._object_cache[obj_id]
        if max_points is None or len(pts) <= max_points:
            return pts
        step = max(1, len(pts) // int(max_points))
        return pts[::step][: int(max_points)]

    @staticmethod
    def _pad_front_z():
        if RobotAdapter is not None:
            try:
                return float(RobotAdapter.derive_tcp_pad_front_m())
            except Exception:
                pass
        return 0.0044

    def scan_pose_offset_before_ik(self, pose_t, object_id, visible_points, mask):
        """Refine a Panda TCP pose in local closing/approach axes before IK."""
        cfg = self.config
        if not cfg.pre_filter_offset_scan:
            return pose_t, (0.0, 0.0, 0.0, np.nan)
        if mask is None:
            return pose_t, (0.0, 0.0, 0.0, np.nan)

        pts = self.object_points(
            visible_points,
            mask,
            object_id,
            max_points=cfg.scan_points_per_object,
        )
        if len(pts) < cfg.scan_min_object_points:
            return pose_t, (0.0, 0.0, 0.0, np.nan)

        dx_extent = cfg.scan_dx_extent_m
        dy_extent = cfg.scan_dy_extent_m
        dz_forward_extent = cfg.scan_forward_extent_m
        dz_backward_extent = cfg.scan_backward_extent_m
        step = cfg.scan_step_m

        z_target = self._pad_front_z() + cfg.scan_target_clearance_m
        opening_half = cfg.pad_opening_half_m
        half_x = cfg.pad_half_x_m

        R = pose_t[:3, :3]
        base_t = pose_t[:3, 3]
        local = (pts - base_t) @ R
        near_mask = (
            (np.abs(local[:, 0]) <= 0.09)
            & (np.abs(local[:, 1]) <= 0.075)
            & (local[:, 2] >= -0.035)
            & (local[:, 2] <= 0.16)
        )
        near = local[near_mask]
        if len(near) < 20:
            near = local
        x05, x95 = np.percentile(near[:, 0], [5, 95])
        y05, y95 = np.percentile(near[:, 1], [5, 95])
        x_center = 0.5 * (float(x05) + float(x95))
        y_center = 0.5 * (float(y05) + float(y95))

        z05_seed = float(np.percentile(near[:, 2], 5))
        seed_dx = float(np.clip(x_center, -dx_extent, dx_extent))
        seed_dy = float(np.clip(y_center, -dy_extent, dy_extent))
        seed_dz = float(np.clip(z05_seed - z_target, -dz_backward_extent, dz_forward_extent))
        if step > 1e-9:
            seed_dx = float(np.round(seed_dx / step) * step)
            seed_dy = float(np.round(seed_dy / step) * step)
            seed_dz = float(np.round(seed_dz / step) * step)
            dx_candidates = seed_dx + step * np.array([-3, -1, 0, 1, 3], dtype=np.float64)
            dy_candidates = seed_dy + step * np.array([-2, 0, 2], dtype=np.float64)
            dz_candidates = seed_dz + step * np.array([-2, -1, 0, 1, 2, 4, 6], dtype=np.float64)
            dx_candidates = np.unique(np.clip(dx_candidates, -dx_extent, dx_extent))
            dy_candidates = np.unique(np.clip(dy_candidates, -dy_extent, dy_extent))
            dz_candidates = np.unique(np.clip(dz_candidates, -dz_backward_extent, dz_forward_extent))
        else:
            dx_candidates = np.array([seed_dx])
            dy_candidates = np.array([seed_dy])
            dz_candidates = np.array([seed_dz])

        usable_half_opening = max(1e-6, opening_half - half_x)
        z_band = max(0.004, cfg.pad_half_z_m + abs(cfg.scan_target_clearance_m))
        z_front = self._pad_front_z()
        z_center = z_front - cfg.pad_half_z_m
        pad_centers = np.array(
            [
                [-(opening_half + cfg.pad_center_extra_x_m), 0.0, z_center],
                [opening_half + cfg.pad_center_extra_x_m, 0.0, z_center],
            ],
            dtype=np.float64,
        )
        pad_half_extents = (
            np.array([cfg.pad_half_x_m, cfg.pad_half_y_m, cfg.pad_half_z_m], dtype=np.float64)
            + cfg.pad_penetration_margin_m
        )
        penetration_check = local[
            (np.abs(local[:, 0]) <= opening_half + 0.025)
            & (np.abs(local[:, 1]) <= 0.04)
            & (local[:, 2] >= z_center - 0.035)
            & (local[:, 2] <= z_front + 0.035)
        ]
        best = None
        for dx_c in dx_candidates:
            for dy_c in dy_candidates:
                for dz_c in dz_candidates:
                    offset = np.array([dx_c, dy_c, dz_c], dtype=np.float64)
                    shifted = near - offset
                    sx05, sx95 = np.percentile(shifted[:, 0], [5, 95])
                    sy05, sy95 = np.percentile(shifted[:, 1], [5, 95])
                    sz05, sz25 = np.percentile(shifted[:, 2], [5, 25])
                    sx_center = 0.5 * (float(sx05) + float(sx95))
                    sy_center = 0.5 * (float(sy05) + float(sy95))
                    sx_span = float(sx95 - sx05)
                    sy_span = float(sy95 - sy05)

                    crosses_center = float(sx05) <= 0.0 <= float(sx95)
                    center_error = abs(sx_center) / usable_half_opening
                    y_center_error = abs(sy_center) / 0.04
                    span_error = max(0.0, sx_span - 2.0 * usable_half_opening) / usable_half_opening
                    y_span_error = max(0.0, sy_span - 0.06) / 0.06
                    z_error = abs(float(sz05) - z_target) / z_band
                    depth_reward = float(dz_c) / max(dz_forward_extent, 1e-6)
                    cross_penalty = 0.0 if crosses_center else 0.8

                    penetrates = False
                    if len(penetration_check):
                        shifted_check = penetration_check - offset
                        for pad_center in pad_centers:
                            if np.any(np.all(np.abs(shifted_check - pad_center) <= pad_half_extents, axis=1)):
                                penetrates = True
                                break
                    penetration_penalty = 10.0 if penetrates else 0.0

                    # z05 keeps the visible surface near the pad front; z25 rewards
                    # enough depth that the target is not merely grazing the fingertip edge.
                    depth_contact_error = max(0.0, float(sz25) - (z_target + 0.012)) / 0.02
                    score = (
                        2.2 * center_error
                        + 1.5 * z_error
                        + 1.0 * span_error
                        + 0.8 * y_center_error
                        + 0.6 * y_span_error
                        + 0.8 * depth_contact_error
                        + cross_penalty
                        + penetration_penalty
                        - 0.20 * depth_reward
                    )
                    if best is None or score < best[0]:
                        best = (float(score), float(dx_c), float(dy_c), float(dz_c))

        if best is None:
            return pose_t, (0.0, 0.0, 0.0, np.nan)
        score, dx, dy, dz = best

        adjusted = np.array(pose_t, dtype=np.float64, copy=True)
        adjusted[:3, 3] += dx * adjusted[:3, 0] + dy * adjusted[:3, 1] + dz * adjusted[:3, 2]
        return adjusted, (dx, dy, dz, float(score))

    def single_object_collides_with(self, poses_world, visible_points, mask):
        """Panda proxy for TORC's exactly-one-object singularity rule.

        The TORC rule remains unchanged: a candidate is singular when the
        gripper contact proxy intersects exactly one segmented object.  Only
        the contact proxy geometry is Panda-specific here.
        """
        points = np.asarray(visible_points, dtype=np.float64).reshape(-1, 3)
        labels = np.asarray(mask, dtype=np.uint32).reshape(-1)
        nonzero = labels != 0
        points = points[nonzero]
        labels = labels[nonzero]
        if len(points) == 0:
            return [set() for _ in range(len(poses_world))]

        cfg = self.config
        per_object_budget = cfg.singularity_points_per_object
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

        half_span = cfg.proxy_half_span_m
        radius = cfg.proxy_radius_m
        contact_depth = _env_float("TORC_FRANKA_PROXY_CONTACT_DEPTH_M", self._pad_front_z())
        sphere_count = cfg.proxy_sphere_count
        min_points = cfg.proxy_min_points
        sphere_xs = np.linspace(-half_span, half_span, max(1, sphere_count))
        sphere_centers = np.stack(
            [
                sphere_xs,
                np.zeros_like(sphere_xs),
                np.full_like(sphere_xs, contact_depth),
            ],
            axis=1,
        )

        collides_with = []
        for pose in poses_world:
            T = self._pose_matrix(pose)
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

    def pad_penetrates_object(self, poses_world, visible_points, mask, object_ids):
        """Reject candidates whose Panda pad boxes already contain target points."""
        cfg = self.config
        points = np.asarray(visible_points, dtype=np.float64).reshape(-1, 3)
        labels = np.asarray(mask, dtype=np.uint32).reshape(-1)
        object_ids = np.asarray(object_ids, dtype=np.int32).reshape(-1)
        if len(points) == 0 or len(labels) != len(points):
            return np.zeros(len(poses_world), dtype=bool)

        opening_half = cfg.pad_opening_half_m
        center_extra_x = cfg.pad_center_extra_x_m
        half_x = cfg.pad_half_x_m
        half_y = cfg.pad_half_y_m
        half_z = cfg.pad_half_z_m
        z_front = _env_float("TORC_FRANKA_PAD_FRONT_Z_M", self._pad_front_z())
        margin = cfg.pad_penetration_margin_m
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
            pts = self.object_points(points, labels, int(obj_id), max_points=cfg.singularity_points_per_object)
            if len(pts) == 0:
                penetrates.append(False)
                continue

            T = self._pose_matrix(pose)
            R = T[:3, :3]
            t = T[:3, 3]
            local = (pts - t) @ R
            in_any_pad = np.zeros(len(local), dtype=bool)
            for center in centers:
                in_box = np.all(np.abs(local - center) <= half_extents, axis=1)
                in_any_pad |= in_box
            penetrates.append(bool(np.any(in_any_pad)))
        return np.asarray(penetrates, dtype=bool)

    def object_is_enclosed_by_fingers(self, poses_world, visible_points, mask, object_ids):
        """Diagnostic: whether the target point cloud crosses Panda finger center."""
        cfg = self.config
        points = np.asarray(visible_points, dtype=np.float64).reshape(-1, 3)
        labels = np.asarray(mask, dtype=np.uint32).reshape(-1)
        object_ids = np.asarray(object_ids, dtype=np.int32).reshape(-1)
        if len(points) == 0 or len(labels) != len(points):
            return np.ones(len(poses_world), dtype=bool)

        min_cross = cfg.enclosure_min_cross_m
        max_center_abs = cfg.enclosure_max_center_m
        min_near_points = cfg.enclosure_min_points
        valid = []
        for pose, obj_id in zip(poses_world, object_ids):
            pts = self.object_points(points, labels, int(obj_id), max_points=cfg.singularity_points_per_object)
            if len(pts) < min_near_points:
                valid.append(True)
                continue

            T = self._pose_matrix(pose)
            R = T[:3, :3]
            t = T[:3, 3]
            local = (pts - t) @ R
            near_mask = (
                (np.abs(local[:, 0]) <= 0.10)
                & (np.abs(local[:, 1]) <= 0.08)
                & (local[:, 2] >= -0.04)
                & (local[:, 2] <= 0.16)
            )
            near = local[near_mask]
            if len(near) < min_near_points:
                valid.append(True)
                continue
            x05, x95 = np.percentile(near[:, 0], [5, 95])
            center = 0.5 * (float(x05) + float(x95))
            crosses_center = (float(x05) <= -min_cross) and (float(x95) >= min_cross)
            centered = abs(center) <= max_center_abs
            valid.append(bool(crosses_center and centered))
        return np.asarray(valid, dtype=bool)

    def enclosure_quality_scores(self, poses_world, visible_points, mask, object_ids):
        """Rank scene-safe grasps by how well the target sits inside Panda fingers."""
        cfg = self.config
        points = np.asarray(visible_points, dtype=np.float64).reshape(-1, 3)
        labels = np.asarray(mask, dtype=np.uint32).reshape(-1)
        object_ids = np.asarray(object_ids, dtype=np.int32).reshape(-1)
        if len(points) == 0 or len(labels) != len(points):
            return np.zeros(len(poses_world), dtype=np.float64)

        min_near_points = cfg.enclosure_min_points
        opening_half = cfg.pad_opening_half_m
        pad_half_x = cfg.pad_half_x_m
        usable_half_opening = max(1e-6, opening_half - pad_half_x)
        z_target = self._pad_front_z() + cfg.scan_target_clearance_m

        quality = []
        for pose, obj_id in zip(poses_world, object_ids):
            pts = self.object_points(points, labels, int(obj_id), max_points=cfg.singularity_points_per_object)
            if len(pts) < min_near_points:
                quality.append(0.0)
                continue

            T = self._pose_matrix(pose)
            R = T[:3, :3]
            t = T[:3, 3]
            local = (pts - t) @ R
            near_mask = (
                (np.abs(local[:, 0]) <= 0.10)
                & (np.abs(local[:, 1]) <= 0.08)
                & (local[:, 2] >= -0.04)
                & (local[:, 2] <= 0.16)
            )
            near = local[near_mask]
            if len(near) < min_near_points:
                near = local
            if len(near) < min_near_points:
                quality.append(0.0)
                continue

            x05, x95 = np.percentile(near[:, 0], [5, 95])
            z05 = float(np.percentile(near[:, 2], 5))
            x_center = 0.5 * (float(x05) + float(x95))
            x_span = max(1e-6, float(x95 - x05))
            crosses_center = 1.0 if float(x05) <= 0.0 <= float(x95) else 0.0
            center_score = max(0.0, 1.0 - abs(x_center) / usable_half_opening)
            span_score = max(0.0, 1.0 - max(0.0, x_span - 2.0 * usable_half_opening) / usable_half_opening)
            z_score = max(0.0, 1.0 - abs(z05 - z_target) / 0.025)
            quality.append(float(0.45 * center_score + 0.25 * span_score + 0.20 * z_score + 0.10 * crosses_center))
        return np.asarray(quality, dtype=np.float64)
