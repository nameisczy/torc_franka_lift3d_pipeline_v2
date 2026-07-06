from dataclasses import dataclass

import numpy as np

try:
    from grasp_representation.cgn_to_canonical import cgn_to_canonical_grasp
except ImportError:
    from cgn_to_canonical import cgn_to_canonical_grasp


@dataclass(frozen=True)
class LowLevelGrasp:
    contact_point_world: np.ndarray
    approach_dir_world: np.ndarray
    base_dir_world: np.ndarray
    contact_width: float
    gripper_opening: float
    score: float
    object_id: int
    source_index: int
    selection_index: int
    offset_bin_index: int = -1

    def validate(self):
        for name in ("contact_point_world", "approach_dir_world", "base_dir_world"):
            vec = np.asarray(getattr(self, name), dtype=float).reshape(3)
            if not np.all(np.isfinite(vec)):
                raise ValueError(f"{name} must be finite")
        for name in ("approach_dir_world", "base_dir_world"):
            norm = float(np.linalg.norm(getattr(self, name)))
            if norm <= 0.0 or not np.isfinite(norm):
                raise ValueError(f"{name} must be nonzero")
        if float(self.contact_width) <= 0.0:
            raise ValueError("contact_width must be positive")
        if float(self.gripper_opening) <= 0.0:
            raise ValueError("gripper_opening must be positive")
        return self


def lowlevel_grasps_from_arrays(arrays: dict, object_id: int | None = None) -> list[LowLevelGrasp]:
    contacts = np.asarray(arrays.get("contact_point_world", []), dtype=float).reshape(-1, 3)
    approaches = np.asarray(arrays.get("approach_dir_world", []), dtype=float).reshape(-1, 3)
    bases = np.asarray(arrays.get("base_dir_world", []), dtype=float).reshape(-1, 3)
    openings = np.asarray(arrays.get("gripper_opening", []), dtype=float).reshape(-1)
    widths = np.asarray(arrays.get("offset_bin_value", openings), dtype=float).reshape(-1)
    scores = np.asarray(arrays.get("score", np.ones(len(contacts))), dtype=float).reshape(-1)
    object_ids = np.asarray(arrays.get("object_id", np.zeros(len(contacts), dtype=int)), dtype=int).reshape(-1)
    source_indices = np.asarray(arrays.get("source_index", np.arange(len(contacts))), dtype=int).reshape(-1)
    selection_indices = np.asarray(arrays.get("selection_index", np.arange(len(contacts))), dtype=int).reshape(-1)
    offset_bin_indices = np.asarray(arrays.get("offset_bin_index", np.full(len(contacts), -1)), dtype=int).reshape(-1)
    lengths = {
        len(contacts),
        len(approaches),
        len(bases),
        len(openings),
        len(widths),
        len(scores),
        len(object_ids),
        len(source_indices),
        len(selection_indices),
        len(offset_bin_indices),
    }
    if len(lengths) != 1:
        raise ValueError(f"lowlevel CGN array length mismatch: {sorted(lengths)}")
    grasps = []
    for idx in range(len(contacts)):
        if object_id is not None and int(object_ids[idx]) not in (0, int(object_id)):
            continue
        grasps.append(
            LowLevelGrasp(
                contact_point_world=contacts[idx],
                approach_dir_world=approaches[idx],
                base_dir_world=bases[idx],
                contact_width=float(widths[idx]),
                gripper_opening=float(openings[idx]),
                score=float(scores[idx]),
                object_id=int(object_id if object_id is not None else object_ids[idx]),
                source_index=int(source_indices[idx]),
                selection_index=int(selection_indices[idx]),
                offset_bin_index=int(offset_bin_indices[idx]),
            ).validate()
        )
    return grasps


def pose_to_lowlevel_grasp(
    pose_world: np.ndarray,
    object_id: int,
    source_index: int = 0,
    selection_index: int = 0,
    gripper_opening: float = 0.06,
    score: float | None = None,
) -> LowLevelGrasp:
    raise RuntimeError(
        "Franka migration must not reconstruct lowlevel grasps from build_6d_grasp poses; "
        "use contact_point_world/approach_dir_world/base_dir_world from CGN infer_lowlevel instead."
    )
    pose = np.asarray(pose_world, dtype=float).reshape(4, 4)
    confidence = 1.0 / (1.0 + float(selection_index)) if score is None else float(score)
    return LowLevelGrasp(
        contact_point_world=np.array(pose[:3, 3], dtype=float, copy=True),
        approach_dir_world=np.array(pose[:3, 2], dtype=float, copy=True),
        base_dir_world=np.array(pose[:3, 1], dtype=float, copy=True),
        contact_width=float(gripper_opening),
        gripper_opening=float(gripper_opening),
        score=confidence,
        object_id=int(object_id),
        source_index=int(source_index),
        selection_index=int(selection_index),
    ).validate()


def lowlevel_to_canonical_grasp(grasp: LowLevelGrasp):
    grasp.validate()
    return cgn_to_canonical_grasp(
        {
            "contact_point": grasp.contact_point_world,
            "approach_direction": grasp.approach_dir_world,
            "base_or_closing_direction": grasp.base_dir_world,
            "opening_width": grasp.contact_width,
            "score": grasp.score,
            "object_id": grasp.object_id,
        },
        object_id=grasp.object_id,
        score=grasp.score,
    )


def poses_to_lowlevel_grasps(poses, object_id: int, opening_width: float = 0.06) -> list[LowLevelGrasp]:
    return [
        pose_to_lowlevel_grasp(
            pose_world=pose,
            object_id=object_id,
            source_index=idx,
            selection_index=idx,
            gripper_opening=opening_width,
        )
        for idx, pose in enumerate(np.asarray(poses, dtype=float))
    ]
