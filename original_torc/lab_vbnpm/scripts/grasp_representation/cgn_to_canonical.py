import numpy as np

try:
    from grasp_representation.canonical_grasp import CanonicalGrasp
except ImportError:
    from canonical_grasp import CanonicalGrasp


def _unit(vec):
    vec = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("direction must be nonzero")
    return vec / norm


def make_frame(approach_direction, grasp_axis):
    z_axis = _unit(approach_direction)
    y_seed = _unit(grasp_axis)
    y_axis = y_seed - z_axis * float(np.dot(y_seed, z_axis))
    y_axis = _unit(y_axis)
    x_axis = _unit(np.cross(y_axis, z_axis))
    y_axis = _unit(np.cross(z_axis, x_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def cgn_to_canonical_grasp(raw_grasp, object_id=0, score=None):
    raw = dict(raw_grasp)
    contact = np.asarray(raw.get("contact_point", raw.get("translation")), dtype=float).reshape(3)
    approach = _unit(raw.get("approach_direction", raw.get("approach")))
    axis = _unit(raw.get("base_or_closing_direction", raw.get("grasp_axis")))
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = make_frame(approach, axis)
    pose[:3, 3] = contact
    width = float(raw.get("opening_width", raw.get("opening_m", raw.get("width", 0.08))))
    confidence = float(raw.get("score", 1.0 if score is None else score))
    return CanonicalGrasp(
        contact_pose=pose,
        approach_direction=approach,
        grasp_axis=axis,
        opening_width=width,
        score=confidence,
        object_id=int(raw.get("object_id", object_id)),
    ).validate()


def poses_to_canonical(poses, object_id=0):
    grasps = []
    for idx, pose in enumerate(np.asarray(poses, dtype=float)):
        item = {
            "translation": pose[:3, 3],
            "approach": pose[:3, 2],
            "grasp_axis": pose[:3, 1],
            "score": 1.0 / (1.0 + idx),
            "object_id": object_id,
        }
        grasps.append(cgn_to_canonical_grasp(item, object_id=object_id))
    return grasps
