import math
import numpy as np

try:
    from grasp_representation.canonical_grasp import CanonicalGrasp
except ImportError:
    from canonical_grasp import CanonicalGrasp


def finite_pose(item: CanonicalGrasp) -> bool:
    return bool(np.isfinite(np.asarray(item.contact_pose, dtype=float)).all())


def valid_width(item: CanonicalGrasp, min_width=0.001, max_width=0.12) -> bool:
    width = float(item.opening_width)
    return bool(math.isfinite(width) and min_width <= width <= max_width)


def valid_score(item: CanonicalGrasp, min_score=0.0) -> bool:
    score = float(item.score)
    return bool(math.isfinite(score) and score >= min_score)


def filter_canonical_grasps(items, min_score=0.0, min_width=0.001, max_width=0.12):
    kept = []
    for item in items:
        item.validate()
        if finite_pose(item) and valid_score(item, min_score) and valid_width(item, min_width, max_width):
            kept.append(item)
    return kept
