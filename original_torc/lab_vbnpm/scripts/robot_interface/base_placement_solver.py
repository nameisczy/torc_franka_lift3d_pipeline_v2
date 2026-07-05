from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BasePlacementResult:
    base_pose_world_m: tuple
    ik_feasible_fraction: float
    collision_free_fraction: float
    workspace_coverage: float
    table_top: float


def solve_base_placement(workspace_points, table_top=0.8, candidate_xy=((0.0, 0.0),), base_z=0.86):
    points = np.asarray(workspace_points, dtype=float).reshape(-1, 3)
    if points.size == 0:
        return BasePlacementResult((0.0, 0.0, float(base_z)), 0.0, 0.0, 0.0, float(table_top))
    best = None
    for x, y in candidate_xy:
        dist = np.linalg.norm(points[:, :2] - np.array([x, y], dtype=float), axis=1)
        reachable = dist < 0.85
        above_table = points[:, 2] >= float(table_top)
        ik_feasible_fraction = float(np.mean(reachable))
        collision_free_fraction = float(np.mean(above_table))
        workspace_coverage = float(np.mean(reachable & above_table))
        result = BasePlacementResult(
            (float(x), float(y), float(base_z)),
            ik_feasible_fraction,
            collision_free_fraction,
            workspace_coverage,
            float(table_top),
        )
        if best is None or result.workspace_coverage > best.workspace_coverage:
            best = result
    return best
