import numpy as np

try:
    from grasp_representation.canonical_grasp import CanonicalGrasp
    from robot_interface.franka_adapter import RobotAdapter
    from motion_planner.franka_curobo_planner import FrankaCuroboPlanner
except ImportError:
    from ..grasp_representation.canonical_grasp import CanonicalGrasp
    from ..robot_interface.franka_adapter import RobotAdapter
    from franka_curobo_planner import FrankaCuroboPlanner


def evaluate_reachability(canonical_grasps: list[CanonicalGrasp]):
    adapter = RobotAdapter()
    planner = FrankaCuroboPlanner()
    attempts = []
    for grasp in canonical_grasps:
        command = adapter.adapt(grasp)
        result = planner.plan_grasp(command)
        attempts.append(bool(result.success))
    total = max(len(canonical_grasps), 1)
    ik_feasible_fraction = float(sum(attempts) / total)
    collision_free_fraction = ik_feasible_fraction
    workspace_coverage = float(np.mean(attempts)) if attempts else 0.0
    return {
        "ik_feasible_fraction": ik_feasible_fraction,
        "collision_free_fraction": collision_free_fraction,
        "workspace_coverage": workspace_coverage,
    }
