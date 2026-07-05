
import json
import sys
from pathlib import Path

import numpy as np
import yaml

payload = json.loads(Path(sys.argv[1]).read_text())
result = {
    "type": payload["type"],
    "policy": payload["policy"],
    "candidate_index": payload["candidate_index"],
    "import_success": False,
    "motiongen_constructed": False,
    "planning_success": False,
}
try:
    import torch
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.types import Mesh, WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

    result["torch_cuda_available"] = bool(torch.cuda.is_available())
    result["import_success"] = True
    tensor_args = TensorDeviceType()
    robot_cfg_full = yaml.safe_load(Path(payload["current_project_franka_yaml"]).read_text())["robot_cfg"]
    robot_cfg = {"kinematics": robot_cfg_full["kinematics"]}
    points = np.load(payload["points_base_npy"]).astype(np.float32)
    scene_mesh = Mesh.from_pointcloud(points, 0.01, "world")
    world = WorldConfig(mesh=[scene_mesh])
    cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world,
        tensor_args,
        use_cuda_graph=False,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"mesh": 2, "obb": 8},
    )
    mg = MotionGen(cfg)
    result["motiongen_constructed"] = True
    start = JointState.from_position(
        torch.tensor(payload["start_qpos"], dtype=torch.float32, device=tensor_args.device).view(1, -1),
        joint_names=robot_cfg["kinematics"]["cspace"]["joint_names"],
    )
    pose = Pose.from_list(payload["goal_pose_base_wxyz"], tensor_args=tensor_args)
    plan_cfg = MotionGenPlanConfig(max_attempts=2, enable_graph_attempt=0)
    ik = mg.ik_solver.solve_single(
        pose,
        retract_config=start.position.clone(),
        seed_config=start.position.view(1, 1, -1).clone(),
        return_seeds=1,
        num_seeds=mg.ik_solver.num_seeds,
    )
    result["ik_success"] = bool(ik.success.reshape(-1)[0].item())
    result["ik_position_error"] = float(ik.position_error.reshape(-1)[0].item())
    if result["ik_success"]:
        plan = mg.plan_single(start, pose, plan_cfg)
        success = bool(plan.success.item()) if hasattr(plan.success, "item") else bool(plan.success)
        result["planning_success"] = success
        result["motiongen_status"] = str(getattr(plan, "status", None))
        if success and hasattr(plan, "optimized_plan"):
            q = plan.optimized_plan.position.detach().cpu().numpy()
            q = q.reshape(q.shape[0], q.shape[-1])
            result["trajectory_qpos"] = [[float(x) for x in row] for row in q]
            result["trajectory_length"] = len(result["trajectory_qpos"])
        elif not success:
            result["failure_reason"] = str(getattr(plan, "status", "planning_failed"))
    else:
        result["failure_reason"] = "ik_failed_before_motiongen_plan"
except Exception as exc:
    import traceback
    result["externally_blocked_with_exact_action"] = {
        "evidence_level": "externally_blocked_with_exact_action",
        "error": repr(exc),
        "traceback": traceback.format_exc(),
        "exact_action": "Run helper with visible CUDA and current-project Franka CuRobo YAML.",
    }
print(json.dumps(result))
