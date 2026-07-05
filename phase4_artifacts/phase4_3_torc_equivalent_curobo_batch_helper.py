
import json
import sys
from pathlib import Path

import numpy as np
import yaml

payload = json.loads(Path(sys.argv[1]).read_text())
result = {
    "type": payload["type"],
    "policy": payload["policy"],
    "batch_tag": payload["batch_tag"],
    "import_success": False,
    "motiongen_constructed": False,
    "results": [],
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
    plan_cfg = MotionGenPlanConfig(max_attempts=2, enable_graph_attempt=0)
    for goal in payload["goals"]:
        row = {
            "type": payload["type"],
            "policy": payload["policy"],
            "candidate_index": int(goal["candidate_index"]),
            "import_success": True,
            "motiongen_constructed": True,
            "planning_success": False,
        }
        try:
            pose = Pose.from_list(goal["goal_pose_base_wxyz"], tensor_args=tensor_args)
            ik = mg.ik_solver.solve_single(
                pose,
                retract_config=start.position.clone(),
                seed_config=start.position.view(1, 1, -1).clone(),
                return_seeds=1,
                num_seeds=mg.ik_solver.num_seeds,
            )
            row["ik_success"] = bool(ik.success.reshape(-1)[0].item())
            row["ik_position_error"] = float(ik.position_error.reshape(-1)[0].item())
            if row["ik_success"]:
                plan = mg.plan_single(start, pose, plan_cfg)
                success = bool(plan.success.item()) if hasattr(plan.success, "item") else bool(plan.success)
                row["planning_success"] = success
                row["motiongen_status"] = str(getattr(plan, "status", None))
                if success and hasattr(plan, "optimized_plan"):
                    q = plan.optimized_plan.position.detach().cpu().numpy()
                    q = q.reshape(q.shape[0], q.shape[-1])
                    row["trajectory_qpos"] = [[float(x) for x in state] for state in q]
                    row["trajectory_length"] = len(row["trajectory_qpos"])
                elif not success:
                    row["failure_reason"] = str(getattr(plan, "status", "planning_failed"))
            else:
                row["failure_reason"] = "ik_failed_before_motiongen_plan"
        except Exception as exc:
            import traceback
            row["failure_reason"] = repr(exc)
            row["traceback"] = traceback.format_exc()
        result["results"].append(row)
except Exception as exc:
    import traceback
    result["externally_blocked_with_exact_action"] = {
        "evidence_level": "externally_blocked_with_exact_action",
        "error": repr(exc),
        "traceback": traceback.format_exc(),
        "exact_action": "Run batch helper with visible CUDA and current-project Franka CuRobo YAML.",
    }
print(json.dumps(result))
