#!/usr/bin/env python3
"""Run Phase 4.3 through the real TORC pipeline with Franka selected.

This file is intentionally a thin launcher.  Object selection, perception,
dependency graph construction, grasp validation, and execution ordering must
come from the reproduced TORC pipeline under ``original_torc``.  Franka-specific
work is limited to the lower robot boundary selected by ``TORC_ROBOT=franka``:
robot model, grasp adapter, CuRobo config, and MuJoCo execution assets.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time


PROJECT_ROOT = Path("/mnt/ssd/ziyaochen/torc_franka_lift3d_pipeline_v2")
TORC_ROOT = PROJECT_ROOT / "original_torc/lab_vbnpm"
RUN_EXPERIMENT = TORC_ROOT / "experiments/run_experiment.py"
DEFAULT_SCENE = "tests/scenes/final/difficult_116.xml"
DEFAULT_TARGET = "obj_000070_0"
DEFAULT_TORC_PYTHON = Path("/home/ziyaochen/miniconda3/envs/ros_env/bin/python")
DEFAULT_CONDA_PREFIX = Path("/home/ziyaochen/miniconda3/envs/ros_env")
DEFAULT_CUROBO_SRC = Path("/home/ziyaochen/curobo_v0_7_8_torc/src")
OUT_MP4 = PROJECT_ROOT / "franka_planning_full.mp4"
MANIFEST = PROJECT_ROOT / "phase4_artifacts/phase4_3_planning_replacement_manifest.json"


def main() -> int:
    scene = os.environ.get("TORC_SCENE_REL", DEFAULT_SCENE)
    target = os.environ.get("TORC_TARGET_OBJECT", DEFAULT_TARGET)
    method = os.environ.get("TORC_METHOD", "dg_only")
    pick_limit = os.environ.get("TORC_PICK_LIMIT", "1")
    run_dir = PROJECT_ROOT / "phase4_artifacts" / f"torc_franka_pipeline_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update(
        {
            "TORC_ROBOT": "franka",
            "TORC_ROBOT_TYPE": "franka",
            "TORC_ROS_SETUP": env.get(
                "TORC_ROS_SETUP",
                "/home/ziyaochen/gc6d_lift3d_traj/ros_workspace/devel/setup.bash",
            ),
            "CONDA_PREFIX": str(DEFAULT_CONDA_PREFIX),
            "TORC_CONDA_PREFIX": str(DEFAULT_CONDA_PREFIX),
            "TORC_CUROBO_SRC": env.get("TORC_CUROBO_SRC", str(DEFAULT_CUROBO_SRC)),
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0"),
            "TORC_USE_CGN_ZMQ": env.get("TORC_USE_CGN_ZMQ", "1"),
            "TORC_GRASP_PLANNER": env.get("TORC_GRASP_PLANNER", "cgn"),
            "TORC_CGN_ZMQ_ADDRESS": env.get("TORC_CGN_ZMQ_ADDRESS", "tcp://127.0.0.1:6007"),
            "TORC_SCENE_PATH": scene,
            "TORC_SCENE_NAME": Path(scene).stem,
            "TORC_RENDER_EXECUTION_VIDEO": env.get("TORC_RENDER_EXECUTION_VIDEO", "1"),
            "TORC_RENDER_CAMERAS": env.get("TORC_RENDER_CAMERAS", "back_view"),
            "TORC_RENDER_STRIDE": env.get("TORC_RENDER_STRIDE", "3"),
            "TORC_RENDER_FPS": env.get("TORC_RENDER_FPS", "20"),
            "TORC_RENDER_WIDTH": env.get("TORC_RENDER_WIDTH", "1280"),
            "TORC_RENDER_HEIGHT": env.get("TORC_RENDER_HEIGHT", "720"),
        }
    )
    env["PATH"] = f"{env['TORC_CONDA_PREFIX']}/bin:" + env.get("PATH", "")
    env["PYTHONPATH"] = f"{env['TORC_CUROBO_SRC']}:{TORC_ROOT / 'scripts'}:{TORC_ROOT}:" + env.get("PYTHONPATH", "")
    env["ROS_PACKAGE_PATH"] = f"{TORC_ROOT.parent}:" + env.get("ROS_PACKAGE_PATH", "")
    torc_python = os.environ.get("TORC_PYTHON", str(DEFAULT_TORC_PYTHON))
    cmd = [
        torc_python,
        str(RUN_EXPERIMENT),
        "experiment",
        "--scene",
        scene,
        "--target-object",
        target,
        "--method",
        method,
        "--headless",
        "--server",
        "--pick-limit",
        str(pick_limit),
        "--base-dir",
        str(run_dir),
        "--mj-pickle",
    ]
    proc = subprocess.run(cmd, cwd=str(TORC_ROOT), env=env, text=True, capture_output=True)
    manifest = {
        "phase": "4.3",
        "entrypoint": "original_torc_pipeline_with_franka_robot_selector",
        "scene": scene,
        "target_object": target,
        "method": method,
        "pick_limit": int(pick_limit),
        "run_dir": str(run_dir),
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "robot_boundary": {
            "TORC_ROBOT": env["TORC_ROBOT"],
            "TORC_GRASP_PLANNER": env["TORC_GRASP_PLANNER"],
            "TORC_USE_CGN_ZMQ": env["TORC_USE_CGN_ZMQ"],
            "TORC_ROS_SETUP": env["TORC_ROS_SETUP"],
            "TORC_CONDA_PREFIX": env["TORC_CONDA_PREFIX"],
            "TORC_CUROBO_SRC": env["TORC_CUROBO_SRC"],
            "ROS_PACKAGE_PATH_prefix": str(TORC_ROOT.parent),
            "PYTHONPATH_prefix": [str(TORC_ROOT / "scripts"), str(TORC_ROOT)],
            "render_execution_video": env["TORC_RENDER_EXECUTION_VIDEO"],
            "grasp_reconstruction": "CGN infer_lowlevel -> CanonicalGrasp -> RobotAdapter -> TORC Pose",
            "forbidden": "no local object-selection or DepGraph reimplementation in phase4 script",
        },
    }
    videos = sorted(run_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if videos:
        shutil.copy2(videos[-1], OUT_MP4)
        manifest["output_video"] = str(OUT_MP4)
        manifest["source_video"] = str(videos[-1])
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
