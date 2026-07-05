# Phase 1 TORC Full Repro Test

Date: 2026-07-04

## Command

Ran original TORC from the original ROS workspace without modifying source files:

```bash
source /home/ziyaochen/miniconda3/etc/profile.d/conda.sh
conda activate ros_env
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
TORC_RENDER_EXECUTION_VIDEO=1 \
TORC_RENDER_CAMERAS=back_view \
TORC_RENDER_STRIDE=2 \
TORC_RENDER_FPS=20 \
TORC_RENDER_WIDTH=1280 \
TORC_RENDER_HEIGHT=720 \
./run_experiment.sh experiment \
  --scene tests/scenes/final/structured_198.xml \
  --target-object obj_000041_0 \
  --method dg_only \
  --headless \
  --server \
  --pick-limit 1 \
  --mj-pickle \
  --base-dir /home/ziyaochen/phase1_runs
```

## Output Video

Target output:

```text
/mnt/ssd/ziyaochen/torc_franka_lift3d_pipeline_v2/torc_repro_baseline.mp4
```

Video validation:

```text
opened True
frames 3142
fps 20.0
width 1280
height 720
```

Important: this is a dense rendered reset / pre-execution video. The full grasp execution trace did not complete because the original pipeline failed before planner initialization finished.

## Run Directory

```text
/home/ziyaochen/phase1_runs/exp_2026-07-04_12-40-17__structured_198_obj_000041_0_dg_only
```

Key files:

```text
curobo_control_stage_log.txt
info_curobo_control.json
info_experiment.json
output.csv
videos/full_robot_continuous_back_view.mp4
```

## Exact Failure Layer

Primary blocking failure:

```text
Layer: grasp / motion-planner Python dependency initialization
Module: scripts/grasp_planner/curobo_grasp_planner.py
Import: from curobo.types.robot import JointState
Error: ModuleNotFoundError: No module named 'curobo.types.robot'; 'curobo.types' is not a package
```

Traceback observed in `control_sim_server:4`:

```text
File ".../scripts/task_planner/curobo_control.py", line 448, in run
    from grasp_planner.curobo_grasp_planner import GraspPlanner
File ".../scripts/grasp_planner/curobo_grasp_planner.py", line 14, in <module>
    from curobo.types.robot import JointState
ModuleNotFoundError: No module named 'curobo.types.robot'; 'curobo.types' is not a package
```

Secondary service failure observed:

```text
Layer: Contact-GraspNet ROS service startup
Module: contact_graspnet_ros/run_container.py
Command: docker image ls --format json
Error: subprocess.CalledProcessError: returned non-zero exit status 1
```

## Pipeline Progress Before Failure

Succeeded:

```text
ROS master started
motoman_node server started on tcp://*:5858
curobo_control server initially started on tcp://*:5757
scene reset request sent to motoman node
MuJoCo scene loaded
dense renderer initialized and produced a valid video
```

Failed before:

```text
GraspPlanner import completed
Contact-GraspNet grasp generation
CuRobo motion planning
trajectory execution
target retrieval / drop success detection
```

## Stop Condition

Phase 1 stopped here per instructions. No Franka replacement or downstream audit phase was run.
