# Phase 1 TORC Full Repro Test - Success

Date: 2026-07-04

## Result

Original TORC baseline completed with:

```text
Received result: retrieve_success
```

Target:

```text
scene: tests/scenes/final/difficult_116.xml
target_object: obj_000070_0
method: dg_only
```

Target retrieval occurred on pick 5:

```text
Pick Count,5
Grasp Choice (Seg ID),0
Object Name,obj_000070_0
Grasp Success,True
Grasped,"obj_000070_0"
Grasp Result,finished
Retrieved Target Object,True
```

## Baseline Video

Output:

```text
/mnt/ssd/ziyaochen/torc_franka_lift3d_pipeline_v2/torc_repro_baseline.mp4
```

Validation:

```text
opened True
frames 49228
fps 20.0
width 1280
height 720
```

The video is the dense `back_view` MuJoCo render from the full run.

## Command Path

The initial failed attempt was resolved by following the previously validated clean/repro run reports:

- `/home/ziyaochen/torc_franka_lift3d_repro/reports/TORC_CGN_ZMQ_SCENE116_AND_SMOKE_RUNS.md`
- `/home/ziyaochen/torc_franka_lift3d_repro/reports/FRANKA_F2_6B_CGN_RECOVERY_AND_SELECTED_GRASP_CAPTURE.md`

The successful run used the report-prescribed compatibility environment:

```bash
TORC_RENDER_EXECUTION_VIDEO=1
TORC_RENDER_CAMERAS=back_view
TORC_RENDER_STRIDE=2
TORC_RENDER_FPS=20
TORC_RENDER_WIDTH=1280
TORC_RENDER_HEIGHT=720
/home/ziyaochen/torc_franka_lift3d_repro/scripts/run_torc_cgn_zmq_experiment.sh \
  /home/ziyaochen/torc_franka_lift3d_repro/logs/phase1_torc_cgn_zmq_20260704_125224.log \
  tests/scenes/final/difficult_116.xml \
  obj_000070_0 \
  /home/ziyaochen/phase1_runs_cgn \
  15 \
  --mj-pickle
```

That wrapper activates `ros_env`, sources the ROS workspace, sets:

```text
TORC_USE_CGN_ZMQ=1
PYTHONPATH=/home/ziyaochen/curobo_v0_7_8_torc/src:$PYTHONPATH
GC6D_ROOT=/mnt/ssd/ziyaochen/GraspClutter6D
CUDA_VISIBLE_DEVICES=1
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

This resolves the earlier `curobo.types.robot` import failure and routes CGN through the already validated ZMQ bridge instead of the failing Docker service.

## Evidence Artifacts

Copied artifacts:

```text
/mnt/ssd/ziyaochen/torc_franka_lift3d_pipeline_v2/phase1_baseline_artifacts/output.csv
/mnt/ssd/ziyaochen/torc_franka_lift3d_pipeline_v2/phase1_baseline_artifacts/curobo_control_stage_log.txt
/mnt/ssd/ziyaochen/torc_franka_lift3d_pipeline_v2/phase1_baseline_artifacts/info_experiment.json
/mnt/ssd/ziyaochen/torc_franka_lift3d_pipeline_v2/phase1_baseline_artifacts/info_curobo_control.json
/mnt/ssd/ziyaochen/torc_franka_lift3d_pipeline_v2/phase1_baseline_artifacts/phase1_torc_cgn_zmq_20260704_125224.log
/mnt/ssd/ziyaochen/torc_franka_lift3d_pipeline_v2/phase1_baseline_artifacts/phase1_cgn_zmq_20260704_125029.log
```

Artifact sizes / line counts:

```text
output.csv: 84 lines
curobo_control_stage_log.txt: 2198 lines
phase1_cgn_zmq_20260704_125029.log: 186 lines
```

## Pipeline Evidence

Stage log confirms the pipeline reached:

```text
GraspPlanner init done
open_loop_pick_or_place done
pick loop exiting | retrieve_success
run(args) returning | result=retrieve_success
```

CGN ZMQ log confirms 5 real CGN inference requests:

```text
13:02:01 infer -> 2260 grasps
13:03:46 infer -> 2910 grasps
13:07:22 infer -> 2296 grasps
13:08:43 infer -> 2742 grasps
13:10:13 infer -> 2500 grasps
```

## Previous Failure Superseded

`PHASE1_TORC_REPRO_FAILURE_REPORT.md` is superseded by this successful rerun. The earlier blockers were:

```text
curobo.types.robot import failure
Contact-GraspNet Docker startup failure
```

Both were resolved by using the clean/repro validated environment:

```text
/home/ziyaochen/curobo_v0_7_8_torc/src
TORC_USE_CGN_ZMQ=1
CGN ZMQ server on tcp://127.0.0.1:6007
```

## Stop Condition

Phase 1 completed successfully. No Phase 1.5 audit or downstream replacement work was run.
