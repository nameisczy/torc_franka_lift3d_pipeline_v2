# Dense Render Video Source Audit

Primary original dense video implementation: `/home/ziyaochen/gc6d_lift3d_traj/ros_workspace/src/lab_vbnpm/scripts/execution_scene/execution_node.py`

Original class/function: `DenseExecutionVideoRecorder.setup/capture/close`

Launch/config path: `/home/ziyaochen/gc6d_lift3d_traj/ros_workspace/src/lab_vbnpm/launch/control_server.launch` and `/home/ziyaochen/gc6d_lift3d_traj/ros_workspace/src/lab_vbnpm/ros_configs/control_sim_server.yaml`

Reference Franka postprocessor inspected: `/home/ziyaochen/torc_franka_lift3d_repro/scripts/franka_level2_close_lift_rerender_good_cameras.py`

1. Which original script/function generates dense videos?
   `DenseExecutionVideoRecorder` in `execution_scene/execution_node.py`.

2. What inputs does it expect?
   A loaded MuJoCo model/data, experiment output directory, named cameras, stride/fps/width/height.

3. How does it replay trajectory qpos?
   Original runtime advances `do_traj()` plus `mj_step()` and captures current `data`. For this D3 proof we use saved qpos trace replay with `mj_forward` only, following the older Franka rerender postprocessor boundary.

4. How does it choose cameras?
   `TORC_RENDER_CAMERAS` comma list, resolved with `mj_name2id`. D3 replaces this with accepted D-series cameras: `external_oblique`, `external_side`, `external_top`, `external_target_close`.

5. How does it render MuJoCo frames?
   One `mujoco.Renderer` per camera, `renderer.update_scene(data, camera=...)`, then `renderer.render()`.

6. How does it write mp4/gif/images?
   Original uses `cv2.VideoWriter(..., mp4v, fps, size)`. D3 writes PNG frames and encodes MP4 with `ffmpeg` when present, otherwise `imageio-ffmpeg`.

7. Which parts are robot-specific and must be replaced for Franka?
   Motoman XML, Motoman qpos mapping, Motoman camera names, and runtime execution step path.

8. Which parts can be reused unchanged?
   Per-camera frame rendering, dense frame capture metadata, MP4 writer pattern, frame validation, and saved-state replay claim boundary.

9. What camera changes are needed?
   Main video becomes `external_oblique`; side-clearance uses `external_side`; top-down path uses `external_top`; close-up uses `external_target_close`.
