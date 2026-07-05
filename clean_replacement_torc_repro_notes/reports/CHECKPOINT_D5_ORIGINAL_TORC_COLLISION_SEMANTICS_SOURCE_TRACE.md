# Checkpoint D5 Original TORC Collision Semantics Source Trace

Evidence level: `proven`

Recovered behavior: original TORC pregrasp MotionGen uses static workspace geometry plus a single anonymous pointcloud mesh named `world` created from `all_pts`.

Named object-world obstacles in MotionGen pregrasp: `false`

Allowed-contact current-object semantics: `absent`

Current object handling: included anonymously in `all_pts` if points are present; no source-level current-object exclusion before `pose_motion_plan`.

Clutter handling: included anonymously in `all_pts` if points are present; no per-clutter named obstacle records.

Key evidence:

- `scripts/task_planner/curobo_open_loop.py:235-356`: constructs `all_pts` and `all_mask`.
- `scripts/task_planner/curobo_open_loop.py:755-807`: calls `planner.set_planning_scene(all_pts)` before `planner.pose_motion_plan(joint_state, p_grasp[ind])`.
- `scripts/motion_planner/curobo_planner.py:731-837`: converts point cloud to `Mesh.from_pointcloud(points, resolution, "world")` and updates MotionGen.
- `scripts/grasp_planner/curobo_grasp_planner.py:966-1130`: object masks and extra spheres are used for grasp filtering before MotionGen.
- `scripts/task_planner/dep_graph.py:386-620`: object relations and grasp blockers feed selection, not named MotionGen contact semantics.

Full JSON trace: `/mnt/ssd/ziyaochen/torc_franka_clean_results/checkpoint_d5_torc_equivalent_policy_scene20_20260630T115233Z/checkpoint_d5_original_torc_collision_semantics_source_trace.json`
