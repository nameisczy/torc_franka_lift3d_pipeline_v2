# Franka Grasp Contract Audit

Scope: `original_torc/lab_vbnpm/scripts/grasp_planner/curobo_grasp_planner.py`

This audit marks robot-dependent semantics in `get_ik_grasps` and
`validate_grasps`.  TORC still owns perception, Dependency Graph ordering,
CGN invocation, stage order, scene collision, and pregrasp search.  Franka
logic is allowed only at explicit robot geometry boundaries.

## Current Boundary

- TORC pipeline stages remain in `curobo_grasp_planner.py`.
- Franka pad/TCP candidate refinement and scoring live in
  `original_torc/lab_vbnpm/scripts/robot_interface/franka_grasp_adapter.py`.
- Franka grasp geometry defaults live in `FrankaGraspGeometryConfig`; environment
  overrides are read only when that config is built.
- `FrankaGraspAdapterScorer` is initialized once at
  `curobo_grasp_planner.py:510`.

## `get_ik_grasps` Contract Points

| Lines | Code area | Dependency | Status | Rule |
| --- | --- | --- | --- | --- |
| 832-852 | Target point conversion and optional outlier filtering | Robot-independent perception data | Keep TORC | No Franka logic here. |
| 854-872 | Workspace bottom plane sampled into `visible_points` and `mask` | Workspace/table geometry, not gripper-specific | Keep TORC | May depend on scene calibration, not robot hand geometry. |
| 883-918 | Grasp service request formats | Grasp backend-specific | Keep TORC | CGN input remains full visible points plus mask. |
| 931-1008 | Ground-truth grasp branch | Strong legacy/TORC grasp-frame semantics | Isolate if used | Contains axis swap at 974-982 and approach offsets at 984-998. This is not the active CGN-ZMQ Franka path. |
| 1009-1017 | CGN-ZMQ raw grasp import | Robot-independent raw grasp source | Keep TORC | Raw CGN poses/scores/object ids must not be changed here. |
| 1024-1040 | Legacy ROS CGN service | Backend-specific, not active current path | Keep TORC | Do not add Franka geometry here. |
| 1042-1077 | Per-object legacy service loop | Backend/object-mask specific | Keep TORC | Object ids and masks must match TORC segmentation. |
| 1083-1092 | Empty grasp guard and raw grasp ids log | Robot-independent | Keep TORC | No Franka filtering before this. |
| 1062-1093 | Capture debug matrices and labels | Debug only | Completed | Labels are robot-neutral: `T_world_robot_tcp_*`. |
| 1082-1086 | `pose_t = pose_to_matrix`, `approach_t = pose_t[:3, 2]`, `pose_t += hand_depth * approach` | Robot-specific pose/frame contract | Franka-adapted | For Franka CGN-ZMQ, `hand_depth` is set to 0.0; legacy TORC hand-depth remains only outside Franka. |
| 1087-1090 | `FrankaGraspAdapterScorer.scan_pose_offset_before_ik` | Franka pad/TCP geometry | Extracted | This is the explicit Franka candidate refinement boundary before IK. |
| 1095 | `self.ik_solver.ik(self._pose_for_tracik(pose_t))` | Robot-specific IK frame/base contract | Franka-adapted elsewhere | Must consume the same TCP/base frame as CuRobo and MuJoCo. |
| 1107-1122 | IK stage logging and scan offset logging | Debug only | Keep | Does not change candidate order. |
| 1152-1205 | Visualization sorting by score | Display/order only | Keep TORC | No Franka filtering here. |

## `validate_grasps` Contract Points

| Lines | Code area | Dependency | Status | Rule |
| --- | --- | --- | --- | --- |
| 1230-1238 | Backup scores/source indices | Robot-independent bookkeeping | Keep TORC | No geometry logic. |
| 1240-1244 | `set_collision_scene(visible_points)` for target collision | Perception/collision scene | Keep TORC | Scene comes from TORC perception. |
| 1246-1251 | EE collision link toggle for target collision | Robot-specific link semantics | Franka-adapted | Franka enables EE links here because Panda pad geometry must be checked against target points. |
| 1253 | `get_joint_tensor_from_list(ik_joints)` | Joint ordering | Franka-adapted elsewhere | Must match Franka CuRobo YAML and MuJoCo joint order. |
| 1255-1258 | Collision distance and `d_world <= 0.01`, `d_self <= 0` | Collision tolerance | Keep TORC unless measured | The `0.01m` target tolerance is legacy TORC behavior; changing it changes TORC semantics. |
| 1279-1294 | Exactly-one-object Franka branch | Robot-specific contact proxy | Extracted | TORC rule is preserved; Panda proxy lives in `FrankaGraspAdapterScorer.single_object_collides_with`. |
| 1295-1324 | Legacy `attach_extra_spheres` object singularity | Robotiq/TORC proxy geometry | Keep for non-Franka only | Do not use for Franka without explicit geometry mapping. |
| 1325-1338 | `len(s) == 1` singularity rule and logging | TORC object-selection semantics | Keep TORC | Franka may change proxy geometry, not the exactly-one-object rule. |
| 1339-1358 | `FrankaGraspAdapterScorer.pad_penetrates_object` and enclosure diagnostic | Franka pad geometry | Extracted | Pad penetration remains a hard physical feasibility check; enclosure diagnostic does not reject. |
| 1359-1378 | Apply singularity valid mask | TORC stage semantics | Keep TORC | Only the Franka pad penetration bit is robot-specific. |
| 1380-1410 | Scene collision hard filter with `collision_points` | TORC scene/world collision | Keep TORC | This must run before Franka enclosure scoring. |
| 1412-1438 | `FrankaGraspAdapterScorer.enclosure_quality_scores` | Franka gripper scoring | Extracted | Score-only rerank after scene collision; it must not reject candidates. |
| 1446-1455 | `find_nearest_grasp_retraction` | Pose local approach axis, IK, collision | Robot-specific contract risk | It assumes local Z is approach and uses current IK frame. Keep stage, but verify axes whenever TCP contract changes. |
| 1457-1510 | Return validated grasps/pregrasps/scores | TORC downstream API | Keep TORC | Output shape/order must remain unchanged. |

## Extracted Franka Module

`original_torc/lab_vbnpm/scripts/robot_interface/franka_grasp_adapter.py`

| Lines | Function | Responsibility |
| --- | --- | --- |
| 13-83 | `FrankaGraspGeometryConfig` | Central default config for Franka grasp geometry and scoring. |
| 104-115 | `object_points` | Decode TORC bitmask labels into target object point clouds. |
| 117-124 | `_pad_front_z` | Derive current Panda pad front from the active Franka adapter/MuJoCo asset. |
| 126-183 | `scan_pose_offset_before_ik` | Local X/Z TCP candidate refinement before IK. |
| 185-243 | `single_object_collides_with` | Panda proxy for TORC's exactly-one-object singularity rule. |
| 245-290 | `pad_penetrates_object` | Hard reject candidates whose Panda pad boxes already contain target points. |
| 292-330 | `object_is_enclosed_by_fingers` | Diagnostic only; does not reject. |
| 332-379 | `enclosure_quality_scores` | Post-scene-collision score bonus for grasps with better Panda finger enclosure. |

## Remaining Cleanup Items

- Keep `scene collision -> enclosure score -> pregrasp` order fixed unless a
  separate audit proves TORC did otherwise.
