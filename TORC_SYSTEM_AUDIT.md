# TORC System Audit - Phase 1.5

Date: 2026-07-04

Workspace:

- `/mnt/ssd/ziyaochen/torc_franka_lift3d_pipeline_v2`

Audited source roots:

- `original_torc/lab_vbnpm`
- `clean_replacement_torc_repro_notes/reports`
- Phase 1 baseline artifacts under `phase1_baseline_artifacts`

Phase 1 status: original TORC baseline reproduced successfully. The successful run produced `retrieve_success` and `/mnt/ssd/ziyaochen/torc_franka_lift3d_pipeline_v2/torc_repro_baseline.mp4`.

## Executive Finding

Original TORC is not robot-neutral after perception. Robot assumptions enter at the robot abstraction layer, grasp postprocessing, IK filtering, CuRobo configuration, execution services, MuJoCo robot/gripper model, success detection, and visualization/rendering. The strongest hard couplings are:

- Motoman SDA10F + Robotiq 2F-85 naming and geometry.
- `base_link -> motoman_right_ee` kinematic contract.
- CGN pose postprocessing with TORC hand-depth and approach-axis conventions.
- CuRobo collision spheres and lock-joint assumptions from `robots/motoman/curobo/motoman.yml`.
- Execution and success detection based on `MotomanNode`, Robotiq geoms, `finger_joint`, and `/experiment_result`.

## System Decomposition

| Subsystem | Runtime role | Primary files | Robot dependency level |
|---|---|---|---|
| Perception | Capture RGB-D/segmentation, fuse point cloud, produce object masks and occlusion points. | `scripts/task_planner/curobo_open_loop.py:235-356`, `scripts/perception/perception_fast.py`, `scripts/execution_scene/execution_node.py:686-777` | Medium: cameras and workspace are selected by robot abstraction; raw point cloud logic is mostly robot-agnostic. |
| Grasp detection: Contact-GraspNet | External service returns grasp poses, scores, samples/object ids. Phase 1 used CGN ZMQ bridge. | `scripts/grasp_planner/curobo_grasp_planner.py:250-257`, Phase 1 report CGN ZMQ evidence | Medium: neural output is parallel-jaw-ish, but pose construction contains gripper-depth/opening conventions before TORC receives it. |
| Grasp filtering | Score/object filtering, table/geometry filters, IK feasibility, collision filters, one-object grasp filtering, pregrasp generation. | `scripts/grasp_planner/curobo_grasp_planner.py:265-356`, `858-877`, `966-1158`; `scripts/grasp_planner/grasp_plotter.py:10-160` | High: Motoman EE, Robotiq dimensions, collision links, TracIK base/EE, and CuRobo robot world are embedded. |
| Grasp parameterization | Converts raw CGN grasp to Motoman EE goal by adding local approach displacement and uses Robotiq visual/contact dimensions. | `scripts/grasp_planner/curobo_grasp_planner.py:279-287`, `858-865`; `grasp_plotter.py:15-18`, `93-123`; clean report `OLD_ROBOT_GRIPPER_MEASURED_ASSUMPTIONS.md` | High: `hand_depth=0.11 m`, CGN default depth `0.1034 m`, Robotiq width/opening assumptions. |
| CuRobo planning | MotionGen pregrasp plan, static workspace obstacles, anonymous point-cloud obstacle mesh, Pink Cartesian IK for approach/lift. | `scripts/motion_planner/curobo_planner.py:116-163`, `728-829`; `scripts/task_planner/curobo_open_loop.py:806-836`, `914-959` | High: `motoman.yml`, `motoman.urdf`, `motoman_right_ee`, collision spheres, locked joints, joint order. |
| Execution interface | Calls `execute_trajectory`, `ee_control`, waits for robot/gripper status, opens before plan0 and closes before plan2. | `scripts/task_planner/eutils.py:15-19`, `61-137`; `scripts/task_planner/curobo_open_loop.py:1288-1310`; `scripts/execution_scene/motoman_interface.py:753-792` | High: Robotiq service name/width, `finger_joint`, Motoman interface. |
| MuJoCo simulation | Loads scene XML, publishes camera images, segmentation, TF, joint states, accepts trajectory and EE controls. | `scripts/execution_scene/execution_node.py:330-439`, `920-986`; `scripts/execution_scene/motoman_node.py:90-215`; `xmls/robot_*.xml`, `robots/motoman/*` | High: MJCF body/joint/actuator names, Robotiq subtree, Motoman joints and initial state. |
| Rendering system | Dense execution video and RViz/marker rendering of grasps, point clouds, trajectories, object state. | `scripts/execution_scene/execution_node.py:118-190`, `904-925`; `scripts/grasp_planner/grasp_plotter.py`; `launch/control_server.launch:5-25` | Medium-high: video cameras are scene/MuJoCo-camera dependent; grasp markers assume Robotiq-like geometry. |
| Success detection logic | `/experiment_result` returns success, dropped, grasping from MuJoCo contact and object pose state. | `scripts/task_planner/eutils.py:158-169`; `scripts/task_planner/curobo_open_loop.py:1323-1362`; `scripts/execution_scene/motoman_node.py:216-332`; `srv/ExperimentResult.srv` | High: success is contact between object and Robotiq gripper geoms; dropped threshold and object naming are simulator-specific. |

## Robot Dependency Origin Points

### 1. Robot Abstraction

Origin:

- `scripts/task_planner/motoman.py:18-130`
- `scripts/task_planner/robot_selector.py` selects robot class in `curobo_control.py:376-386`.

Embedded assumptions:

- Robot class is `MotomanSDA10F`.
- End effector is `motoman_right_ee`.
- Active group is `arm_right`.
- Cameras are `camera0`, `camera1` in sim and `zedm` in real mode.
- URDF/CuRobo root is `robots/motoman/curobo/`.
- Ignored collision links are Robotiq/Motoman terminal links.
- Workspace defaults in sim are hard-coded in Motoman abstraction: `pose_x=0.9-0.29`, `pose_y=0-0.5`, `pose_z=0.90`, size `[0.58, 1.0, 0.5]`.

Evidence:

- `motoman.py:24-27`: gripper link/group/camera link.
- `motoman.py:31-47`: ignored Motoman/Robotiq collision links.
- `motoman.py:49-61`: Motoman URDF and CuRobo config path.
- `motoman.py:112-130`: `CuroboPlanner(... ["motoman_left_ee", "motoman_right_ee"], ...)`.
- `motoman.py:138-182`: perception interface and workspace defaults.

### 2. Grasp Generation

Origin:

- `scripts/grasp_planner/curobo_grasp_planner.py:250-257`
- External CGN service, Phase 1 ZMQ bridge.
- Clean audit: `clean_replacement_torc_repro_notes/reports/CLEAN_CGN_ROBOT_SPECIFICITY_AUDIT.md`.

Embedded assumptions:

- CGN raw neural API is partly robot-neutral, but the ROS pose output is not purely raw: `build_6d_grasp` encodes parallel-jaw depth/opening conventions.
- TORC consumes pose-level `poses`, scores, samples, object ids, rather than low-level contact/direction/opening fields.

Critical measured facts from prior clean audit:

- CGN `build_6d_grasp` default gripper depth: `0.1034 m`.
- CGN config includes `DATA.gripper_width=0.08 m` and `TEST.extra_opening=0.005 m`.
- TORC then applies its own Motoman/Robotiq hand-depth semantics.

### 3. Grasp Filtering

Origin:

- `scripts/grasp_planner/curobo_grasp_planner.py:265-356`, `858-877`, `966-1158`.

Embedded assumptions:

- For CGN grasps, TORC sets `self.hand_depth = 0.11`.
- It converts raw pose to a Motoman EE goal by adding `hand_depth * local_approach` to translation before IK.
- IK feasibility is checked by `TracIKSolver(urdf_file, base_link, ee_link)` where base and EE come from Motoman CuRobo config.
- Collision filtering uses `RobotWorld` from the Motoman CuRobo robot config.
- End-effector collision toggles depend on Robotiq/Motoman link names.
- Extra collision spheres are attached to the EE at local y offsets `[-0.04, -0.02, 0, 0.02, 0.04]`, radius `0.012`.
- GraspPlotter uses Robotiq-like geometry: `finger_width=0.0065`, `outer_diameter=0.098`, `hand_depth=0.055`.

Evidence:

- `curobo_grasp_planner.py:279-287`: grasp planner hand-depth constants.
- `curobo_grasp_planner.py:302-317`: RobotWorld config, base/EE link, TracIK.
- `curobo_grasp_planner.py:343-356`: extra EE spheres.
- `curobo_grasp_planner.py:858-865`: local approach displacement before IK.
- `curobo_grasp_planner.py:987-1094`: collision filters using `RobotWorld`.
- `curobo_grasp_planner.py:1102-1123`: pregrasp/retraction generation.
- `grasp_plotter.py:15-18`, `93-123`: Robotiq visualization dimensions and axes.

### 4. IK Solver

Origin:

- `scripts/grasp_planner/curobo_grasp_planner.py:302-317`
- `scripts/motion_planner/curobo_planner.py:38`, `51-64`, `116-163`.

Embedded assumptions:

- IK frame contract is `base_link -> motoman_right_ee`.
- `TracIKSolver` consumes the Motoman URDF.
- Pink Cartesian IK in `CuroboPlanner` is driven by Pinocchio model built from the same URDF.
- Limit overrides are explicitly Motoman/UR5e-specific, not generic robot abstractions.

Evidence:

- `robots/motoman/curobo/motoman.yml:23-35`: `base_link`, `ee_link`, lock joints.
- `motion_planner/curobo_planner.py:51-64`: TORC planner frames default to Motoman.
- `motion_planner/curobo_planner.py:151-162`: Pinocchio model and Motoman limit override.

### 5. Motion Planner

Origin:

- `scripts/motion_planner/curobo_planner.py:116-163`, `728-829`.
- `scripts/task_planner/curobo_open_loop.py:806-836`, `914-959`.

Embedded assumptions:

- CuRobo `MotionGen` is configured with Motoman YAML, URDF, USD, collision link names, collision spheres, lock joints.
- Static world is based on workspace ROS params.
- Planning scene uses an anonymous point-cloud mesh named `world` from `all_pts`; it is not a semantic object-world map.
- `plan1` is a CuRobo `pose_motion_plan` from current joint state to selected pregrasp.
- `plan2` is Pink Cartesian IK from pregrasp to contact pose.
- `plan3` is Pink Cartesian IK lift from contact pose with world collision reset via `planner.set_planning_scene(None)`.

Evidence:

- `motion_planner/curobo_planner.py:146-149`: EE/link chain selection.
- `motion_planner/curobo_planner.py:728-829`: point cloud to `Mesh.from_pointcloud(..., "world")` and MotionGen world update.
- `curobo_open_loop.py:806-836`: `planner.set_planning_scene(all_pts)` then `planner.pose_motion_plan(joint_state, p_grasp[ind])`.
- `curobo_open_loop.py:914-919`: `pink_cartesian_motion(... grasp[ind])` for approach/contact.
- `curobo_open_loop.py:931-959`: lift planning after clearing scene.
- Clean source trace: `CHECKPOINT_D5_ORIGINAL_TORC_COLLISION_SEMANTICS_SOURCE_TRACE.md`.

### 6. Execution Layer

Origin:

- `scripts/task_planner/eutils.py`
- `scripts/task_planner/curobo_open_loop.py:1288-1310`
- `scripts/execution_scene/motoman_interface.py`
- `scripts/execution_scene/motoman_node.py`

Embedded assumptions:

- `ee_open()` sends service `ee_control("robotiq", 0.085)`.
- `ee_close()` sends service `ee_control("robotiq", 0.0)`.
- `wait_till_gripper()` watches `/joint_states` for `finger_joint`.
- Trajectories go through `execute_trajectory`.
- Motoman interface supports only `robotiq` in `ee_control`.
- Execution sequence opens before plan index `0` and closes before plan index `2`, after pregrasp/contact approach.

Evidence:

- `eutils.py:15-19`: command name and open/close width.
- `eutils.py:38-58`: gripper wait uses `finger_joint`.
- `eutils.py:61-99`: trajectory service interface.
- `eutils.py:102-137`: EE open/close service calls.
- `curobo_open_loop.py:1288-1310`: execution loop and close timing.
- `motoman_interface.py:753-792`: only `robotiq` gripper is accepted.
- `motoman_node.py:96-141`: Motoman joint list, init joints, gripper name, Robotiq stroke.

### 7. Simulator Coupling

Origin:

- `scripts/execution_scene/execution_node.py`
- `scripts/execution_scene/motoman_node.py`
- `xmls/robot_*.xml`
- `robots/motoman/*`
- `launch/control_server.launch`

Embedded assumptions:

- MuJoCo scene XML includes Motoman/Robotiq bodies, joints, actuators, cameras, and collision geoms.
- `MotomanNode` owns the runtime simulation node.
- Robot body names, joint names, actuator names, gripper subtree, and object naming conventions are hard-coded.
- Segmentation excludes bodies named `base`, `motoman_base`, `world`, and `workspace`.
- Contact success detection seeds from geoms under ancestor `robotiq_2f85`.

Evidence:

- `control_server.launch:19-31`: launches `motoman_node.py`, `motoman_interface.py`, Motoman xacro, Motoman SRDF, static base transforms.
- `execution_node.py:361-378`: loads MuJoCo model/data and dense video recorder.
- `execution_node.py:703-728`: segmentation remaps robot/workspace bodies by name.
- `execution_node.py:920-986`: each step applies trajectories, advances MuJoCo, and records video.
- `motoman_node.py:90-215`: Motoman-specific sim node fields.
- `motoman_node.py:209-214`: Robotiq gripper geom ids are found via ancestor `robotiq_2f85`.

### 8. Visualization Rendering

Origin:

- `scripts/execution_scene/execution_node.py:118-190`, `904-925`
- `scripts/grasp_planner/grasp_plotter.py`
- `launch/control_server.launch:5-25`

Embedded assumptions:

- Dense video records named MuJoCo cameras, default `back_view`.
- RViz grasp markers assume Robotiq-style dimensions and axes.
- Object markers depend on object body names and visual geom naming convention `name + "_vis"`.
- ROS image topics are `/camera{i}/...` and frame ids are `camera{i}_color_optical_frame`.

Evidence:

- `execution_node.py:118-190`: dense video setup and render loop.
- `execution_node.py:686-777`: RGB/depth/segmentation rendering and publish.
- `execution_node.py:782-881`: object pose/name marker publishing.
- `grasp_plotter.py:61-160`: marker geometry from Robotiq-like hand model.

## Dependency Graph

Main runtime dependency chain:

```text
perception
  -> Contact-GraspNet grasp generation
  -> TORC grasp parameterization
  -> IK feasibility filter
  -> collision / singular-object / pregrasp filters
  -> CuRobo pregrasp planner
  -> Pink Cartesian approach/contact and lift
  -> execution interface
  -> MuJoCo Motoman/Robotiq simulation
  -> success detection
  -> rendering / dense video
```

Requested chain:

```text
grasp -> filter -> IK -> planner -> execution -> sim -> render
```

Expanded edge map:

| Edge | Producer | Consumer | Robot-dependent contract |
|---|---|---|---|
| perception -> grasp | fused points, masks, object ids | CGN service / GraspPlanner | Point clouds are in TORC world/camera frame; object masks from MuJoCo/ROS segmentation. |
| grasp -> parameterization | CGN pose + score + object id | TORC hand-depth transform | CGN pose already has gripper-depth convention; TORC adds `0.11 m` along local approach for Motoman EE. |
| parameterization -> IK | transformed `T_world_motoman_ee_goal` | TracIK | Requires Motoman URDF, `base_link`, `motoman_right_ee`. |
| IK -> filter | joint candidates | RobotWorld collision filters | Requires Motoman CuRobo collision spheres and ignored Robotiq/Motoman links. |
| filter -> planner | selected pregrasp/contact pose | CuroboPlanner | Requires Motoman MotionGen config and joint ordering. |
| planner -> execution | `JointTrajectory` plan1/2/3 | `execute_trajectory`, `ee_control` | Requires Motoman joint names, Robotiq command interface, `finger_joint`. |
| execution -> sim | arm/EE trajectory queues | MuJoCo model/data | Requires MJCF actuator/joint/body names and Robotiq geoms. |
| sim -> success | contacts/object poses | `/experiment_result` | Grasp success is target body in recursive contact with Robotiq geoms. |
| sim -> render | MuJoCo data/cameras | dense video + ROS images + RViz | Requires named cameras and visual body/geom conventions. |

## Phase 1 Baseline Runtime Evidence

Phase 1 success report confirms:

- Run target: `tests/scenes/final/difficult_116.xml`, target `obj_000070_0`, method `dg_only`.
- CGN ZMQ route was used: `TORC_USE_CGN_ZMQ=1`.
- Five real CGN inference requests occurred.
- The task loop reached `GraspPlanner init done`, `open_loop_pick_or_place done`, and `retrieve_success`.
- Dense video: `torc_repro_baseline.mp4`.

Evidence files:

- `PHASE1_TORC_REPRO_SUCCESS_REPORT.md`
- `phase1_baseline_artifacts/curobo_control_stage_log.txt`
- `phase1_baseline_artifacts/output.csv`
- `phase1_baseline_artifacts/phase1_cgn_zmq_20260704_125029.log`

## Replacement-Relevant Boundaries

Do not carry these into Franka unchanged:

- `p_grasp` as if it were a robot-neutral contact pose.
- `hand_depth=0.11` and Robotiq GraspPlotter dimensions.
- `base_link -> motoman_right_ee` IK frame.
- Motoman collision spheres and ignored Robotiq links.
- Execution command `ee_control("robotiq", width)` and `finger_joint` wait logic.
- `/experiment_result` contact seeding from `robotiq_2f85`.
- `control_server.launch` Motoman xacro/SRDF/static transform wiring.

Can likely remain conceptually unchanged, after frame/robot tests:

- RGB-D fusion and object mask interface.
- CGN low-level candidate concept: contact point, approach direction, closing/base direction, score, opening.
- Anonymous point-cloud world obstacle concept for CuRobo, if the Franka planner world frame is proven.
- Dense video recorder pattern, if camera names and scene layout are redefined.

## Minimum Tests Implied Before Modification

The Phase 3 test plan should include at least:

- FK/IK frame test: selected EE frame must be Franka TCP/hand frame, not inherited Motoman EE.
- Grasp adapter test: CGN raw/pose candidate must produce visually correct Franka contact and pregrasp in MuJoCo.
- Collision filter test: Franka gripper collision links and ignored links must be explicit and verified.
- Planner scene test: `all_pts` frame must align with Franka CuRobo world.
- Execution test: open/close command must actuate Franka fingers and wait on the correct joints.
- Success detection test: target contact/lift detection must use Franka finger/contact geoms, not Robotiq subtree names.
- Rendering test: dense video camera list must exist and show the full Franka execution.

## Audit Conclusion

TORC is a working Motoman SDA10F + Robotiq 2F-85 pipeline. Its perception-to-planning architecture is reusable, but the concrete grasp pose contract, IK/planning robot model, execution interface, MuJoCo simulation, success detector, and grasp visualization are Motoman/Robotiq-specific. Phase 2 should define a single Franka robot interface and replace all robot-dependent contracts together, rather than patching individual constants.
