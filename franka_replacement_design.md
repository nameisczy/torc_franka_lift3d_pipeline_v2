# Franka Replacement Design - Phase 2

Date: 2026-07-04

Workspace: `/mnt/ssd/ziyaochen/torc_franka_lift3d_pipeline_v2`

Inputs:

- `TORC_SYSTEM_AUDIT.md`
- `SYSTEM_DEPENDENCY_GRAPH.png`
- `geometry_diff.json`
- `assets/franka/config/phase_0_5_validation_report.json`
- `assets/franka/mjcf/robosuite/panda_robot.xml`
- `assets/franka/mjcf/robosuite/panda_gripper.xml`
- `assets/franka/mjcf/robosuite/panda_lift_compiled.xml`

## Design Goal

Replace TORC's Motoman SDA10F + Robotiq 2F-85 stack with a Franka Panda + PandaGripper stack for a Lift3D-style manipulation pipeline, while correcting the architectural flaw discovered in Phase 1.5:

```text
old incorrect flow:
CGN -> TORC grasp hack -> IK -> planner -> sim

corrected flow:
CGN -> CanonicalGrasp -> RobotAdapter -> IK -> planner -> execution -> simulation
```

The central rule is strict:

```text
No robot-specific geometry is allowed above the RobotAdapter layer.
```

Perception and grasp representation must stay robot-independent. CuRobo, MuJoCo, execution, visualization, and task success logic must consume only robot-specific outputs derived from `CanonicalGrasp` by the selected `RobotAdapter`.

## Geometry Mismatch

Measured values are preserved from the first Phase 2 pass and are recorded in `geometry_diff.json`.

| Quantity | TORC / Motoman + Robotiq | Franka / PandaGripper | Replacement implication |
|---|---:|---:|---|
| Model-relative active shoulder / joint height | Motoman right shoulder raw z `1.2 m` | Franka joint1 z `0.333 m` from source model | Franka base placement must be solved from workspace calibration or reachability, not copied from TORC. |
| Robosuite task placement | N/A in original TORC scene | table top z `0.8 m`, robot0_base z `0.912 m`, joint1 z `1.245 m` | Robosuite gives a validated placement example, not a universal constant. |
| Kinematic path upper bound | right shoulder to tool `1.022 m`; base to tool `2.50524 m` including torso | Franka base to grip site `1.415762 m` upper bound | Reach must be verified by IK/planning sweep over the actual TORC workspace. |
| Gripper opening | Robotiq command open `0.085 m` | XML ideal total open `0.08 m`; validated open `0.07605 m` | Opening feasibility is checked inside the Franka adapter, never in canonical grasp generation. |
| TCP / tool offset | arm7 to `ee_right` approx z `0.147 m`; old TORC applied `0.11 m` local approach shift | `robot0_right_hand` to `gripper0_right_grip_site` z `0.097 m` | TCP offset belongs only to RobotAdapter / robot model. It must not appear in `CanonicalGrasp`. |
| Collision model | Motoman CuRobo spheres, Robotiq links, extra EE spheres | Panda link collision meshes plus pad boxes | Collision models are robot-specific and live below RobotAdapter. |

These measurements explain why adapter-only patching was insufficient: the old pipeline embedded robot-specific offsets before IK and planning, so the "grasp" object was already polluted before any robot abstraction could act.

## Canonical Grasp Definition

`CanonicalGrasp` is the only valid interface between perception/grasp detection and robot-specific systems.

It is robot-independent and expressed in the world frame used by perception.

```python
@dataclass(frozen=True)
class CanonicalGrasp:
    contact_pose: SE3
    approach_direction: np.ndarray
    grasp_axis: np.ndarray
    opening_width: float
    score: float
    object_id: int | str
```

Required semantics:

- `contact_pose`: an SE(3) pose in the perception world frame. Its origin is the intended object contact-centered grasp frame, not any robot wrist, hand, EE, or TCP frame.
- `approach_direction`: unit vector in world frame indicating approach direction toward contact.
- `grasp_axis`: unit vector in world frame indicating finger closing direction.
- `opening_width`: scalar candidate opening in meters, expressed as a robot-agnostic object/grasp width request.
- `score`: confidence from grasp generator or post-ranking.
- `object_id`: object/segment id from perception.

`CanonicalGrasp` must not include:

- robot-specific hand depth
- Motoman, Robotiq, Franka, Panda, TCP, EE, or frame names
- gripper geometry assumptions
- Robotiq or Panda opening limits
- collision links or collision spheres
- heuristic offsets such as the old `+0.11 m` local approach shift
- pre-IK translation modifications
- planner or simulator joint names

Validation of `CanonicalGrasp` is limited to representation-level checks:

- vectors are unit length and mutually valid for a grasp frame
- pose is in the declared world frame
- opening is non-negative
- score and object id are present

Robot feasibility is not a canonical-grasp concern. It belongs below `RobotAdapter`.

## Pipeline Redesign

The corrected pipeline is:

```text
RGB-D / segmentation
  -> point cloud + object masks
  -> Contact-GraspNet or other grasp generator
  -> CanonicalGrasp list
  -> ranking / object selection on CanonicalGrasp only
  -> RobotAdapter(CanonicalGrasp, RobotModel)
  -> RobotGraspCommand
  -> IK
  -> CuRobo planner
  -> execution interface
  -> MuJoCo simulation
  -> TaskSuccessOracle
  -> rendering / diagnostics
```

The old pipeline is explicitly invalid:

```text
CGN pose -> TORC hand-depth hack -> Motoman EE goal -> IK
```

No subsystem above `RobotAdapter` may know whether the active robot is Motoman or Franka.

Allowed above `RobotAdapter`:

- point clouds
- segmentation masks
- object ids
- canonical contact pose
- approach and grasp axes
- robot-agnostic opening request
- confidence scores

Forbidden above `RobotAdapter`:

- TCP offsets
- EE link names
- gripper pad sizes
- robot joint names
- robot collision links
- robot-specific opening clamps
- robot-specific pregrasp distances
- MuJoCo body or geom names

## RobotAdapter Responsibility Boundary

`RobotAdapter` is a pure geometry mapping layer from `CanonicalGrasp` to robot-specific commands.

It is responsible for:

- mapping canonical contact frame to robot TCP contact pose
- creating robot-specific pregrasp and lift target poses from configured robot geometry
- converting canonical opening request into a robot-specific gripper command
- exposing robot-specific TCP, base frame, joint names, and collision model to IK/planning
- producing typed outputs consumed by CuRobo and MuJoCo execution

It must not:

- modify canonical grasp semantics
- re-rank grasps using hidden robot-specific heuristics above the adapter boundary
- encode CGN-specific assumptions
- apply TORC hand-depth constants
- mutate the original `CanonicalGrasp`
- hide frame conversions inside planner or simulator code

Adapter output should be explicit:

```python
@dataclass(frozen=True)
class RobotGraspCommand:
    canonical_grasp_id: str
    robot_name: str
    base_frame: str
    tcp_frame: str
    tcp_contact_pose_world: SE3
    tcp_pregrasp_pose_world: SE3
    tcp_lift_pose_world: SE3
    gripper_open_width_m: float
    gripper_close_command: object
    collision_model_id: str
    joint_name_order: tuple[str, ...]
```

CuRobo consumes `RobotGraspCommand` target poses, frame names, collision model, and joint order. MuJoCo execution consumes only planned trajectories and gripper commands derived from the same `RobotGraspCommand`.

## RobotBase and FrankaRobot

`RobotBase` describes a concrete robot model. It does not own perception or canonical grasp generation.

```python
class RobotBase:
    name: str
    planner_names: RobotNameMap
    sim_names: RobotNameMap
    base_frame: str
    ee_frame: str
    tcp_frame: str
    arm_joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]

    def fk(self, joint_state, frame: str) -> SE3:
        """Return T_world_frame using the robot model and current base placement."""

    def ik(self, tcp_pose_world: SE3, seed=None, collision_world=None):
        """Solve IK for a robot TCP pose produced by RobotAdapter."""

    def collision_model(self) -> CollisionModel:
        """Return the robot-specific collision model for planning."""

    def make_adapter(self) -> RobotAdapter:
        """Return the geometry adapter for this robot."""
```

`FrankaRobot(RobotBase)` is the only source for Franka-specific names and geometry:

| Field | Value / source |
|---|---|
| `name` | `franka_panda` |
| Planner base | generated `panda_link0` or equivalent |
| Sim base | `robot0_base` in compiled robosuite MJCF |
| Planner EE | generated `panda_hand` or equivalent |
| Sim EE | `robot0_right_hand` |
| TCP | `gripper0_right_grip_site` mapped to planner `panda_tcp` |
| Arm joints | `robot0_joint1` ... `robot0_joint7` in MuJoCo; planner aliases must map exactly |
| Finger joints | `gripper0_right_finger_joint1`, `gripper0_right_finger_joint2` |
| TCP offset source | `robot0_right_hand -> gripper0_right_grip_site`, approx `[0, 0, 0.097] m` |
| Max opening source | XML ideal `0.08 m`, validated effective open `0.07605 m` |
| Dense render cameras | `agentview`, `sideview`, `frontview`, `birdview`, optional `robot0_eye_in_hand` |

Franka-specific limits, TCP offsets, pad sizes, collision meshes, and joint order are used only by `FrankaRobot` and `FrankaRobotAdapter`.

## Removed TORC Heuristics

The following old TORC assumptions are removed from the architecture:

| Removed assumption | Old location / behavior | Replacement |
|---|---|---|
| `hand_depth = 0.11 m` heuristic | TORC grasp planner shifted CGN pose along local approach before IK | No canonical shift. Franka adapter maps canonical contact frame to TCP targets using calibrated robot geometry. |
| CGN pose plus local approach hack | `pose.translation += hand_depth * approach` | CGN output is normalized into `CanonicalGrasp`; robot geometry is applied only by adapter. |
| Robotiq width-based filtering in grasp representation | Old candidate filtering assumed Robotiq opening and visual geometry | Canonical opening remains a request. Robot-specific feasibility is adapter/planner output metadata. |
| Implicit EE offset embedded in grasp pose | Old `p_grasp` mixed grasp, EE, and hand-depth semantics | Canonical pose is contact-centered. TCP offset is explicit in `RobotGraspCommand`. |
| Pre-IK translation based on robot type | Old grasp candidates were moved before IK | IK only receives adapter-produced TCP poses. |
| Robotiq link names in success logic | `/experiment_result` seeded contacts from `robotiq_2f85` geoms | `TaskSuccessOracle` uses object pose/contact state/lift height, not gripper-specific geom names. |

The old constants may remain in historical audit docs and in `geometry_diff.json` as measured mismatch evidence, but they are not valid design parameters for the new pipeline.

## MuJoCo vs CuRobo Consistency Constraints

The migration is valid only if MuJoCo and CuRobo share the same robot contract below `RobotAdapter`.

Consistency requirements:

- TCP frame: CuRobo planner TCP and MuJoCo execution TCP must refer to the same physical point, `gripper0_right_grip_site` / planner `panda_tcp`.
- Base frame: planner base and MuJoCo base placement must be related by a single explicit transform.
- Collision geometry: CuRobo collision model must correspond to MuJoCo collision geometry closely enough for contact and clearance tests.
- Joint ordering: planned trajectory joint order must map exactly to MuJoCo control joint order.
- Gripper state: adapter gripper command must map to both MuJoCo finger joints consistently.
- Object/world frame: perception point cloud, canonical grasp poses, CuRobo world obstacles, and MuJoCo object poses must share one world-frame convention.

Data ownership:

```text
CanonicalGrasp
  owns: world-frame grasp semantics
  does not own: robot geometry

RobotAdapter
  owns: canonical -> robot TCP/command conversion

CuRobo
  consumes: RobotGraspCommand target poses, collision model, joint order

MuJoCo
  consumes: executed trajectories and gripper commands

TaskSuccessOracle
  consumes: object pose/contact/lift state
```

Hard rule:

```text
If a value mentions panda, franka, motoman, robotiq, tcp, ee_link, joint, geom, collision sphere, or gripper pad, it must not exist above RobotAdapter.
```

## Base Placement Solver

Franka base placement is a solved variable, not a constant.

Hardcoded Motoman base poses and static transforms must be removed from the Franka path. Robosuite's `robot0_base z = 0.912 m` with table top `z = 0.8 m` is a useful initial prior, not a universal value.

Inputs:

- calibrated table plane / table top height
- workspace bounds from perception
- target object distribution
- Franka reachability under joint limits
- collision clearance from table, shelf, objects, camera geometry
- desired camera visibility, if using fixed external cameras

Solver output:

```yaml
franka_base_pose:
  frame: world
  position: [x, y, z]
  orientation_xyzw: [qx, qy, qz, qw]
  evidence:
    ik_feasible_fraction: ...
    collision_free_fraction: ...
    workspace_coverage: ...
```

Initial z prior:

```text
base_z ~= table_top_z + 0.112
```

But final base pose must be selected by workspace calibration or IK feasibility sweep. Phase 3 must reject a base placement if sampled canonical grasps cannot be reached collision-free.

## Planning Replacement

Generate or add a Franka CuRobo config:

```text
assets/franka/config/curobo/franka_panda.yml
assets/franka/urdf/franka_panda.urdf
assets/franka/config/curobo/franka_panda_collision_spheres.yml
```

Required planner contract:

- Base link maps to the solved Franka base frame.
- TCP link maps to `panda_tcp` / `gripper0_right_grip_site`.
- Arm joints are exactly seven Franka joints.
- Gripper joints are excluded from arm MotionGen but tracked by execution and simulation.
- Collision links are Panda links, hand, fingers, and pad proxies.
- Ignored collision pairs are Franka-specific and test-proven.

Planner sequence:

1. Adapter converts `CanonicalGrasp` to `RobotGraspCommand`.
2. IK checks `tcp_pregrasp_pose_world`.
3. CuRobo plans to pregrasp.
4. Cartesian IK approaches to `tcp_contact_pose_world`.
5. Execution closes gripper.
6. Cartesian lift moves to `tcp_lift_pose_world`.

All target poses are TCP poses. No planner receives a CGN pose or TORC-mutated grasp pose.

## Execution Replacement

Replace Motoman/Robotiq services with a Franka execution path:

- `FrankaExecutionInterface.execute_trajectory(trajectory)`
- `FrankaExecutionInterface.set_gripper_width(width_m)`
- wait on `gripper0_right_finger_joint1` and `gripper0_right_finger_joint2`
- open width comes from `RobotGraspCommand.gripper_open_width_m`
- close command comes from `RobotGraspCommand.gripper_close_command`

The public callsites may keep high-level verbs such as `open_gripper()` and `close_gripper()`, but those verbs must dispatch through the selected `RobotBase` / `RobotAdapter`. They must not inspect `TORC_ROBOT` and branch inside grasp or planner code.

## Simulation Replacement

Create a Franka simulation node rather than mutating `MotomanNode`:

```text
scripts/execution_scene/franka_node.py
scripts/execution_scene/franka_interface.py
launch/franka_control_server.launch
```

Responsibilities:

- Load a Franka-compatible MuJoCo scene built from robosuite Panda + TORC/YCB object world.
- Publish camera RGB/depth/segmentation with the same perception topic contract.
- Publish object poses and id/name maps.
- Publish robot joint states through a stable name map.
- Accept arm trajectories for `robot0_joint1..7`.
- Accept gripper commands for Panda finger actuators.
- Expose object pose and contact state for `TaskSuccessOracle`.

MuJoCo must consume only robot-specific execution outputs:

- planned arm trajectories
- gripper commands
- solved base placement
- robot-specific collision/contact labels

MuJoCo must not consume raw CGN poses or `CanonicalGrasp` directly.

## TaskSuccessOracle

Replace TORC-dependent `/experiment_result` logic with `TaskSuccessOracle`.

It must depend on:

- object pose
- object contact state
- object lift height relative to table
- task target id

It must not depend on:

- Robotiq geoms
- Panda pad geom names as semantic success definitions
- gripper-specific subtree traversal
- fixed dropped threshold such as global `z < 0.5`

Recommended contract:

```python
class TaskSuccessOracle:
    def evaluate(self, target_object_id, object_states, contact_state, table_frame):
        return TaskResult(
            grasped=...,
            lifted=...,
            success=...,
            dropped=...,
        )
```

Success rule:

```text
lifted = object_pose.z > table_top_z + lift_margin
success = target_object_id lifted and not dropped
```

Contact state may be used to distinguish grasped vs incidental lifted state, but the oracle receives contact state as semantic data, not hard-coded Robotiq/Panda geom names. The simulator-specific mapping from MuJoCo contacts to semantic contact state lives below the oracle.

## Visualization Replacement

Update rendering and debug output:

- Dense video cameras: `agentview`, `sideview`, `frontview`, `birdview`, optional `robot0_eye_in_hand`.
- Debug frames:
  - `T_world_canonical_contact`
  - `canonical_approach_direction`
  - `canonical_grasp_axis`
  - `T_world_robot_tcp_contact`
  - `T_world_robot_tcp_pregrasp`
  - `T_world_robot_tcp_lift`
- RViz grasp markers above adapter render only canonical grasp axes/contact frames.
- Robot-specific gripper markers are rendered only below adapter with explicit robot label.
- Collision visualization displays the selected robot collision model and must identify whether it came from MuJoCo or CuRobo.

## Updated System Replacement Plan

| Subsystem | Corrected decision | Boundary rule |
|---|---|---|
| Perception | Unchanged algorithmically | Produces world-frame points, masks, object ids. No robot geometry. |
| Grasp detection | Keep CGN or any grasp generator | Output must be normalized to `CanonicalGrasp`. No TORC pose hack. |
| Grasp representation | New mandatory layer | `CanonicalGrasp` is the only interface above robot-specific systems. |
| Grasp ranking/filtering | Split into canonical and robot-feasibility stages | Canonical filters may use score/object/geometry only. Robot feasibility occurs after adapter. |
| RobotAdapter | New hard boundary | Converts `CanonicalGrasp` to `RobotGraspCommand`; pure geometry mapping. |
| IK | Robot-specific below adapter | Consumes only adapter TCP poses and robot model. |
| CuRobo planning | Reconfigured for Franka | Consumes only `RobotGraspCommand`, collision model, base transform, joint order. |
| Execution | Replaced | Consumes planned trajectories and gripper commands, not canonical grasps. |
| MuJoCo simulation | Rewritten | Consumes execution outputs and publishes semantic object/contact state. |
| Success detection | Replaced with `TaskSuccessOracle` | Uses object pose, contact state, lift height relative to table. |
| Rendering | Updated | Canonical visualization above adapter; robot visualization below adapter. |

## Proposed File Layout

```text
assets/franka/
  config/
    franka_robot.yaml
    curobo/franka_panda.yml
    curobo/franka_panda_collision_spheres.yml
  urdf/
    franka_panda.urdf
  mjcf/
    robosuite/
      panda_robot.xml
      panda_gripper.xml
      panda_lift_compiled.xml

original_torc/lab_vbnpm/scripts/
  grasp_representation/
    canonical_grasp.py
    cgn_to_canonical.py
  robot_interface/
    robot_base.py
    robot_adapter.py
    franka_robot.py
    franka_adapter.py
    motoman_robot.py
    motoman_adapter.py
  task_success/
    task_success_oracle.py
  execution_scene/
    franka_node.py
    franka_interface.py
  motion_planner/
    franka_curobo_planner.py
  launch/
    franka_control_server.launch
```

`TORC_ROBOT=motoman | franka` may select the robot implementation, but it must only choose a `RobotBase` and `RobotAdapter`. It must not create scattered conditional logic in perception, canonical grasp generation, planning, execution, simulation, or rendering.

## Migration Order

1. Add `CanonicalGrasp` and `cgn_to_canonical.py`.
2. Add representation-level tests proving no robot names or robot offsets exist in canonical grasps.
3. Add `RobotBase`, `RobotAdapter`, `MotomanRobot`, and `MotomanAdapter`.
4. Move old Motoman assumptions below `MotomanAdapter` and verify the old Phase 1 path still reproduces.
5. Add `FrankaRobot` and `FrankaAdapter` using `geometry_diff.json`.
6. Add MuJoCo/CuRobo frame and joint-order consistency tests.
7. Add base placement solver / feasibility sweep.
8. Add Franka CuRobo config and collision model.
9. Add `FrankaNode`, `FrankaExecutionInterface`, and `TaskSuccessOracle`.
10. Rewire open-loop pick to consume `CanonicalGrasp -> RobotAdapter -> RobotGraspCommand`.
11. Run Phase 3 mandatory tests before enabling Franka rollout.

## Risks and Stop Conditions

Stop if any of these fail:

- `CanonicalGrasp` contains robot names, TCP names, hand depth, gripper geometry, or joint names.
- Any code above `RobotAdapter` branches on Motoman/Franka/Robotiq/Panda.
- FK between MuJoCo and CuRobo disagrees for the TCP beyond tolerance.
- Joint ordering differs between planned trajectory and MuJoCo control order.
- CuRobo collision geometry does not match MuJoCo enough to predict obvious collisions.
- Franka base solver cannot reach sampled canonical grasps collision-free.
- TaskSuccessOracle depends on `robotiq_2f85` or hard-coded Panda pad geom names.
- Dense render camera does not show the full Franka/scene/object execution.

## Phase 3 Test Requirements Derived From Design

Canonical representation:

- CGN output converts to `CanonicalGrasp` with contact pose, approach direction, grasp axis, opening, score, object id.
- Canonical representation contains no robot-specific field.
- No pre-IK translation is applied before adapter.

Robot consistency:

- FK: `robot0_joint*` MuJoCo FK vs generated URDF/CuRobo FK at `gripper0_right_grip_site`.
- Joint ordering: planner order maps exactly to sim order.
- TCP: `robot0_right_hand -> gripper0_right_grip_site = [0, 0, 0.097]` equivalent after compiled transforms.
- Base placement: solved base pose passes workspace reachability sweep.

Grasp pipeline:

- Canonical contact frame visualized before adapter.
- Franka adapter TCP contact/pregrasp/lift poses visualized after adapter.
- Franka opening feasibility is reported by adapter, not canonical generator.

Planning:

- Reachability sweep across TORC workspace.
- Collision feasibility with Panda arm, hand, fingers, table, and object cloud.
- MotionGen plan to adapter-produced pregrasp and Cartesian approach/lift.

Simulation:

- Franka base placement relative to table/workspace.
- Gripper actuation open/close.
- Semantic contact state exported to TaskSuccessOracle.

Rendering:

- Dense video from `agentview`/`sideview`/`frontview`/`birdview`.
- Canonical grasp frames and robot-specific TCP frames are both visible and labeled distinctly.

## Final Recommendation

The migration must be rebuilt around `CanonicalGrasp` as the robot-independent grasp boundary. `RobotAdapter` is the only place where robot geometry enters the pipeline. CuRobo and MuJoCo must consume consistent adapter-derived robot commands and share TCP, base frame, collision model, and joint order. This architecture prevents TORC-specific grasp pollution from leaking into Franka and makes future robot replacements possible without rewriting perception or grasp detection.
