# Checkpoint D0.5 Original Robot Base Pose: scene20

Original scene20 XML: `/home/ziyaochen/gc6d_lift3d_traj/ros_workspace/src/lab_vbnpm/tests/scenes/tabletop_unstructured/scene20.xml`

Original old robot root body: `base`

Original old robot base pose in MuJoCo world:

```json
{
  "body_id": 1,
  "exists": true,
  "xpos_world": [
    0.0,
    0.0,
    0.0
  ],
  "xquat_wxyz_world": [
    1.0,
    0.0,
    0.0,
    0.0
  ],
  "yaw_rad": 0.0
}
```

Old robot support/root structure: `torso_link_b1` is at `[0.0, 0.0, 1.2]`.

TORC/CuRobo source: `/home/ziyaochen/gc6d_lift3d_traj/ros_workspace/src/lab_vbnpm/scripts/task_planner/motoman.py` initializes Motoman with `/home/ziyaochen/gc6d_lift3d_traj/ros_workspace/src/lab_vbnpm/robots/motoman/curobo/motoman.yml` and `/home/ziyaochen/gc6d_lift3d_traj/ros_workspace/src/lab_vbnpm/robots/motoman/curobo/motoman.urdf`. No scene20-specific post-load base transform was found in this D0.5 inspection.
