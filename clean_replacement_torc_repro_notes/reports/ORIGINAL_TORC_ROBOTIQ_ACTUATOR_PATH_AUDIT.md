# Original TORC Robotiq Actuator Path Audit

Final classification: `blocked_franka_actuator_control_repaired_but_synthetic_lift_failed`
Focused pytest result: `python -m pytest -q tests/test_pass038_gripper_actuator_control.py -> 6 passed in 0.17s`
Full pytest result: `python -m pytest -q -> 70 passed in 0.48s`

| Original TORC file/function/line | Parameter/control value | Semantics | Franka equivalent needed |
| --- | --- | --- | --- |
| /home/ziyaochen/gc6d_lift3d_traj/ros_workspace/src/lab_vbnpm/tests/scenes/tabletop_unstructured/scene20.xml:1121: <fixed name="split"> | fixed tendon `split`, right/left driver coef 0.5 | symmetric Robotiq driver coupling | two symmetric Franka finger position actuators |
| /home/ziyaochen/gc6d_lift3d_traj/ros_workspace/src/lab_vbnpm/tests/scenes/tabletop_unstructured/scene20.xml:1128: <general name="left_driver" class="2f85" tendon="split" ctrlrange="0 255" forcerange="-5 5" gainprm="0.313725" biasprm="0 -100 -10"/> | ctrlrange 0..255, forcerange -5..5, gain/bias terms | tendon position-like close driver with force limit | ctrlrange 0..0.04 per finger, forcerange (-80.0, 80.0) |
| /home/ziyaochen/gc6d_lift3d_traj/ros_workspace/src/lab_vbnpm/scripts/execution_scene/execution_node.py | capture after `mj_step` | object physics and controls advance together | Pass038 steps MuJoCo after arm qpos and gripper ctrl are applied |
