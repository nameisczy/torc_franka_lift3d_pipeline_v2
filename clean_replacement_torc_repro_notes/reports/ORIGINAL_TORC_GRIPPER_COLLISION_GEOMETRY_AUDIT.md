# Original TORC Gripper Collision Geometry Audit

Original TORC uses a mixture: visual meshes for display, full/simplified mesh collision for gripper links, and explicit Robotiq pad proxy boxes for pad contact. The pad contact surface is represented by two stacked boxes per pad.

| File/line | Geom | Parent body | Type | Size | Local pose | Contact bits | Friction | Solref/Solimp | Role | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| /home/ziyaochen/gc6d_lift3d_traj/ros_workspace/src/lab_vbnpm/tests/scenes/tabletop_unstructured/scene20.xml:63-67 | pad_box1 default | right_pad / left_pad | box | 0.011 0.004 0.009375 | instantiated at pos 0 -0.0026 0.028125 | inherits collision default contype=3 conaffinity=3 | 1 0.9 0.9 | solref=0.004 solimp=0.95 0.99 | collision/contact pad proxy | upper half of Robotiq pad contact surface |
| /home/ziyaochen/gc6d_lift3d_traj/ros_workspace/src/lab_vbnpm/tests/scenes/tabletop_unstructured/scene20.xml:63-67 | pad_box2 default | right_pad / left_pad | box | 0.011 0.004 0.009375 | instantiated at pos 0 -0.0026 0.009375 | inherits collision default contype=3 conaffinity=3 | 1 0.9 0.9 | solref=0.004 solimp=0.95 0.99 | collision/contact pad proxy | lower half of Robotiq pad contact surface |
| /home/ziyaochen/gc6d_lift3d_traj/ros_workspace/src/lab_vbnpm/tests/scenes/shelf_structured/scene179.xml:501-503 and 537-539 | right/left pad_box1 + pad_box2 instances | right_pad / left_pad | box | from defaults | 0 -0.0026 0.028125 and 0 -0.0026 0.009375 | 3/3 via collision default | 1 0.9 0.9 | 0.004 / 0.95 0.99 | collision-only pad boxes plus visual pad mesh | two stacked boxes cover full pad length from z=0 to z=0.0375 |
