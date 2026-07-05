# Old Robot/Gripper Measured Assumptions

Machine-readable output: `/mnt/ssd/ziyaochen/torc_franka_clean_results/old_robot_gripper_assumptions.json`

Measured from original TORC source of truth, not the old messy branch:

- Old robot class: `task_planner.motoman.MotomanSDA10F`; source `scripts/task_planner/motoman.py`; evidence `proven`.
- Old EE link: `motoman_right_ee`; sources `motoman.py` and `robots/motoman/curobo/motoman_left.yml`; evidence `proven`.
- Old base frame: `base_link`; source `motoman_left.yml`; evidence `proven`.
- Old start/locked right-arm state: `arm_right_joint_1_s=1.75`, `arm_right_joint_2_l=0.8`, `arm_right_joint_3_e=0.0`, `arm_right_joint_4_u=-0.66`, `arm_right_joint_5_r=0.0`, `arm_right_joint_6_b=0.0`, `arm_right_joint_7_t=0.0`, `finger_joint=0.0`; evidence `proven`.
- Robotiq open/close widths: open `0.085 m`, close `0.0 m`; source `eutils.py`; evidence `proven`.
- TORC CGN downstream hand depth: `0.11 m` added along local approach before IK; source `curobo_grasp_planner.py`; evidence `proven`.
- CGN `build_6d_grasp` default gripper depth: `0.1034 m`; source `contact_graspnet.py`; evidence `proven`.
- CGN config: `DATA.gripper_width=0.08 m`, `TEST.extra_opening=0.005 m`; evidence `proven`.
- `attach_extra_spheres`: five `0.012 m` radius spheres on EE link at local y `[-0.04, -0.02, 0, 0.02, 0.04]`; evidence `proven`.
- `is_grasp_above_table`: uses GraspPlotter `finger_width=0.0065`, `outer_diameter=0.098`, `hand_depth=0.055`, safety margin `0.001`, approach axis local `+Z`, closing axis local `-Y`; evidence `proven`.
- Collision names include Robotiq/finger links and Motoman terminal links; evidence `proven`.

No conclusion here verifies Franka behavior. Evidence level for Franka applicability: `unverified`.

