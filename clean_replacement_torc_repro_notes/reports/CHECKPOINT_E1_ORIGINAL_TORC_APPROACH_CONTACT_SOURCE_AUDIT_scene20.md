# Checkpoint E1 Original TORC Approach/Contact Source Audit: scene20

Evidence level: `proven`

Recovered approach policy: original TORC uses `pink_cartesian_motion(joint_state2, grasp[ind], return_all=True)` from the terminal pregrasp joint state to the grasp/contact pose. It does not run a second active MotionGen pregrasp-to-contact plan.

Collision world: the D5 anonymous `all_pts`/`world` mesh remains loaded because no `set_planning_scene` call occurs between plan1 and plan2, but `pink_cartesian_motion` itself is a Pinocchio/Pink Cartesian IK interpolation routine and has no explicit RobotWorld collision query.

Close timing: `ee_close()` is called only later in the execution loop at plan index `2`, after pregrasp and approach/contact. E1 must not close.

JSON artifact: `/mnt/ssd/ziyaochen/torc_franka_clean_results/checkpoint_e1_approach_contact_scene20_20260630T121142Z/checkpoint_e1_original_torc_approach_contact_source_audit.json`
