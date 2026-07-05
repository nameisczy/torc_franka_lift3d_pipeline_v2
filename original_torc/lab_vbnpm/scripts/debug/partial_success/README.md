# Partial Success Cases

**runs_locked**

1. `experiments/runs/runs_locked/grasp_planning-3/trial_2026-02-23_17-51-40__unstructured__dg_only/128__obj_000048_1__dg_only`

    - Accidental
      - `obj_000059_0, obj_000025_2`
    - Orange juice box is knocked over in the middle of a grasp
      - ![](assets/1-1.png)
    - Cup is stuck to bottom of grasp
      - ![](assets/1-2.png)

2. `experiments/runs/runs_locked/grasp_planning-3/trial_2026-02-24_00-48-51__structured__dg_only/586__obj_000063_0__dg_only`

    - Accidental
      - `obj_000083_0`
    - Toothpaste is knocked on the floor
      - ![](assets/2.png)

3. `experiments/runs/runs_locked/grasp_planning-4/trial_2026-02-24_00-48-52__structured__dg_only/223__obj_000041_0__dg_only`

    - Accidental
      - `obj_000041_2, obj_000044_0`
    - Multigrasp of mushroom can and peaches can
      - Grasp is considered a failure due to the multipick, and objects are considered dropped
      - ![](assets/3.png)

4. `experiments/runs/runs_locked/grasp_planning-4/trial_2026-02-24_00-48-52__structured__dg_only/233__obj_000006_0__dg_only`

    - Accidental
      - `obj_000025_0`
    - Multigrasp of red cup and yogurt
    - Handle of cup is hooked onto the side of the gripper, and is flung out on retrieval, causing the cup to be dropped
      - ![](assets/4.png)

5. `experiments/runs/runs_locked/grasp_planning-4/trial_2026-02-24_00-48-52__structured__dg_only/249__obj_000017_0__dg_only`

    - Accidental
      - `obj_000052_1, obj_000027_0, obj_000050_1`
    - Grabs bowl, and knocks over a cheese can on retrieval
    - Retrieves a bowl that is fused with another object
      - Grasp is considered a failure due to the multipick, and objects are considered dropped 
      - ![](assets/5-1.png)
      - ![](assets/5-2.png)

6. `experiments/runs/runs_locked/grasp_planning-4/trial_2026-02-24_00-48-52__structured__dg_only/251__obj_000049_0__dg_only`

    - Accidental
      - `obj_000043_1`
    - Grasps yogurt, which has a peas can resting ontop
    - Peas can is knocked over on retrieval
      - ![](assets/6.png)


7. `experiments/runs/runs_locked/grasp_planning-4/trial_2026-02-23_17-51-42__unstructured__dg_only/110__obj_000044_2__dg_only`

    - Accidental
      - `obj_000057_1`
    - Retrieval of cheese can (upward motion) knocks off a popcorn box
      - ![](assets/7.png)

8. `experiments/runs/runs_locked/grasp_planning-4/trial_2026-02-23_17-51-42__unstructured__dg_only/128__obj_000048_1__dg_only`

    - Accidental
      - `obj_000059_0, obj_000025_2`
    - Retrieval of clamp knocks over orange juice and red cup
      - ![](assets/8-1.png)
      - ![](assets/8-2.png)

9. `experiments/runs/runs_locked/grasp_planning-4/trial_2026-02-23_17-51-42__unstructured__dg_only/30__obj_000043_0__dg_only`

    - Accidental
      - `obj_000053_0`
    - **LOCKING ISSUE**
    - Pudding box slips out of gripper?
      - The pudding box is still in collision with the gripper in the experiment_result mujoco state. Why was the grasp not considered successful?
        - May be issue with locking
      - ![](assets/9.png)

10. `experiments/runs/runs_locked/grasp_planning-2/trial_2026-02-23_17-51-39__unstructured__dg_only/122__obj_000047_1__dg_only`

    - Accidental
      - `obj_000062_0`
    - Lifting the mac & cheese box causes the cookies box to fall over
      - ![](assets/10-1.png)
      - ![](assets/10-2.png)

11. `experiments/runs/runs_locked/grasp_planning-2/trial_2026-02-23_17-51-39__unstructured__dg_only/30__obj_000067_0__dg_only`

    - Accidental
      - `obj_000066_0`
    - Grabbing cookies box causes BBQ bottle to fall over
      - ![](assets/11-1.png)
      - ![](assets/11-2.png)

12. `experiments/runs/runs_locked/grasp_planning-0/trial_2026-02-24_00-48-48__structured__dg_only/249__obj_000048_0__dg_only`

    - Accidental
      - `obj_000050_1`
    - Bowl is fused, and is flung
      - ![](assets/12-1.png)
      - ![](assets/12-2.png)

13. `experiments/runs/runs_locked/grasp_planning-6/trial_2026-02-24_00-48-55__structured__dg_only/251__obj_000046_0__dg_only`

    - Accidental
      - `obj_000060_0`
    - Multipick fusion
      - ![](assets/13.png)

14. `experiments/runs/runs_locked/grasp_planning-6/trial_2026-02-23_17-51-45__unstructured__dg_only/110__obj_000044_2__dg_only`

    - Accidental
      - `obj_000057_1`
    - Popcorn box ontop knocked over on retrieval
      - ![](assets/14.png)

15. `experiments/runs/runs_locked/grasp_planning-6/trial_2026-02-23_17-51-45__unstructured__dg_only/105__obj_000070_0__dg_only`

    - Accidental
      - `obj_000071_1`
    - **LOCKING ISSUE**
    - Grabbed object not touching gripper
      - ![](assets/15.png)

16. `experiments/runs/runs_locked/grasp_planning-6/trial_2026-02-23_17-51-45__unstructured__dg_only/14__obj_000053_0__dg_only`

    - Accidental
      - `obj_000066_0`
    - **FORCE ISSUE**
    - Grasp fails and flings BBQ bottle off table
      - ![](assets/16.png)

17. `experiments/runs/runs_locked/grasp_planning-6/trial_2026-02-23_17-51-45__unstructured__dg_only/30__obj_000066_0__dg_only`

    - Accidental
      - `obj_000068_0`
    - **FORCE ISSUE**
    - Grasp fails and flings mustard bottle off table
      - ![](assets/17.png)  

18. `experiments/runs/runs_locked/grasp_planning-6/trial_2026-02-23_17-51-45__unstructured__dg_only/128__obj_000048_1__dg_only`

    - Accidental
      - `obj_000025_1`
    - Clamp is stuck inside of cup
    - When clamp is retrieved, cup falls off of clamp and onto the ground
      - ![](assets/18-1.png)
      - ![](assets/18-2.png)

19. `experiments/runs/runs_locked/grasp_planning-6/trial_2026-02-23_17-51-45__unstructured__dg_only/30__obj_000047_0__dg_only`

    - Accidental
      - `obj_000057_0`
    - **FORCE ISSUE**
    - Grasp of cookies flings popcorn
      - ![](assets/19-1.png)
      - ![](assets/19-2.png)

20. `experiments/runs/runs_locked/grasp_planning-1/trial_2026-02-23_17-51-37__unstructured__dg_only/110__obj_000044_2__dg_only`

    - Accidental
      - `obj_000057_1`
    - Retrieval flings popcorn
      - ![](assets/20-1.png)
      - ![](assets/20-2.png)

21. `experiments/runs/runs_locked/grasp_planning-1/trial_2026-02-23_17-51-37__unstructured__dg_only/30__obj_000067_0__dg_only`

    - Accidental
      - `obj_000057_0`
    - **FORCE ISSUE**
    - Retrieval of cookies flings popcorn (again?)
      - ![](assets/21-1.png)
      - ![](assets/21-2.png)

22. `experiments/runs/runs_locked/grasp_planning-5/trial_2026-02-24_00-48-53__structured__dg_only/251__obj_000046_0__dg_only`

    - Accidental
      - `obj_000043_1`
    - Yogurt rests ontop of peas can
    - Retrieving peas cans causes yogurt to slip off
    - ![](assets/22.png)

23. `experiments/runs/runs_locked/grasp_planning-5/trial_2026-02-24_00-48-53__structured__dg_only/241__obj_000059_0__dg_only`

    - Accidental
      - `obj_000010_0`
    - **PHYSICS ERROR**
    - Retrieval of peas somehow causes cheezit box to fly off?
      - ![](assets/23-1.png)
      - ![](assets/23-2.png)

24. `experiments/runs/runs_locked/grasp_planning-5/trial_2026-02-24_00-48-53__structured__dg_only/241__obj_000025_2__dg_only`

    - Accidental
      - `obj_000010_0`
    - **PHYSICS ERROR**
    - Retrieval of peas causes cheezit box to fly off (again)
    - Peas can clips into cheezit box after its grasped, causing the cheezit box to jitter out of the way and off the shelf
      - ![](assets/24.png)

25. `experiments/runs/runs_locked/grasp_planning-5/trial_2026-02-24_00-48-53__structured__dg_only/249__obj_000017_0__dg_only`

    - Accidental
      - `obj_000027_0`
    - **LOCKING ERROR**
    - **PHYSICS ERROR**
    - Grabbing tomato can inside bowl causes entire bowl to fling out
      - ![](assets/25-1.png)
      - ![](assets/25-2.png)

26. `experiments/runs/runs_locked/grasp_planning-5/trial_2026-02-24_00-48-53__structured__dg_only/586__obj_000063_0__dg_only`

    - Accidental
      - `obj_000043_0`
    - Yogurt balances ontop of grasped sphagetti box
    - Retrieving sphagetti box causes yogurt to topple off
      - ![](assets/26.png)

27. `experiments/runs/runs_locked/grasp_planning-5/trial_2026-02-23_17-51-43__unstructured__dg_only/105__obj_000070_0__dg_only`

    - Accidental
      - `obj_000078_1, obj_000071_1`
    - **LOCKING ISSUE**
    - Grasped tennis ball not touching gripper
      - ![](assets/27-1.png)
    - Retrieval of toilet brush knocks over a trash can
      - ![](assets/27-2.png)
      - ![](assets/27-3.png)

28. `experiments/runs/runs_locked/grasp_planning-5/trial_2026-02-23_17-51-43__unstructured__dg_only/122__obj_000047_1__dg_only`

    - Accidental
      - `obj_000047_0`
    - Retrieval knocks over corn can
      - ![](assets/28-1.png)
      - ![](assets/28-2.png)

29. `experiments/runs/runs_locked/grasp_planning-5/trial_2026-02-23_17-51-43__unstructured__dg_only/30__obj_000066_0__dg_only`

    - Accidental
      - `obj_000060_0`
    - **FORCE ISSUE**
    - Bad pick causes granola box to fall onto the floor
      - ![](assets/29-1.png)
      - ![](assets/29-2.png)