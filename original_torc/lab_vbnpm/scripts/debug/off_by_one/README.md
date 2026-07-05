# Off-by-one bug

The off-by-one bug occurs when motoman fails to pick the target object and the object is not detected as dropped. However, in the next pick the target object is detected as dropped.

This happens becauase
- Knocking over target at last pick
  - Motomans fails to pick the target object because slips out of the grippers near the end of the retrieval. After slipping out, the target object is in the air and in the middle of falling down. The target object is not touching the gripper and it's also not fallen far down enough to be considered dropped.
- Slippage of target at second to last pick
  - Motoman fails to pick the target object, and it lands on a ledge. Motoman attempts to repick the target object in the next pick, which knocks it onto the ground.
- Last pick grabs multiple objects, including the target object
  - Since multiple objects are grasped, the grasp is considered a failure. However, we don't stop the experiment early.

Other issues
- Objects are fused together
  - In `structuerd_249.xml`, the bowl is fused with the green bean can

## TODO

### Tasks
- [X] Search through new Daniel trials + old trials to identify experiments with off-by-one error
  - [X] Find out if there is a pattern to this error (specific scenes, etc.)
- [X] Add logging of mujoco scenes on get experiment result
  - [X] Add thread locking to avoid race conditions from logging

### Not Planned
- [ ] Try reproducing bug without docker, only running experiments single threaded on one computer.
- Might be related to motoman not fully resetting issue 
  - [ ] Creating isolated ROS and motoman_node setup, and benchmarking timing
    - [ ] Ensure motoman_node only
- Might be a docker only issue
  - [ ] Turn isolated ROS into docker setup, and rFVun a hundred experiments to attempt to reproduce the error


## Info

**Replaying ROSBag**
```
roslaunch lab_vbnpm replay.launch bag_path:=
```

### Errors

**_OFF_BY_ONE_user_custom_runs**

This is a local run.

1. `experiments/runs/_OFF_BY_ONE_user_custom_runs/trial_2026-02-15_10-34-54__unstructured__dg_only/24__obj_000041_1__dg_only/`

  ```
  ⚠️  Detected second-to-last drop target object bug. Applying fix @
      experiments/runs/trial_2026-02-15_10-34-54__unstructured__dg_only/24__obj_000041_1__dg_only/output.csv
  ```
   - experiments/runs/trial_2026-02-15_10-34-54__unstructured__dg_only/rosbag.bag
   - unstructured_24.xml
     - obj_000041_1 -> Mushrooms can
   - **DIAGNOSIS**
     - Mushrooms can slips from grasp at last second, causing the can to be in the air and not detected as grasped nor dropped

**daniel_test_runs_OFF_BY_ONE**

1. `experiments/runs_cache/daniel_test_runs_OFF_BY_ONE/grasp_planning-3/trial_2026-01-12_04-04-13__unstructured__dg_only/24__obj_000041_1__dg_only/`

  ```
  ⚠️  Detected second-to-last drop target object bug. Applying fix @
    experiments/runs_cache/daniel_test_runs_OFF_BY_ONE/grasp_planning-3/trial_2026-01-12_04-04-13__unstructured__dg_only/24__obj_000041_1__dg_only/output.csv
  ```
   - Mushrooms can slips from grasp at last second, causing the can to be in the air and not detected as grasped nor dropped


2. `experiments/runs_cache/daniel_test_runs_OFF_BY_ONE/grasp_planning-4/trial_2026-01-12_04-04-14__unstructured__dg_only/30__obj_000043_0__dg_only/`

  ```
  ⚠️  Detected second-to-last drop target object bug. Applying fix @
    experiments/runs_cache/daniel_test_runs_OFF_BY_ONE/grasp_planning-4/trial_2026-01-12_04-04-14__unstructured__dg_only/30__obj_000043_0__dg_only/output.csv
  ```
   - Orange yogurt can in gripper but doesn't detect as grasped (could be collision issue)

**_OFF_BY_ONE_daniel_test_runs_2**

This run had thick grippers, and a new MuJoCo state pickle format.

1. `experiments/runs/_OFF_BY_ONE_daniel_test_runs_2/grasp_planning-3/trial_2026-02-17_11-57-09__structured__dg_only/251__obj_000046_0__dg_only/`

  ```
  ⚠️  Detected second-to-last drop target object bug. Applying fix @
    experiments/runs/_OFF_BY_ONE_daniel_test_runs_2/grasp_planning-3/trial_2026-02-17_11-57-09__structured__dg_only/251__obj_000046_0__dg_only/output.csv
  ```
   - Cherry can


2. `experiments/runs/_OFF_BY_ONE_daniel_test_runs_2/grasp_planning-3/trial_2026-02-17_04-32-13__unstructured__dg_only/134__obj_000044_0__dg_only/`

  ```
  ⚠️  Detected second-to-last drop target object bug. Applying fix @
    experiments/runs/_OFF_BY_ONE_daniel_test_runs_2/grasp_planning-3/trial_2026-02-17_04-32-13__unstructured__dg_only/134__obj_000044_0__dg_only/output.csv
  ```
   - Peach can


3. `experiments/runs/_OFF_BY_ONE_daniel_test_runs_2/grasp_planning-3/trial_2026-02-17_04-32-13__unstructured__dg_only/10__obj_000043_0__dg_only/`

  ```
  ⚠️  Detected second-to-last drop target object bug. Applying fix @
    experiments/runs/_OFF_BY_ONE_daniel_test_runs_2/grasp_planning-3/trial_2026-02-17_04-32-13__unstructured__dg_only/10__obj_000043_0__dg_only/output.csv
  ```
   - Orange yogurt can

**run_2-19_18-31-01**

This run has j_user's gripper force changes and normal grippers.

1. `experiments/runs/runs_2-19_18-31-01/grasp_planning-4/trial_2026-02-19_18-31-07__structured__dg_only/223__obj_000044_0__dg_only/`

  ```
  ⚠️  Detected second-to-last drop target object bug. Applying fix @
    experiments/runs/runs_2-19_18-31-01/grasp_planning-4/trial_2026-02-19_18-31-07__structured__dg_only/223__obj_000044_0__dg_only/output.csv
  ```
   - Peach can slipped at last grasp
   ![](assets/run_2-19_18-31-01/1.png)


2. `experiments/runs/runs_2-19_18-31-01/grasp_planning-7/trial_2026-02-19_18-31-11__structured__dg_only/249__obj_000048_0__dg_only/`

  ```
  ⚠️  Detected second-to-last drop target object bug. Applying fix @
    experiments/runs/runs_2-19_18-31-01/grasp_planning-7/trial_2026-02-19_18-31-11__structured__dg_only/249__obj_000048_0__dg_only/output.csv
  ```
   - Green beans can + bowl upside down on edge of table (can is likely clipped/fused into bowl) 
   ![](assets/run_2-19_18-31-01/2.png)


3. `experiments/runs/runs_2-19_18-31-01/grasp_planning-0/trial_2026-02-19_22-00-09__unstructured__dg_only/128__obj_000048_1__dg_only/`

  ```
  ⚠️  Detected second-to-last drop target object bug. Applying fix @
    experiments/runs/runs_2-19_18-31-01/grasp_planning-0/trial_2026-02-19_22-00-09__unstructured__dg_only/128__obj_000048_1__dg_only/output.csv
  ```
   - Green beans can slipped at last grasp
   ![](assets/run_2-19_18-31-01/3.png)


4. `experiments/runs/runs_2-19_18-31-01/grasp_planning-0/trial_2026-02-19_18-31-01__structured__dg_only/249__obj_000048_0__dg_only/`

  ```
  ⚠️  Detected second-to-last drop target object bug. Applying fix @
    experiments/runs/runs_2-19_18-31-01/grasp_planning-0/trial_2026-02-19_18-31-01__structured__dg_only/249__obj_000048_0__dg_only/output.csv
  ```
   - Gripper grabbing the bowl, and green beans is fused to the bowl
   ![](assets/run_2-19_18-31-01/4.png)


5. `experiments/runs/runs_2-19_18-31-01/grasp_planning-6/trial_2026-02-19_18-31-09__structured__dg_only/249__obj_000048_0__dg_only/`

  ```
  ⚠️  Detected second-to-last drop target object bug. Applying fix @
      experiments/runs/runs_2-19_18-31-01/grasp_planning-6/trial_2026-02-19_18-31-09__structured__dg_only/249__obj_000048_0__dg_only/output.csv
  ```
   - Gripper grabbing bowl + target green beans (considered failure because there were multiple objects grasped)
   ![](assets/run_2-19_18-31-01/5.png)

**run_locked**

1. `experiments/runs/runs_locked/grasp_planning-0/trial_2026-02-24_00-48-48__structured__dg_only/198__obj_000041_0__dg_only/`

  ```
  ⚠️  Detected second-to-last drop target object bug. Applying fix @
      /home/atlinx/projects/lab_ws/src/lab_vbnpm/experiments/runs/runs_locked/grasp_planning-0/trial_2026-02-24_00-48-48__structured__dg_only/198__obj_000041_0__dg_only/output.csv
  ```
   - Grips two objects fused together, one of which is the target object.
   - Entire grasp is considered a failure, therefore the experiment continues
   ![](assets/runs_locked/1.png)

1. `experiments/runs/runs_locked/grasp_planning-1/trial_2026-02-24_00-48-49__structured__dg_only/828__obj_000168_0__dg_only/`

  ```
  ⚠️  Detected second-to-last drop target object bug. Applying fix @
      /home/atlinx/projects/lab_ws/src/lab_vbnpm/experiments/runs/runs_locked/grasp_planning-1/trial_2026-02-24_00-48-49__structured__dg_only/828__obj_000168_0__dg_only/output.csv
  ```
   - Grips two objects fused together
   - Entire grasp is considered a failure, therefore the experiment continues
   ![](assets/runs_locked/2.png)