# Required System Packages:

On Ubuntu systems, run the following:
```
sudo apt update
sudo apt -y install swig libeigen3-dev liborocos-kdl-dev libkdl-parser-dev liburdfdom-dev libnlopt-dev libgraphviz-dev libnlopt-cxx-dev
sudo apt -y install tmux wmctrl libxrender1 libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev xpra patchelf libglfw3-dev libglfw3 libglew-dev
```

# Required ROS Packages:
Refer to the [parent git repository.](https://github.com/gh_user/lab_manipulation_ws)

# Required Python Packages:
Refer to [requirements.txt](https://github.com/MiaoDragon/lab_vbnpm/blob/master/requirements.txt)

# Running the System

## Components to run (see 2mux or launchsim for shortcut to running them all at once):
- **Control.launch**
  - Mujoco
  - Rviz
  - Bioik
- **Segmentation Service** (GSAM or LANGSAM)
- **Grasping** (GPD)
- **Main (Closed Loop)**

## Functionality:
Main calls various pipelines and processes, including:
- **Perception Pipeline**
- **Grasping Pipeline**
- **Move to Target**
  - **Motion Planning**

### **Perception Pipeline**
The perception pipeline continuously runs `update_and_get_points()` in a loop. This function:
1. Uses the provided camera values.
2. Retrieves the point cloud (PCD).
3. Segments the target points using semgentation service.
4. Fuses the point cloud with segmentation voting.
5. Keeps shared memory with updated information for grasping and move_to_target

Additionally, a **Perception Interface Node** runs alongside the perception pipeline. This node:
- Handles multiple callbacks for joint states.
- Processes camera transformations from joint states.
- Updates variables for camera values, making them accessible for further computations.

### **Grasping Pipeline**
Talks to GPD service to create grasps to place in shared memory

### **Motion Planning Pipeline**
1. Prepares MotionGen from curobo and MpcSolver class instances and performs warmup on the cuda_graph
2. Updates these instances with mesh of the point cloud, preparing the mesh asynchronously and updating it when calling the motion planning
3. While no good-enough grasp is found, the visibility rays are moved towards using MPC where we update the visibility rays with collision and the best cone ray before planning
4. We retime the trajectory with Toppra
5. We sleep for some time before starting the next planning so that it uses a more updates scene
