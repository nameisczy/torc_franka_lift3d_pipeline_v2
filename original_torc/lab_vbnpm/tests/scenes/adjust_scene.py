import sys
import mujoco
sys.path.append('..')
try:
    from mjcf_robot_scene_integrator import stabilize
except ModuleNotFoundError:
    from scripts.execution_scene.mjcf_robot_scene_integrator import stabilize

## stabilize model ##
model = stabilize(sys.argv[1], True)

## save file ##
mujoco.mj_saveLastXML(sys.argv[2], model)
