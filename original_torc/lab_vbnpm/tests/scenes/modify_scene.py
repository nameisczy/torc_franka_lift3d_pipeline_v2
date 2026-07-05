import time

import mujoco
import mujoco.viewer

path = "tests/scenes/final/unstructured_14.xml"

m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)


try:
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(m, d)
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)

            print("Corn: ", d.xpos[m.body("obj_000047_0").id], d.xquat[m.body("obj_000047_0").id])
            print("BBQ: ", d.xpos[m.body("obj_000066_0").id], d.xquat[m.body("obj_000066_0").id])

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
except Exception as e:
    pass

mujoco.mj_saveLastXML("out.xml", m)