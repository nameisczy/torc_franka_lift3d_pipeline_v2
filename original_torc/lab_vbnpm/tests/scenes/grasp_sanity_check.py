from tracikpy import TracIKSolver

import rospkg
import numpy as np

rp = rospkg.RosPack()
lab_path = rp.get_path('lab_vbnpm')
urdf = f"{lab_path}/robots/motoman/curobo/motoman.urdf"

ik_solver = TracIKSolver(urdf, "base_link", "motoman_right_ee")

grasp = np.array([[ 0.06570041, -0.18246917, -0.98101401, 0.77755846],
                  [-0.61124651, -0.78444803, 0.10497143, 0.10941908],
                  [-0.78870855, 0.59274472, -0.1630722, 1.0891233],
                  [0., 0., 0., 1.]])

print(ik_solver.ik(grasp))