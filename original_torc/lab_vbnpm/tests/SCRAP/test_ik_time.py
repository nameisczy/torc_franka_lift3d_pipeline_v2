#!/usr/bin/env python

import time
import numpy as np
from roboticstoolbox.robot.Robot import Robot
from spatialmath import SE3

from rospkg import RosPack

from tracikpy import TracIKSolver
import pyikfast_motoman_right as ikf


class Motoman(Robot):

    def __init__(self):

        rp = RosPack()
        urdf = rp.get_path('lab_vbnpm')
        urdf += '/robots/motoman/curobo/motoman.urdf'
        links, name, urdf_string, urdf_filepath = self.URDF_read(urdf)
        self.urdf = urdf

        print([l.name for l in links])
        print(len(links))

        super().__init__(
            links,
            name=name,
            manufacturer="Yaskawa",
            gripper_links=links[-4],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        self.qz = np.zeros(8)

        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    rob = Motoman()
    tik = TracIKSolver(rob.urdf, "base_link", "motoman_right_ee")
    for link in rob.grippers[0].links:
        print(link)

    rob = rob.ets()

    Tep = rob.fkine([0.1, -0.3, 0.2, -2.2, 0, 2, 0.7, -1])
    Tep2 = tik.fk([0.1, -0.3, 0.2, -2.2, 0, 2, 0.7, -1])
    translation, rotation = ikf.forward([0.1, -0.3, 0.2, -2.2, 0, 2, 0.7, -1])
    print(Tep, Tep2)
    print(translation, np.reshape(rotation,(3,3)))
    print(translation, rotation)
    translation = [0.5, 0.5, 0.5]
    rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1]

    print(rob)

    # t0 = time.time()
    # c = 0
    # for i in range(1000):
    # s0 = ikf.inverse(translation, rotation)
    # print("IKF: ", time.time() - t0)
    # print(s0)

    t0 = time.time()
    c = 0
    for i in range(1000):
        s0 = tik.ik(Tep2)
        c += s0 is not None
    print("TIK: ", time.time() - t0, c)

    t0 = time.time()
    c = 0
    for i in range(1000):
        s0 = rob.ik_LM(Tep)
        c += bool(s0[1])
    print("LM: ", time.time() - t0, c)

    t0 = time.time()
    c = 0
    for i in range(1000):
        s0 = rob.ik_GN(Tep)
        c += bool(s0[1])
    print("GN: ", time.time() - t0, c)

    t0 = time.time()
    c = 0
    for i in range(1000):
        s0 = rob.ik_NR(Tep)
        c += bool(s0[1])
    print("NR: ", time.time() - t0, c)
