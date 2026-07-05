import os

import numpy as np
import pytorch_kinematics as pk
from tracikpy import TracIKSolver


class MotionPlanner:

    def __init__(self, robot_file, end_effector_links):
        #TODO initialize c++ class object though pybindings
        urdf_str = open(robot_file).read()
        self.ik_for_ees = {}
        self.chains = {}
        base_link = os.environ.get("TORC_CUROBO_BASE_LINK", "base_link")
        for ee in end_effector_links:
            self.ik_for_ees[ee] = TracIKSolver(robot_file, base_link, ee)
            self.chains[ee] = pk.build_serial_chain_from_urdf(urdf_str, ee)

    def reset(self):
        pass

    def set_planning_scene(self, scene_info):
        pass

    def joint_motion_plan(self, start, goal, ee):
        pass

    def pose_motion_plan(self, start, goal, ee, constraints=None):
        if type(goal) is list:
            #TODO multi goal
            pass
        else:
            #TODO single goal
            pass

    def cartesian_motion(self, start, goal, ee):
        # use for inspiration

        T = self.fk(start, ee)
        Td = goal

        gain = 1
        dt = 0.05
        threshold = 1e-3
        arrived = False
        while not arrived:
            # e = axis-aingle difference
            e = np.empty(6)
            e[:3] = Td[:3, -1] - T[:3, -1]
            R = Td[:3, :3] @ T[:3, :3].T
            li = np.array(
                [
                    R[2, 1] - R[1, 2],
                    R[0, 2] - R[2, 0],
                    R[1, 0] - R[0, 1],
                ]
            )
            if base.iszerovec(li):
                # diagonal matrix case
                if np.trace(R) > 0:
                    # (1,1,1) case
                    a = np.zeros((3, ))
                else:
                    a = np.pi / 2 * (np.diag(R) + 1)
            else:
                # non-diagonal matrix case
                ln = base.norm(li)
                a = math.atan2(ln, np.trace(R) - 1) * li / ln
            e[3:] = a

            # get desired ee velocity
            if base.isscalar(gain):
                k = gain * np.eye(6)
            else:
                k = np.diag(gain)
            v = k @ e

            # did ee arrive?
            print('Close?', np.sum(np.abs(e)), threshold)
            arrived = np.sum(np.abs(e)) < threshold
            qd = np.linalg.pinv(self.jacobian(start)) @ v
            env.step(dt)
