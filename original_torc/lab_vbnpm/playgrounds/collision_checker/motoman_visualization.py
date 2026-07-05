import time

import mujoco
import mujoco.viewer

import numpy as np

from tracikpy import TracIKSolver

#  xpos: array([ 0.89353834, -0.02177764,  1.12317152])
#  xquat: array([-0.20565503,  0.67810245,  0.21544047,  0.67191404])
# pose of end-effector with respect to the world

motoman_right_arm = [
    "arm_right_joint_1_s",
    "arm_right_joint_2_l",
    "arm_right_joint_3_e",
    "arm_right_joint_4_u",
    "arm_right_joint_5_r",
    "arm_right_joint_6_b",
    "arm_right_joint_7_t",
]

motoman_left_arm = [
    "arm_left_joint_1_s",
    "arm_left_joint_2_l",
    "arm_left_joint_3_e",
    "arm_left_joint_4_u",
    "arm_left_joint_5_r",
    "arm_left_joint_6_b",
    "arm_left_joint_7_t",
]

motoman_both_arms = motoman_left_arm + motoman_right_arm

motoman_left_arm += ["torso_joint_b1"]
motoman_right_arm += ["torso_joint_b1"]
motoman_both_arms += ["torso_joint_b1"]

def get_objq_indices(model, obj_name):
    jnt = model.joint(model.body(obj_name).jntadr[0])
    qpos_inds = np.array(range(jnt.qposadr[0], jnt.qposadr[0] + len(jnt.qpos0)))
    return qpos_inds


def get_qpos_indices(model, joints=motoman_both_arms):
    qpos_inds = np.array([model.joint(j).qposadr[0] for j in joints])
    return qpos_inds


def get_ctrl_indices(model, joints=motoman_both_arms, repl=''):
    ctrl_inds = np.array([model.actuator(j.replace('joint_', repl)).id for j in joints])
    return ctrl_inds


def get_act_indices(model, joints=motoman_both_arms, repl='v'):
    ctrl_inds = np.array(
        [model.actuator(j.replace('joint', repl)).actadr[0] for j in joints]
    )
    return ctrl_inds


def get_joint_values(mdata, joint_names):
    # this assumes that all joints are revolute joints
    joint_val_dict = {}
    for name in joint_names:
        joint = mdata.joint(name)
        joint_val_dict[name] = joint.qpos[0]
    return joint_val_dict


def set_joint_values(mdata, joint_val_dict):
    # this assumes that all joints are revolute joints
    for name, val in joint_val_dict.items():
        joint = mdata.joint(name)
        joint.qpos[0] = val


def set_joint_values_list(mdata, joint_inds, joint_vals):
    mdata.qpos[joint_inds] = joint_vals

m = mujoco.MjModel.from_xml_path('src/lab_vbnpm/tests/ycb_non_perishables.xml')
d = mujoco.MjData(m)

# print(m.body())

m.opt.timestep = .002
m.opt.impratio = 15

ik_solver = TracIKSolver("src/motoman/motoman_sda10f_moveit_config/config/gazebo_motoman_sda10f.urdf", "base_link", "motoman_right_ee")

pose = np.array([
    [0.00422704, 0.56854313, 0.82264259, 0.95622154],
    [0.01581709, -0.82258504, 0.56842208, 0.02153473],
    [0.99986597, 0.01060907, -0.0124698, 1.12221966],
    [0., 0., 0., 1.]
])

qpos_inds = get_qpos_indices(m, motoman_right_arm)
ctrl_inds = get_ctrl_indices(m, motoman_right_arm)

inds = get_qpos_indices(m, ['torso_joint_b1', 'arm_left_joint_1_s', 'arm_left_joint_2_l', 'arm_left_joint_3_e', 'arm_left_joint_4_u', 'arm_left_joint_5_r', 'arm_left_joint_6_b', 'arm_left_joint_7_t', 'arm_right_joint_1_s', 'arm_right_joint_2_l', 'arm_right_joint_3_e', 'arm_right_joint_4_u', 'arm_right_joint_5_r', 'arm_right_joint_6_b', 'arm_right_joint_7_t'])

q = None
for i in range(100):
    q = ik_solver.ik(pose)
    if q is not None:
        break
print(q)
if q is not None:
    print("Setting qpos")
    d.qpos[qpos_inds] = q

q_test = [1.3906658537451002, 1.7499290108899863, 0.800268621773491, -1.7827558987548395e-05, -0.6600173692363857, 4.65753268001006e-09, -1.0747483358576621e-06, -6.286166352698305e-13, 1.6884221421735908, 0.9414986616373716, -0.2033344224865124, 1.5697934324254554, -1.986735789045772, 1.0936984808699837, -2.4089230056234903]
# q_test = [1.2305123324099683, 1.7499290108899863, 0.800268621773491, -1.7827558987548395e-05, -0.6600173692363857, 4.65753268001006e-09, -1.0747483358576621e-06, -6.286166352698305e-13, -2.117272787827582, -1.0004309776336449, -0.053102234746897826, -1.685149816444337, 0.7839548222597807, 1.4261827059497607, -2.6058842014423664]
# q_test = [0.27577755322768255, 1.7499290108899863, 0.800268621773491, -1.7827558987548395e-05, -0.6600173692363857, 4.65753268001006e-09, -1.0747483358576621e-06, -6.286166352698305e-13, 0.002689504288957692, 0.9300854833209051, 0.46748737124221745, -1.2272871282925715, -0.8503449586764327, 0.4913571637638276, 1.0675004174532565]
d.qpos[inds] = q_test

mujoco.mj_step1(m, d)

print(d.body())

# set_joint_values_list(d, get_qpos_indices(m, motoman_right_arm), q)

with mujoco.viewer.launch_passive(m, d) as viewer:
  start = time.time()
  while viewer.is_running():
    step_start = time.time()
    mujoco.mj_step1(m, d)
    viewer.sync()
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)