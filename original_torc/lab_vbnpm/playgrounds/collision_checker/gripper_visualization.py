import time

import mujoco
import mujoco.viewer

import numpy as np

from tracikpy import TracIKSolver
import transformations as tf

#  xpos: array([0.95622154, 0.02153473, 1.12221966])
#  xquat: array([ 0.20565274, -0.67810063, -0.21544008, -0.6719167 ])
# pose of base of end-effector with respect to world

# we have the desired pose of the end-effector. we want the base of the end-effector. thus, we need to invert the end-effector pose

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

m = mujoco.MjModel.from_xml_path('src/lab_vbnpm/robots/ycb_non_perishables_gripper_test.xml')
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

if True:

    EE_OFFSET = np.array([
    [1.00000000e+00, -4.42674501e-06, -5.22695682e-06, 1.69286113e-06],
    [4.42672396e-06, 1.00000000e+00, -4.02737275e-06, 1.03249058e-07],
    [5.22697465e-06, 4.02734961e-06, 1.00000000e+00, -7.61974474e-02],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
    
    EE_ORI_OFFSET = tf.quaternion_matrix([.7071, 0, 0, .7071])

    EE_OFFSET @= EE_ORI_OFFSET

    
    pose @= EE_OFFSET

#    EE_OFFSET = np.array([
#        [1.00000000e+00, 4.42672396e-06, 5.22697465e-06, -1.29457946e-06],
#        [-4.42674501e-06, 1.00000000e+00, 4.02734961e-06, 2.03632196e-07],
#        [-5.22695682e-06, -4.02737275e-06, 1.00000000e+00, 7.61974474e-02],
#        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
#    ])
#
#    pose @= np.linalg.inv(EE_OFFSET)
    # pose @= EE_OFFSET
else: 
    EE_OFFSET = tf.identity_matrix()

    EE_OFFSET[:3, 3] = [0, 0, -.135]

    pose @= EE_OFFSET

# EE_ORI_OFFSET = tf.quaternion_matrix([.7071, 0, 0, .7071])

# pose @= EE_ORI_OFFSET

# qpos_inds = get_qpos_indices(m, motoman_right_arm)
# ctrl_inds = get_ctrl_indices(m, motoman_right_arm)

# inds = get_qpos_indices(m, ['torso_joint_b1', 'arm_left_joint_1_s', 'arm_left_joint_2_l', 'arm_left_joint_3_e', 'arm_left_joint_4_u', 'arm_left_joint_5_r', 'arm_left_joint_6_b', 'arm_left_joint_7_t', 'arm_right_joint_1_s', 'arm_right_joint_2_l', 'arm_right_joint_3_e', 'arm_right_joint_4_u', 'arm_right_joint_5_r', 'arm_right_joint_6_b', 'arm_right_joint_7_t'])

d.joint("floating joint").qpos[3:] = tf.quaternion_from_matrix(pose)
d.joint("floating joint").qpos[:3] = pose[:3, 3]

# d.qpos[inds] = q_test

mujoco.mj_step1(m, d)

print(d.body("robotiq_85_base_link"))

with mujoco.viewer.launch_passive(m, d) as viewer:
  start = time.time()
  while viewer.is_running():
    step_start = time.time()
    mujoco.mj_step1(m, d)
    viewer.sync()
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)