import mujoco
import mujoco.viewer

import time
import numpy as np

from tracikpy import TracIKSolver

import rospy
import rospkg

import transformations as tf

import torch

def sim_configs_mujoco(model, data, poses, qpos_inds, ref_frames=[], tlim=10):
    i = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            if time.time() - start >= tlim:
                i += 1
                if i >= len(poses):
                    viewer.close()
                    time.sleep(1)
                    break
                start = time.time()
            step_start = time.time()
            data.qpos[qpos_inds] = poses[i]
            mujoco.mj_step1(model, data)
            viewer.sync()
            time_until_next_step = model.opt.timestep - \
                (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

def axis_angle_matrix(axis=np.array([0, 0, 1]), angle=None):
    if angle is None:
        angle = 2 * np.pi * np.random.rand()
    quat = np.concatenate([[np.cos(angle)], axis * np.sin(angle)])
    return tf.quaternion_matrix(quat)

def get_objq_indices(obj_name):
    jnt = model.joint(model.body(obj_name).jntadr[0])
    qpos_inds = np.array(
        range(jnt.qposadr[0], jnt.qposadr[0] + len(jnt.qpos0))
    )
    return qpos_inds

def get_qpos_indices(joints):
    qpos_inds = np.array([model.joint(j).qposadr[0] for j in joints])
    return qpos_inds

def get_qvel_indices(joints):
    qvel_inds = np.array([model.joint(j).dofadr[0] for j in joints])
    return qvel_inds

def get_ctrl_indices(joints, prefix='', replace=''):
    ctrl_name = lambda j: prefix + j.replace('_joint', replace)
    ctrl_inds = [model.actuator(ctrl_name(j)).id for j in joints]
    return np.array(ctrl_inds)

def get_act_indices(joints, prefix='', replace=''):
    act_name = lambda j: prefix + j.replace('_joint', replace)
    act_inds = [model.actuator(act_name(j)).actadr[0] for j in joints]
    return np.array(act_inds)
    
def get_jnt_indices(joints):
    jnt_inds = np.array([model.joint(j).id for j in joints])
    return jnt_inds

def compute_ik(solver, pose, lim=5):
    q = solver.ik(pose)

    if q is not None:
        return q
    else:
        for i in range(lim-1):
            q = solver.ik(pose)
            if q is not None:
                break

    return q

rp = rospkg.RosPack()
try:
    lab_vbnpm_path = rp.get_path('lab_vbnpm')
    motoman_sda10f_path = rp.get_path('motoman_sda10f_moveit_config')
except rospkg.common.ResourceNotFound:
    lab_vbnpm_path = "/data/local/kc1317/workspace/src/lab_vbnpm/"
    motoman_sda10f_path = "/data/local/kc1317/workspace/src/motoman/motoman_sda10f_moveit_config/"
urdf = lab_vbnpm_path + '/robots/motoman/curobo/motoman.urdf'

link_0 = "motoman_right_ee"
link_1 = "camera_arm_link"

ik_0 = TracIKSolver(urdf, "base_link", link_0)
ik_1 = TracIKSolver(urdf, "base_link", link_1)

poses = np.load(lab_vbnpm_path + "/valid_poses.npy")
views = np.load(lab_vbnpm_path + "/valid_views.npy")

link_0_pose = ik_0.fk(np.zeros(ik_0.number_of_joints))
link_1_pose = ik_1.fk(np.zeros(ik_1.number_of_joints))

rel_transform = np.linalg.pinv(link_0_pose) @ link_1_pose
rel_transform[np.abs(rel_transform) < 1e-5] = 0

# mjcf = lab_vbnpm_path + '/tests/ycb_02_non_perishables.xml'
mjcf = lab_vbnpm_path + '/tests/ycb_06_crackers_part_occl.xml'
model = mujoco.MjModel.from_xml_path(mjcf)
data = mujoco.MjData(model)
ind = 1

link_0_configs = []
link_1_configs = []
for i, (pose, view) in enumerate(zip(poses, views)):
    q0 = compute_ik(ik_0, pose)
    q1 = compute_ik(ik_1, view)
    # q1 = compute_ik(ik_1, pose @ rel_transform)

    if q0 is None or q1 is None:
        raise KeyboardInterrupt("failure")

    link_0_configs.append(q0)
    link_1_configs.append(q1)

    temp_rel = np.linalg.pinv(pose) @ view
    temp_rel[np.abs(temp_rel) < 1e-5] = 0

    print(i)
    print("success")
    print("observed rel transform\n", temp_rel)
    # print("INV\n", np.linalg.pinv(temp_rel))
    print("true urdf transform\n", rel_transform)

    # transform3 = np.linalg.pinv(rel_transform)
    # transform3[np.abs(transform3) < 1e-5] = 0
    # print(transform3)

temp = np.array([
                [0, -1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, -1, 0.152],
                [0, 0, 0, 1]
            ])
temp[np.abs(temp) < 1e-5] = 0
# print(rel_transform @ np.linalg.pinv(temp))

qpos_inds_0 = get_qpos_indices(ik_0.joint_names)
qpos_inds_1 = get_qpos_indices(ik_1.joint_names)

# sim_configs_mujoco(model, data, link_0_configs, qpos_inds_0)
# sim_configs_mujoco(model, data, link_1_configs, qpos_inds_1)

for q0, q1 in zip(link_0_configs, link_1_configs):
    pose_0_temp = ik_1.fk(q0)
    pose_1_temp = ik_1.fk(q1)

    print(np.max(np.abs(pose_0_temp - pose_1_temp)))

sim_configs_mujoco(model, data, link_0_configs, qpos_inds_0, tlim=2)
sim_configs_mujoco(model, data, link_1_configs, qpos_inds_1, tlim=2)
