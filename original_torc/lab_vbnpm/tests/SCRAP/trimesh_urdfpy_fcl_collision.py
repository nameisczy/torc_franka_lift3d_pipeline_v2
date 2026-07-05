import time
import trimesh
import numpy as np
from urdfpy import URDF
import transformations as tf

import seaborn as sb
import matplotlib.pyplot as plt

t0 = time.time()
moto = URDF.load("../robots/motoman/motoman_mujoco.urdf")
links = [
    'torso_base_link',
    'torso_link_b1',
    'arm_left_link_1_s',
    'arm_left_link_2_l',
    'arm_left_link_3_e',
    'arm_left_link_4_u',
    'arm_left_link_5_r',
    'arm_left_link_6_b',
    'arm_left_link_7_t',
    'arm_right_link_1_s',
    'arm_right_link_2_l',
    'arm_right_link_3_e',
    'arm_right_link_4_u',
    'arm_right_link_5_r',
    'arm_right_link_6_b',
    'arm_right_link_7_t',
]
links = list(moto.link_fk(use_names=True, links=links).keys())  # sort links
geoms = moto.collision_trimesh_fk(links=links)
scene = trimesh.scene.Scene()
for link, geometry_transform in zip(links, geoms.items()):
    geometry, transform = geometry_transform
    scene.add_geometry(geometry, node_name=link, transform=transform)
cmanager, cobjects = trimesh.collision.scene_to_collision(scene)
box = trimesh.primitives.Box(
    [0.6, 1.6, 1], transform=tf.translation_matrix([1, 0, 0.5])
)


def test_collision(joints):
    transforms = moto.link_fk(joints, use_names=True, links=links)
    for link, transform in transforms.items():
        cmanager.set_transform(link, transform)
    return cmanager.in_collision_single(box)


t1 = time.time()
print('init_sim: ', t1 - t0)
# scene.show()

# * generate a random joint angle within the range
ll = moto.joint_limits[:, 0]
ul = moto.joint_limits[:, 1]
print(ll)
print(ul)

num_samples = 1000
rand_joints = np.random.uniform(ll, ul, size=[num_samples] + list(ll.shape))

t0 = time.time()
total_times = []
collisions = []
for i in range(num_samples):
    #     print('random joint: ', rand_joint)
    start_time_i = time.time()
    col = test_collision(rand_joints[i])
    duration_i = time.time() - start_time_i
    total_times.append(duration_i)
    collisions.append(col)

print('done: ', time.time() - t0)
total_times = np.array(total_times)
collisions = np.array(collisions).astype(bool)
# print(total_times, collisions)
# * draw a statistics of the total time

plt.figure()
sb.boxplot(total_times)
plt.savefig('total_timing_boxplot.png')
print('collision timing')
plt.figure()
sb.boxplot(total_times[collisions])
plt.savefig('collision_timing_boxplot.png')
print('non-collision timing')
plt.figure()
sb.boxplot(total_times[collisions & 0])
plt.savefig('non_collision_timing_boxplot.png')
# number of collisions
print('number of collisions: ', collisions.astype(int).sum() / len(collisions))
