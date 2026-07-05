# Hard-coded computation of transformation from end-effector to base of end-effector

import transformations as tf
import numpy as np

pose_ee = tf.quaternion_matrix(np.array([-0.20565503,  0.67810245,  0.21544047,  0.67191404]))
pose_ee[:3, 3] = [ 0.89353834, -0.02177764,  1.12317152]

pose_ee_base = tf.quaternion_matrix(np.array([ 0.20565274, -0.67810063, -0.21544008, -0.6719167 ]))
pose_ee_base[:3, 3] = [0.95622154, 0.02153473, 1.12221966]

# A = world
# B = end-effector
# C = end-effector base

# BC = BA AC 
# transformation = np.linalg.inv(pose_ee) @ pose_ee_base
transformation = np.linalg.inv(pose_ee_base) @ pose_ee
print(transformation)