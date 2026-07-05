import time
import math
import torch
import pytorch_kinematics as pk

chain = pk.build_serial_chain_from_urdf(
    open("../robots/motoman/curobo/motoman.urdf").read(), "motoman_right_ee"
)
print(chain)
th = torch.tensor(
    [0.0, 0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0]
)
# (1,6,7) tensor, with 7 corresponding to the DOF of the robot
t0 = time.time()
J = chain.jacobian(th)
print("Time taken: ", time.time() - t0)
print(J.shape)
print(chain.get_joint_parent_frame_names())
print([j.name for j in chain.get_joints()])
m = chain.forward_kinematics(th).cuda().get_matrix()
pos = m[0, :3, 3]
rot = pk.matrix_to_quaternion(m[0, :3, :3])
print(pos)
print(rot)
