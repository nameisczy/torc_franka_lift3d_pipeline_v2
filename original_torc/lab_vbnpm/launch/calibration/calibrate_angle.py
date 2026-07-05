import sys
import numpy as np
import transformations as tf

# calculate height and distance from angle
theta = np.radians(float(sys.argv[1]))
x = 40.75 - np.sin(theta) * 28.9
z = 68.5 + np.cos(theta) * 28.9
x /= 1000
z /= 1000

# adjust relative to front of mount where it touches the robot
x += 0.14
z += 1.338 - 1.2

quat = tf.quaternion_from_euler(0, theta, 0)

print(f"  x: {x}\n  y: 0.0\n  z: {z}")
print(f"  qx: {quat[1]}\n  qy: {quat[2]}\n  qz: {quat[3]}\n  qw: {quat[0]}")
