import sys
import numpy as np
import trimesh as tm

import rospy

i = sys.argv[1] if len(sys.argv) > 1 else ''
t = np.load(f'/tmp/target_points{i}.npy')
a = np.load(f'/tmp/visible_points{i}.npy')

pt = tm.PointCloud(t, [255, 0, 0])
pa = tm.PointCloud(a, [0, 0, 0, 255 / 2])

tm.scene.Scene([pa, pt]).show()
