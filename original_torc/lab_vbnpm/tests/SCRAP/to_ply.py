import sys
import time
import copy
import glob
import numpy as np
import trimesh as tm
import open3d as o3d

p_name = sys.argv[1]
p = o3d.io.read_point_cloud(p_name)

# show original points
points = np.array(p.points)
colors = np.array(p.colors)
cloud = tm.points.PointCloud(points, colors)
cloud.show()

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors)

if input('Save?? (y/n): ') == 'y':
    o3d.io.write_point_cloud(sys.argv[2], p)
