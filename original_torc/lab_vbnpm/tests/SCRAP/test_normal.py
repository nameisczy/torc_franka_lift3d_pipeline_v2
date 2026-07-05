import os
import numpy as np
import open3d as o3d
import trimesh as tm

p = np.load('./points_test.npy')
pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(p)
pcl.estimate_normals()
# o3d.visualization.draw_geometries([pcl], point_show_normal=True)

points = np.asarray(pcl.points)
normals = np.asarray(pcl.normals)
tpcl = tm.points.PointCloud(points,colors=[255,0,0])

# Visualizing normals using lines in Trimesh
# For each point, create a line that extends from the point along the normal direction
lines = []
for i, point in enumerate(points):
    # Starting point is the original point, and the endpoint is along the normal vector
    line_start = point
    line_end = point + 0.1 * normals[i]  # Scale the normal vector
    lines.append([line_start, line_end])

# Convert the lines to a Trimesh object for visualization
lines_mesh = tm.load_path(np.array(lines))

tm.Scene([tpcl, lines_mesh]).show()
