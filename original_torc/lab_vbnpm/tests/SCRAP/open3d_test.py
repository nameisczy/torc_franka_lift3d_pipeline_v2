import trimesh as tm
import open3d as o3d
import numpy as np
import time
from utils.visual_utils import vis_normals

# --- 1. Data Setup ---
# Create a large synthetic point cloud (e.g., 5 million points)
N_POINTS = 40_000
points = np.random.rand(N_POINTS, 3).astype(np.float32)
# Apply a slight curve to ensure non-planar neighborhood for normals
points[:, 2] = np.sin(points[:, 0] * 10) * 0.1
# visualize points with trimesh
pcd_tm = tm.PointCloud(points)
pcd_tm.show()

print(f"Total points: {N_POINTS}\n")

# --- 1. Trimesh mesh normals for PCD ---
print("Starting Trimesh mesh generation and proximity search...")
start_time_tm = time.time()

mesh = tm.voxel.ops.points_to_marching_cubes(points, pitch=0.01)
# mesh = pcd_tm.convex_hull
# near_faces = tm.proximity.nearby_faces(mesh, points)
_close, _dist, triangles = tm.proximity.closest_point(mesh, points)
normals, _valid = tm.triangles.normals(mesh.triangles)
normals = [normals[t] for t in triangles]

end_time_tm = time.time()
print(f"Trimesh Time: {end_time_tm - start_time_tm:.4f} seconds")
input("Press Enter to visualize...")
mesh.show()
normals = np.asarray(normals)
vis_normals(points,normals, 0.02).show()
input("Press Enter to continue to Open3D normal estimation...")

# --- 2. CPU-Based (Legacy API) Normal Estimation ---
# Convert NumPy array to the legacy Open3D PointCloud object
pcd_cpu = o3d.geometry.PointCloud()
pcd_cpu.points = o3d.utility.Vector3dVector(points)

print("Starting CPU (Legacy) Normal Estimation...")
start_time_cpu = time.time()

# Perform normal estimation (CPU-based implementation)
pcd_cpu.estimate_normals()
pcd_cpu.orient_normals_consistent_tangent_plane(30, 10, 0.5)
pcd_cpu.normalize_normals()

end_time_cpu = time.time()
print(f"CPU Time: {end_time_cpu - start_time_cpu:.4f} seconds")

#visualize normals with trimesh
# normals = np.asarray(pcd_cpu.normals)
# vis_normals(points,normals,0.02).show()

# --- 3. GPU-Based (Tensor API) Normal Estimation ---
# Define the target device (will default to CPU if CUDA is not available)
device = o3d.core.Device('CUDA:0')
print(f"Using device: {device}\n")

# Convert NumPy array to the tensor-based Open3D PointCloud object on the device
pcd_gpu = o3d.t.geometry.PointCloud(
    o3d.core.Tensor(points, dtype=o3d.core.float32, device=device)
)

print("Starting GPU (Tensor) Normal Estimation...")
start_time_gpu = time.time()

# Perform normal estimation (accelerated by the Tensor API)
# The estimate_normals function on the tensor object will use the device it is on.
pcd_gpu.estimate_normals()
pcd_gpu.orient_normals_consistent_tangent_plane(30, 10, 0.5)
pcd_gpu.normalize_normals()
# Wait for the GPU to finish execution (synchronization)
if device.get_type() == o3d.core.Device.DeviceType.CUDA:
    o3d.core.cuda.synchronize()

end_time_gpu = time.time()
print(f"GPU Time: {end_time_gpu - start_time_gpu:.4f} seconds")
