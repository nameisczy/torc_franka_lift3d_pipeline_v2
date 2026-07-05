import sys
import numpy as np
import trimesh as tm

import rospy
import resource_retriever as rr
from visualization_msgs.msg import MarkerArray, Marker
from utils.conversions import pose_to_matrix


def save_rviz_marker(msg: MarkerArray, DENSITY):
    # scene = tm.Scene()
    pose_x, pose_y, pose_z = rospy.get_param('/workspace/pose')
    size_x, size_y, size_z = rospy.get_param('/workspace/size')
    offsetxF = 0.1
    offsetxB = 0.01
    offsetyLR = 0.3
    # offsetz = 0.05
    pose_x -= offsetxF
    pose_y -= size_y - offsetyLR / 2
    # pose_z += offsetz
    size_x += offsetxF - offsetxB
    size_y -= offsetyLR
    # size_z -= 0.5 * offsetz
    pos_a = [pose_x, pose_y, pose_z]
    pos_b = [pose_x + size_x, pose_y + size_y, pose_z + size_z]
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = np.minimum(pos_a, pos_b)
    vol_bnds[:, 1] = np.maximum(pos_a, pos_b)
    lx, ly = vol_bnds[0, 0], vol_bnds[1, 0]
    hx, hy = vol_bnds[0, 1], vol_bnds[1, 1]
    z_min = vol_bnds[2, 0]
    vol_bnds[2, 1] = vol_bnds[2, 0] + 0.05
    vol_bnds[2, 0] -= 0.5
    print(lx, ly, hx, hy, z_min)
    # num_points = int(DENSITY * (hx - lx) * (hy - ly))
    box = tm.primitives.Box(*tm.bounds.to_extents(vol_bnds.T))
    num_points = int(DENSITY * box.area)
    print(f"Generating {num_points} points for the plane")
    # pts = np.random.uniform(
    #     [lx, ly, z_min], [hx, hy, z_min], size=(num_points, 3)
    # )
    up_inds = np.nonzero(box.face_normals[:, 2] == 1)[0]
    samples = tm.sample.sample_surface_even(box, num_points)
    pts = samples[0][(samples[1] == up_inds[0]) | (samples[1] == up_inds[1])]
    tgt_pts = None
    for marker in msg.markers:
        pose = pose_to_matrix(marker.pose)
        path = rr.get_filename(marker.mesh_resource).replace('file://', '')
        mesh = tm.load_mesh(path)
        mesh.apply_transform(pose)
        # scene.add_geometry(mesh, marker.id)
        num_points = int(DENSITY * mesh.area)
        print(f"Generating {num_points} points for marker {marker.id}")
        p = tm.sample.sample_surface_even(mesh, num_points)[0]
        pts = np.vstack((pts, p))
        if marker.id == 35:
            tgt_pts = p

    # pcd_all = tm.PointCloud(pts)
    # pcd_tgt = tm.PointCloud(tgt_pts, [255, 0, 0])
    # tm.Scene([pcd_all, pcd_tgt]).show()
    return pts, tgt_pts


if __name__ == "__main__":
    rospy.init_node("save_gt_pcd")
    i = sys.argv[1] if len(sys.argv) > 1 else 0
    DENSITY = int(sys.argv[2]) if len(sys.argv) > 2 else 10000  # points per m^2
    path_all = f'/tmp/visible_points{i}.npy'
    path_tgt = f'/tmp/target_points{i}.npy'
    print('Saving point clouds to', path_all, 'and', path_tgt)

    markers = rospy.wait_for_message('/visualization_marker', MarkerArray)
    p,t = save_rviz_marker(markers, DENSITY)
    np.save(path_all, p)
    np.save(path_tgt, t)
