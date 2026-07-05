import os
import sys
import cv2
import numpy as np
import open3d as o3d

from graspclutter6dAPI import GraspClutter6D

def generate_views(N, phi=(np.sqrt(5)-1)/2, center=np.zeros(3, dtype=np.float32), R=1):
    ''' Author: chenxi-wang
    View sampling on a sphere using Febonacci lattices.

    **Input:**

    - N: int, number of viewpoints.

    - phi: float, constant angle to sample views, usually 0.618.

    - center: numpy array of (3,), sphere center.

    - R: float, sphere radius.

    **Output:**

    - numpy array of (N, 3), coordinates of viewpoints.
    '''
    idxs = np.arange(N, dtype=np.float32)
    Z = (2 * idxs + 1) / N - 1
    X = np.sqrt(1 - Z**2) * np.cos(2 * idxs * np.pi * phi)
    Y = np.sqrt(1 - Z**2) * np.sin(2 * idxs * np.pi * phi)
    views = np.stack([X,Y,Z], axis=1)
    views = R * np.array(views) + center
    return views

def viewpoint_params_to_matrix(towards, angle):
    '''
    Author: chenxi-wang

    **Input:**

    - towards: numpy array towards vector with shape (3,).

    - angle: float of in-plane rotation.

    **Output:**

    - numpy array of the rotation matrix with shape (3, 3).
    '''
    axis_x = towards
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    if np.linalg.norm(axis_y) == 0:
        axis_y = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R1 = np.array([[1, 0, 0],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle), np.cos(angle)]])
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2.dot(R1)
    return matrix

def plot_gripper_pro_max(center, R, width, depth, score=1, color=None):
    '''
    Author: chenxi-wang
    
    **Input:**

    - center: numpy array of (3,), target point as gripper center

    - R: numpy array of (3,3), rotation matrix  of gripper

    - width: float, gripper width

    - score: float, grasp quality score

    **Output:**

    - open3d.geometry.TriangleMesh
    '''
    x, y, z = center
    # sharpen the volume for better visualization
    height = 0.003
    finger_width = 0.003
    tail_length = 0.04
    depth_base = 0.02


    
    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = score # red for high score
        color_g = 0
        color_b = 1 - score # blue for low score
    
    left = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    right = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:,0] -= depth_base + finger_width
    left_points[:,1] -= width/2 + finger_width
    left_points[:,2] -= height/2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:,0] -= depth_base + finger_width
    right_points[:,1] += width/2
    right_points[:,2] -= height/2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:,0] -= finger_width + depth_base
    bottom_points[:,1] -= width/2
    bottom_points[:,2] -= height/2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:,0] -= tail_length + finger_width + depth_base
    tail_points[:,1] -= finger_width / 2
    tail_points[:,2] -= height/2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    colors = np.array([ [color_r,color_g,color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper

def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    ''' Author: chenxi-wang
    Create box instance with mesh representation.
    '''
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,depth],
                         [width,0,depth],
                         [0,height,0],
                         [width,height,0],
                         [0,height,depth],
                         [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box

def load_grasps(gc6d_path, object_index):
    grasp_file = f"{gc6d_path}/grasp_label/obj_{object_index:06d}_labels.npz"

    grasp_labels = np.load(grasp_file)

    sampled_points, offsets, scores, collision = grasp_labels['points'], grasp_labels['offsets'], grasp_labels['scores'], grasp_labels['collision']
    num_views, num_angles, num_depths = 300, 12, 4

    views = generate_views(num_views)

    cnt = 0
    point_inds = np.arange(sampled_points.shape[0])
    np.random.shuffle(point_inds)
    grippers = []

    th = 0.5
    max_depth = 0.055
    max_width = 0.085

    poses = []
    
    for point_ind in point_inds:
            target_point = sampled_points[point_ind]
            offset = offsets[point_ind]
            score = scores[point_ind]
            view_inds = np.arange(300)
            np.random.shuffle(view_inds)
            flag = False
            for v in view_inds:
                if flag: break
                view = views[v]
                angle_inds = np.arange(12)
                np.random.shuffle(angle_inds)
                for a in angle_inds:
                    if flag: break
                    depth_inds = np.arange(4)
                    np.random.shuffle(depth_inds)
                    for d in depth_inds:
                        if flag: break
                        angle, depth, width = offset[v, a, d]
                        if depth > max_depth:
                            continue
                        if score[v, a, d] > th or score[v, a, d] < 0 or width > max_width:
                            continue
                        R = viewpoint_params_to_matrix(-view, angle)
                        t = target_point
                        gripper = plot_gripper_pro_max(t, R, width, depth, 1.1-score[v, a, d])
                        grippers.append(gripper)
                        poses.append([R, t])
                        flag = True

    return poses

## INPUT: object index

## OUTPUT: list of grasps for the object

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python load_gc6d_grasps.py <object_index>")
        exit(1)

    object_index = int(sys.argv[1])

    if 'GC6D_ROOT' not in os.environ:
        print('Please set the environment variable GC6D_ROOT (e.g. export GC6D_ROOT=/path/to/GraspClutter6D)')
        exit(0)

    gc6d_path = os.environ['GC6D_ROOT']

    poses = load_grasps(gc6d_path, object_index)

    # print(dir(grippers[0]))

    print(poses)

    # print(grippers)