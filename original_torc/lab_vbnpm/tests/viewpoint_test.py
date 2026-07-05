import sys
import cv2
import time
import trimesh
import numpy as np
import open3d as o3d
import transformations as tf
from scipy.spatial import KDTree
from tracikpy import TracIKSolver

import rospy
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from trajectory_msgs.msg import JointTrajectory
from moveit_commander import conversions as conv

from utils import conversions as conv2
from grasp_planner.grasp_planner import GraspPlanner
from task_planner.open_loop import update_and_get_points
from task_planner.eutils import ee_open, ee_close, execute
from perception.perception_fast import PerceptionInterface

br = tf2_ros.TransformBroadcaster()


def vis_view(pose):
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "world"
    t.child_frame_id = "TEST"
    t.transform.translation.x = pose.position.x
    t.transform.translation.y = pose.position.y
    t.transform.translation.z = pose.position.z
    t.transform.rotation.x = pose.orientation.x
    t.transform.rotation.y = pose.orientation.y
    t.transform.rotation.z = pose.orientation.z
    t.transform.rotation.w = pose.orientation.w
    br.sendTransform(t)


def viewpoint_test(is_sim, gt, object_to_grasp, pcd_path):
    from task_planner.motoman import MotomanSDA10F
    robot = MotomanSDA10F(is_sim, gt)
    perception = robot.init_perception_interface()
    planner = robot.init_motion_planner()

    ## compile ##
    t0 = time.time()
    PerceptionInterface.raycast(
        1.0,
        np.zeros(3),
        np.zeros((40, 40, 40), dtype=np.float32),
        np.eye(3),
        np.zeros(3),
        1.0,
        1.0,
        1.0,
        1.0,
        0,
        1,
        0,
        1,
    )
    print('Compile time: ', time.time() - t0)

    result = update_and_get_points(
        object_to_grasp,
        robot,
        perception,
        camera_inds=[0, 1],
        save_debug=False,
        debug_number=0,
    )
    points, tgt_pts, tgt_rgb, all_pts, all_rgb, ds_all = result

    radius = 0.5 * (max(tgt_pts[:, 2]) - min(tgt_pts[:, 2]))  # + 0.28
    # radii = [radius + 0.28]
    radii = np.linspace(0.28, 0.56, 2) + radius
    angles = [15, 30, 45, 60, 75]
    views = PerceptionInterface.sample_views(center, radii, angles)
    print('Num views: ', len(views))
    scores = perception.get_info_gain2(robot.camera[1], views)
    print('Score: ', scores)

    for view, score in zip(views, scores):
        print(score)
        vis_view(view)
        input('Next')


def viewpoint_test2(is_sim, gt, object_to_grasp, pcd_path):
    from task_planner.motoman import MotomanSDA10F
    robot = MotomanSDA10F(is_sim, gt)
    perception = robot.init_perception_interface()
    planner = robot.init_motion_planner()
    grasp_planner = GraspPlanner()

    result = update_and_get_points(
        object_to_grasp,
        robot,
        perception,
        camera_inds=[0, 1],
        save_debug=False,
        debug_number=0,
    )
    points, tgt_pts, tgt_rgb, all_pts, all_rgb, ds_all = result
    planner.set_planning_scene(
        points=all_pts,
        update_moveit=True,
    )
    color_mask = np.concatenate(
        [
            # [255, 255, 255] * np.ones_like(surface_points),
            # self.np_int_to_rgba(range(len(occluded_points)))[:, 1:] / 255
            [1, 0, 0] * np.ones_like(points),
            [0, 0, 1] * np.ones((len(all_pts) - len(points), 3)),
        ]
    )

    cam_intr = np.array(perception.cam_info[robot.camera[1]].K).reshape((3, 3))

    center = np.mean(tgt_pts, axis=0)

    radius = 0.5 * (max(tgt_pts[:, 2]) - min(tgt_pts[:, 2]))  # + 0.28
    # radii = [radius + 0.22]
    radii = np.linspace(0.28, 0.56, 2)
    angles = [15, 30, 45, 60, 75]

    for radius in radii:
        views = PerceptionInterface.sample_views(center, [radius], angles)
        print('Num views: ', len(views))

        ik = TracIKSolver(robot.urdf, "base_link", "camera_arm_link")
        valid_views = []
        t0 = time.time()
        for view in views:
            # vis_view(view)
            # input('Next')
            pose0 = conv2.pose_to_matrix(view)
            for i, j, k, l in [
                [1., 1., 0., 0.],
                [-1, -1, 0., 0.],
                [0., 0., -1, 1.],
                [0., 0., 1., -1],
            ]:
                pose1 = np.matmul(
                    pose0,
                    [
                        [k, 0, j, 0],
                        [i, 0, l, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                    ],
                )
                ee_pose = conv2.matrix_to_pose(pose1)
                if ik.ik(pose1) is not None:
                    # vis_view(view)
                    # print(grasp_planner.ik_collision(ee_pose, 'camera_arm_link',True))
                    # input('valid?')
                    valid_views.append(view)
                    break
        print('Ik time: ', time.time() - t0)
        print('Num valid views: ', len(valid_views))

        c_valid_views = []
        t1 = time.time()
        for view in valid_views:
            pose0 = conv2.pose_to_matrix(view)
            for i, j, k, l in [
                [1., 1., 0., 0.],
                [-1, -1, 0., 0.],
                [0., 0., -1, 1.],
                [0., 0., 1., -1],
            ]:
                pose1 = np.matmul(
                    pose0,
                    [
                        [k, 0, j, 0],
                        [i, 0, l, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                    ],
                )
                ee_pose = conv2.matrix_to_pose(pose1)
                if not grasp_planner.ik_collision(ee_pose, 'camera_arm_link',
                                                  False):
                    c_valid_views.append(view)
                    break
                # vis_view(view)
                # input('rotate')
        print('BioIk time: ', time.time() - t1)
        print('Num Collision Valid views: ', len(c_valid_views))
        if len(c_valid_views) > 0:
            break

    t1 = time.time()
    scores = PerceptionInterface.get_info_gain(
        points, all_pts, cam_intr, c_valid_views
    )
    print('Score time: ', time.time() - t1)

    print('Total time: ', time.time() - t0)

    for view, score in zip(c_valid_views, scores):
        vis_view(view)
        img = PerceptionInterface.project_point_cloud_to_image(
            all_pts, color_mask, cam_intr, view, visualize=True
        )
        print(score)
        # cv2.imshow('test', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    is_sim = sys.argv[1][0] not in (
        '0',
        'r',
        'R',
        'n',
        'N',
    ) if len(sys.argv) > 1 else True
    gt = sys.argv[2][0] not in (
        '0',
        'r',
        'R',
        'n',
        'N',
    ) if len(sys.argv) > 2 else True
    ## select object from argument ##
    object_to_grasp = sys.argv[3] if len(sys.argv) > 3 else 'tomato_soup_can'
    # place = [0.6, 0, 1.1, 0.5, -0.5, -0.5, -0.5] if len(sys.argv) > 4 else None
    pcd_path = sys.argv[4] if len(sys.argv) > 4 else None

    rospy.init_node("viewpoint_test")
    viewpoint_test2(is_sim, gt, object_to_grasp, pcd_path)
