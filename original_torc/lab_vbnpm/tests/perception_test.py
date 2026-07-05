import sys
import copy
import time
import heapq
import multiprocessing as mp
from multiprocessing import Process, Pipe, shared_memory

import numpy as np
import trimesh as tm
import open3d as o3d
import transformations as tf
from scipy.spatial import KDTree
from tracikpy import TracIKSolver
from scipy.interpolate import CubicHermiteSpline

import rospy
from industrial_msgs.msg import RobotStatus
from sensor_msgs.msg import JointState, PointCloud2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lab_vbnpm.srv import GetNextPlanningPoint
from grasp_planner.grasp_planner import GraspPlanner
from perception.perception_fast import PerceptionInterface
from utils.conversions import pose_to_matrix, matrix_to_pose
from execution_scene.motoman_interface import MotomanInterface as MI
from task_planner.open_loop import update_and_get_points, plan_place
from task_planner.eutils import ee_open, ee_close, execute, wait_till

import mujoco
import random


def perception_pipeline(object_to_grasp, robot, gt, eval_xml_file="nothing"):
    # rospy.init_node("planning_perception_child")
    f = open("perception_time.txt", "w")
    seg_eval_f = open("segmentation_evaluation.txt", "w")

    # [j_user] For visualizing gt point cloud
    debug_gt_pub = rospy.Publisher(
        "/debug/gt_object_points", PointCloud2, queue_size=3, latch=True
    )
    debug_target_pub = rospy.Publisher(
        "/debug/target_points", PointCloud2, queue_size=3, latch=True
    )

    ## init vars ##
    perception = robot.init_perception_interface()
    iters = 0
    cam_inds = [0, 1]
    start_time = time.time()
    original_object_to_grasp = object_to_grasp
    if eval_xml_file != "nothing":
        mj_model = mujoco.MjModel.from_xml_path(
            eval_xml_file
        )  # This sometimes breaks so just do it once
    total_pts = np.array([]).reshape(0, 3)
    total_rgb = np.array([]).reshape(0, 3)
    while not rospy.is_shutdown():
        print("perception running", object_to_grasp)
        # print('Perception Start', flush=True)

        ## simulate noisy detection perception module when we are using gt##
        if gt == True and iters % 5 == 0 and iters != 0:
            object_to_grasp = 35 + random.randint(1, 3)
            print("[j_user DEBUG random object simulating noisy detection running]")
        else:
            object_to_grasp = original_object_to_grasp

        for i in cam_inds:
            visible_tuple = perception.get_visible_points(
                robot.camera[i], object_to_grasp, False
            )
            points, colors, target_mask, bg_mask, image_mask_tuple = visible_tuple
            total_pts = np.vstack((total_pts, np.reshape(points, (-1, 3))))
            total_rgb = np.vstack((total_rgb, np.reshape(colors, (-1, 3))))

        if iters % 5 == 0:
            vis_pcd = o3d.geometry.PointCloud()
            vis_pcd.points = o3d.utility.Vector3dVector(total_pts)
            vis_pcd.colors = o3d.utility.Vector3dVector(total_rgb)
            o3d.visualization.draw_geometries([vis_pcd])

        result = update_and_get_points(
            object_to_grasp,
            robot,
            perception,
            camera_inds=cam_inds,
            save_debug=False,
            debug_number=iters,
        )
        iters += 1

        points, tgt_pts, tgt_rgb, all_pts, all_rgb, ds_all = result
        if iters % 5 == 0:
            vis_pcd = o3d.geometry.PointCloud()
            vis_pcd.points = o3d.utility.Vector3dVector(all_pts)
            vis_pcd.colors = o3d.utility.Vector3dVector(all_rgb)
            o3d.visualization.draw_geometries([vis_pcd])

        print("After update and get points")

        # Evaluate point cloud
        if eval_xml_file != "nothing":
            # print("before eval")
            metric_1, metric_2, metric_3, pcd_gt = perception.evaluate_target_pcd(
                mj_model
            )
            seg_eval_f.write(
                f"RMSE:{metric_1}, percent of gt that are matched:{metric_2}, percent of tar too far away:{metric_3}\n"
            )
            print(
                f"RMSE:{metric_1}, percent of gt that are matched:{metric_2}, percent of tar too far away:{metric_3}\n"
            )
            seg_eval_f.flush()

            msg = PerceptionInterface.create_cloud(
                pcd_gt,
                np.linspace([0, 0, 0], [255, 255, 255], len(pcd_gt)),
                255,
                "world",
                rospy.Time.now(),
            )
            debug_gt_pub.publish(msg)

            # print("after eval")
            """
            If I want to be able to run on real data then I can just run the perception pipeline. I can swap in the mujoco RGBD images for the real ones.
            -Way to just run perception pipeline (COMPLETED)
            -Subscribe to the real RGBD images from a rosbag
            -Generate the should-be point cloud (COMPLETED)
                You get surface points of all the points. You can then just filter them for points nearby the mesh. (COMPLETED)
            -Compare the should-be point cloud to the created point cloud (COMPLETED)
            -Make it a better starting point for the object
                How would I even be able to do this? Maybe give some starting point from a point cloud point from the topic. 
            -Make a script to run the tests a bit easier (COMPLETED)
                I need to open everything and the adjust so maybe just make it part of the tmux. (COMPLETED)
            """

        ## visualize perception output ##
        # if iters % 10 == 0:
        #     visualize stuff

        # if len(cam_inds) > 1:
        #     cam_inds.pop(0)
        if result is not None:
            # t0 = time.time()

            # j_user: To prevent crashing due to no target points
            _, tgt_pts, _, _, _, _ = result
            if len(tgt_pts) == 0:
                print("[j_user DEBUG] No target points in perception pipeline")
                pass
            else:
                target_msg = PerceptionInterface.create_cloud(
                    tgt_pts,
                    np.linspace([0, 0, 0], [255, 255, 255], len(tgt_pts)),
                    255,
                    "world",
                    rospy.Time.now(),
                )
                debug_target_pub.publish(target_msg)
        else:
            # print('Perception Failed!', flush=True)
            pass
        perception_time = time.time() - start_time
        f.write(f"{iters}: {perception_time}\n")
        f.flush()
    f.close()
    seg_eval_f.close()


if __name__ == "__main__":
    is_sim = (
        sys.argv[1][0]
        not in (
            "0",
            "r",
            "R",
            "n",
            "N",
        )
        if len(sys.argv) > 1
        else True
    )
    gt = (
        sys.argv[2][0]
        not in (
            "0",
            "r",
            "R",
            "n",
            "N",
        )
        if len(sys.argv) > 2
        else True
    )
    ## select object from argument ##
    object_to_grasp = sys.argv[3] if len(sys.argv) > 3 else "tomato_soup_can"
    eval_xml_file = sys.argv[4] if len(sys.argv) > 4 else "nothing"

    print(
        "Args: is_sim (s or r), gt (g or r), object_to_grasp (id sim or name real), xml_file (path or 'nothing')"
    )

    ## init perception and planning interfaces ##
    rospy.init_node("perception_test")
    t0 = time.time()
    from task_planner.motoman import MotomanSDA10F

    robot = MotomanSDA10F(
        is_sim, gt
    )  # To use the real camera messages, set the first argument is_sim = r
    print("after made robot")
    t1 = time.time()
    # print('Init Time:', t1 - t0)

    perception_pipeline(object_to_grasp, robot, gt, eval_xml_file)
    """
    Sometimes my mujoco won't open but that is because of OOM. If you don't see any error and the process just dies it because of being OOM. Don't run Sam in that case. 
    For boxes 36 is sugar box
    For pile: 38 is going to be a good target to try: python closed_loop.py s g 38 p "../../xmls/adjusted_pos_ycb_boxes.xml"
    35: '001_chips_can',
    36: '003_cracker_box',
    37: '004_sugar_box',
    38: '005_tomato_soup_can',
    39: '006_mustard_bottle',
    40: '008_pudding_box',
    41: '009_gelatin_box',
    42: '010_potted_meat_can'

    The trouble running is maybe to do with the number of objects in ycb_pile1
    """

    # killall -9 -u j_user
