import os
import sys
import cv2
import time
import heapq
import trimesh
import numpy as np
import open3d as o3d
from glob import glob
import transformations as tf
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from trajectory_msgs.msg import JointTrajectory
from moveit_commander import conversions as conv
from sensor_msgs.msg import JointState, PointCloud2

from utils import conversions as conv2
from grasp_planner.grasp_planner import GraspPlanner

# from grasp_planner.grasp_planner_collision_checking import GraspPlannerHPPFCL as GraspPlanner
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


def grasping_test(is_sim, gt, object_to_grasp, pcd_paths):
    num_pts_to_sample = 100
    grasp_heap_size = 50

    debug_target_pub = rospy.Publisher(
        "/debug/target_points", PointCloud2, queue_size=3, latch=True
    )
    debug_full_pcd = rospy.Publisher(
        "/debug/full_pcd", PointCloud2, queue_size=3, latch=True
    )
    debug_surface_pcd = rospy.Publisher(
        "/debug/surface_pcd", PointCloud2, queue_size=3, latch=True
    )

    ## init modules ##
    from geometry_msgs.msg import Pose

    Pose.__lt__ = lambda a, b: a.position.z < b.position.z
    from task_planner.motoman import MotomanSDA10F

    robot = MotomanSDA10F(is_sim, gt)
    planner = robot.init_motion_planner(planner="dummy")
    if not pcd_paths:
        perception = robot.init_perception_interface()
    grasp_planner = GraspPlanner()

    # goal = {
    #     "torso_joint_b1": 0,
    #     "arm_left_joint_1_s": 1.75,
    #     "arm_left_joint_2_l": 0.8,
    #     "arm_left_joint_3_e": 0,
    #     "arm_left_joint_4_u": -0.66,
    #     "arm_left_joint_5_r": 0,
    #     "arm_left_joint_6_b": 0,
    #     "arm_left_joint_7_t": 0,
    #     "arm_right_joint_1_s": 0.8227,
    #     "arm_right_joint_2_l": 0.7819,
    #     "arm_right_joint_3_e": -1.9520,
    #     "arm_right_joint_4_u": -0.5853,
    #     "arm_right_joint_5_r": -1.4359,
    #     "arm_right_joint_6_b": -1.6822,
    #     "arm_right_joint_7_t": 0.0,
    # }
    # links = robot.pcd_link_dict.keys()
    # print(links)
    # pcd = robot.get_pcd_at_joints(goal, links)
    # trimesh.points.PointCloud(pcd).show()

    ## init vars ##
    grasp_heap = []
    vis_iter = 1
    iters = 0
    while not rospy.is_shutdown() and vis_iter > 0:
        if pcd_paths:
            if iters < len(pcd_paths):
                print(f"Loading pcd {iters}")
                print(pcd_paths[iters])
                pcd_path_t = pcd_paths[iters]
                pcd_path_s = pcd_path_t.replace("target", "surface")
                pcd_path_a = pcd_path_t.replace("target", "all")
                pcd_t = o3d.io.read_point_cloud(pcd_path_t)
                pcd_s = o3d.io.read_point_cloud(pcd_path_s)
                pcd_a = o3d.io.read_point_cloud(pcd_path_a)
                # o3d.visualization.draw_geometries([pcd_t])
                # o3d.visualization.draw_geometries([pcd_s])
                # o3d.visualization.draw_geometries([pcd_a])

                tgt_pts = np.asarray(pcd_t.points)
                tgt_rgb = np.asarray(pcd_t.colors)
                points = np.asarray(pcd_s.points)
                all_pts = np.asarray(pcd_a.points)
                all_rgb = np.asarray(pcd_a.colors)
            else:
                print("Done")
                break
        iters += 1

        while pcd_paths is None:
            ## get latest perception info ##
            result = update_and_get_points(
                object_to_grasp,
                robot,
                perception,
                camera_inds=[0, 1],
                save_debug=False,
                debug_number=0,
            )
            points, tgt_pts, tgt_rgb, all_pts, all_rgb, ds_all = result
            if len(tgt_pts) == 0:
                print(
                    "[j_user DEBUG] tgt_pts in grasping pipeline are zero", len(tgt_pts)
                )
                continue
            else:
                print("[j_user DEBUG] found tgt_pts", len(tgt_pts))
                break

        target_msg = PerceptionInterface.create_cloud(
            tgt_pts,
            np.linspace([0, 0, 0], [255, 255, 255], len(tgt_pts)),
            255,
            "world",
            rospy.Time.now(),
        )
        debug_target_pub.publish(target_msg)
        surface_msg = PerceptionInterface.create_cloud(
            points,
            np.linspace([0, 0, 0], [255, 255, 255], len(points)),
            255,
            "world",
            rospy.Time.now(),
        )
        debug_surface_pcd.publish(surface_msg)
        full_msg = PerceptionInterface.create_cloud(
            all_pts,
            all_rgb * 255,
            1,
            "world",
            rospy.Time.now(),
        )
        debug_full_pcd.publish(full_msg)

        # tgt_mesh = PerceptionInterface.get_shape_estimate(
        #     tgt_pts,
        # )
        # est_tgt_pts = tm.sample.sample_surface(tgt_mesh, 100)[0]
        # est_tgt_pts = np.concatenate([est_tgt_pts, tgt_pts])
        # est_tgt_rgb = np.ones((est_tgt_pts.shape[0], 3))

        t0 = time.time()
        num_samples = min(num_pts_to_sample - len(grasp_heap), len(tgt_pts))
        indices = np.random.choice(len(tgt_pts), num_samples, replace=False)
        points_to_sample = np.concatenate(
            [
                np.array(tgt_pts)[indices],
                np.reshape([g[3] for g in grasp_heap], (-1, 3)),
            ]
        )
        colors_to_sample = np.concatenate(
            [
                np.array(tgt_rgb)[indices],
                np.zeros((len(grasp_heap), 3)),
            ]
        )
        grasps, pre_grasps, scores, samples = grasp_planner.get_grasp_poses(
            points_to_sample,
            colors_to_sample,
            # est_tgt_pts,
            # est_tgt_rgb,
            planner.ik_for_ees[robot.gripper_link].ik,
            collision_voxel_tuple=(points, all_pts, 0.005),
            # octomap_reset=planner.reset_octomap,
            # octomap_set=planner.set_planning_scene,
            # visualize=True,
        )
        t1 = time.time()

        inds = np.array(scores) > 1
        if inds.any():
            grasps = [grasps[i] for i in range(len(grasps)) if scores[i] > 1]
            pre_grasps = [
                pre_grasps[i] for i in range(len(pre_grasps)) if scores[i] > 1
            ]
            samples = [samples[i] for i in range(len(samples)) if scores[i] > 1]
            scores = [scores[i] for i in range(len(scores)) if scores[i] > 1]

        # get best grasp per sample
        best_for_sample = {}
        for i in range(len(grasps)):
            s = samples[i]
            sample = (s.x, s.y, s.z)
            if sample not in best_for_sample:
                best_for_sample[sample] = i
            else:
                if scores[i] > scores[best_for_sample[sample]]:
                    best_for_sample[sample] = i

        # TODO reevalute the grasp scores in the heap
        # if len(grasp_heap) > 0:
        #     for i in range(len(grasp_heap)):
        #         new_score = #TODO
        #         grasp_heap[i] = (new_score, grasp_heap[i][1])
        #     heapq.heapify(grasp_heap,)

        raw_scores = [(x * 10 % 10) / 10 for x in scores]
        # score_type = [int(x) for x in scores]
        # n_scores = list(zip(raw_scores, score_type))
        n_scores = list(zip(scores, raw_scores))
        best_samples = list(best_for_sample.items())

        # fill the heap if empty
        i = 0
        while len(grasp_heap) < grasp_heap_size and i < len(best_for_sample):
            sample, j = best_samples[i]
            heapq.heappush(
                grasp_heap,
                (n_scores[j], grasps[j], pre_grasps[j], sample),
            )
            i += 1

        # pushpop new grasps and maintain heap size
        while i < len(best_for_sample):
            sample, j = best_samples[i]
            heapq.heappushpop(
                grasp_heap,
                (n_scores[j], grasps[j], pre_grasps[j], sample),
            )
            # consider heapq.heapreplace to gauranty new items are considered
            i += 1

        t2 = time.time()

        print(f"Grasp planning time: {t1-t0}")
        print(f"Grasp heap update time: {t2-t1}")
        sorted_grasps = sorted(zip(scores, grasps, pre_grasps), reverse=True)
        # plot histogram of scores
        if iters == vis_iter:
            # plt.hist(scores, bins=10, range=(0.2, 0.8))
            # plt.show()
            sorted_heap = sorted(grasp_heap, reverse=True)
            heap_scores = [x[0][0] for x in sorted_heap]
            # plt.hist(heap_scores, bins=10, range=(0.2, 0.8))
            # plt.show()
            f, a = plt.subplots(2, 1)
            a = a.ravel()
            titles = [f"Iter {iters}", "Accumulated"]
            data = [scores, heap_scores]
            for idx, ax in enumerate(a):
                ax.hist(data[idx], bins=20, range=(0.2, 0.8))
                ax.set_title(titles[idx])
                ax.set_xlabel("score")
                ax.set_ylabel("#")
            # plt.tight_layout()
            print(f"Current iter size: {len(scores)}")
            print(f"Grasp heap size: {len(grasp_heap)}")
            plt.show()
            if pcd_paths:
                print(f"Max iters = {len(pcd_paths)}")
            vis_iter = iters + int(input(f"Iter to visualize? {iters} + ?: "))

        # visualize best grasps
        if False:
            i = 0
            while i < min(3, len(sorted_grasps)):
                score, grasp, pre_grasp = sorted_grasps[i]
                print(f"Grasp {i}: {score}")
                vis_view(grasp)
                grasp_planner.ik_collision(grasp, visualize=True)
                input("Pre-grasp...")
                vis_view(pre_grasp)
                grasp_planner.ik_collision(pre_grasp, visualize=True)
                input("Next grasp...")
                i += 1
            # visualize median grasp
            i = len(sorted_grasps) // 2
            score, grasp, pre_grasp = sorted_grasps[i]
            print(f"Grasp {i}: {score}")
            vis_view(grasp)
            grasp_planner.ik_collision(grasp, visualize=True)
            input("Pre-grasp...")
            vis_view(pre_grasp)
            grasp_planner.ik_collision(pre_grasp, visualize=True)
            input("Next grasp...")
            # visualize worst grasps
            i = len(sorted_grasps) - 1
            while i > max(len(sorted_grasps) - 3, 0):
                score, grasp, pre_grasp = sorted_grasps[i]
                print(f"Grasp {i}: {score}")
                vis_view(grasp)
                grasp_planner.ik_collision(grasp, visualize=True)
                input("Pre-grasp...")
                vis_view(pre_grasp)
                grasp_planner.ik_collision(pre_grasp, visualize=True)
                input("Next grasp...")
                i -= 1

    sorted_heap = sorted(grasp_heap, reverse=True)
    # plot histogram of scores
    heap_scores = [x[0][0] for x in sorted_heap]
    plt.hist(heap_scores, bins=20, range=(0.2, 0.8))
    plt.show()
    # visualize best grasps
    i = 0
    while i < min(3, len(sorted_heap)):
        score, grasp, pre_grasp, sample = sorted_heap[i]
        print(f"Grasp {i}: {score}")
        vis_view(grasp)
        grasp_planner.ik_collision(grasp, visualize=True)
        input("Pre-grasp...")
        vis_view(pre_grasp)
        grasp_planner.ik_collision(pre_grasp, visualize=True)
        input("Next grasp...")
        i += 1
    # visualize median grasp
    i = len(sorted_heap) // 2
    score, grasp, pre_grasp, sample = sorted_heap[i]
    print(f"Grasp {i}: {score}")
    vis_view(grasp)
    grasp_planner.ik_collision(grasp, visualize=True)
    input("Pre-grasp...")
    vis_view(pre_grasp)
    grasp_planner.ik_collision(pre_grasp, visualize=True)
    input("Next grasp...")
    # visualize worst grasps
    i = len(sorted_heap) - 1
    while i > max(len(sorted_heap) - 3, 0):
        score, grasp, pre_grasp, sample = sorted_heap[i]
        print(f"Grasp {i}: {score}")
        vis_view(grasp)
        grasp_planner.ik_collision(grasp, visualize=True)
        input("Pre-grasp...")
        vis_view(pre_grasp)
        grasp_planner.ik_collision(pre_grasp, visualize=True)
        input("Next grasp...")
        i -= 1


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
    # place = [0.6, 0, 1.1, 0.5, -0.5, -0.5, -0.5] if len(sys.argv) > 4 else None
    pcd_paths = sys.argv[4:] if len(sys.argv) > 4 else None

    rospy.init_node("grasping_test")
    grasping_test(is_sim, gt, object_to_grasp, pcd_paths)
