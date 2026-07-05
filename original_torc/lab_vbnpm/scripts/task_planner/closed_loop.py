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
import tf2_ros
from industrial_msgs.msg import RobotStatus
from sensor_msgs.msg import CameraInfo, JointState, PointCloud2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TransformStamped, Pose, Point, Quaternion

from lab_vbnpm.srv import GetNextPlanningPoint
from grasp_planner.grasp_planner import GraspPlanner

# from grasp_planner.grasp_planner_collision_checking import GraspPlannerHPPFCL as GraspPlanner
from perception.perception_fast import PerceptionInterface
from utils.conversions import pose_to_matrix, matrix_to_pose
from execution_scene.motoman_interface import MotomanInterface as MI
from task_planner.open_loop import update_and_get_points, plan_place
from task_planner.eutils import ee_open, ee_close, execute, wait_till
import mujoco

SAVE_RECORDING = False
# SAVE_RECORDING = 'video'
# SAVE_RECORDING = 'pcd'
num_pts_to_sample = 1000
grasp_heap_size = 50
lift_height = 0.05
NUM_ITERS = 2
WINDOW = 0.4
shared_arr_names = (
    "pts",
    "tgt_pts",
    "tgt_rgb",
    "all_pts",
    "all_rgb",
    "ds_all",
)
s_shape = (2000000, 3)
s_dtype = np.float32
s_size = np.zeros(s_shape, dtype=s_dtype).nbytes


def copy_content_from_shared_arrays(s_arrays):
    contents = []
    for i in range(len(s_arrays)):
        length = int(s_arrays[i][0][0])
        contents.append(np.array(s_arrays[i][1 : length + 1]))
    return contents


def copy_content_to_shared_arrays(s_arrays, contents):
    for i in range(len(s_arrays)):
        length = len(contents[i])
        # desired_shape = s_arrays[i].shape
        # content_shape = contents[i].shape
        # print('Desired Shape:', desired_shape, flush=True)
        # print('Content Shape:', content_shape, flush=True)
        s_arrays[i][1 : length + 1] = np.reshape(contents[i], (-1, 3))
        s_arrays[i][0][0] = length


import random


def perception_pipeline(
    object_to_grasp, robot, gt, eval_xml_file="nothing", body_name="004_sugar_box"
):
    rospy.init_node("planning_perception_child")
    f = open("perception_time.txt", "w")
    seg_eval_f = open("segmentation_evaluation.txt", "w")

    # [j_user] For visualizing gt point cloud
    debug_gt_pub = rospy.Publisher(
        "/debug/gt_object_points", PointCloud2, queue_size=3, latch=True
    )
    debug_target_pub = rospy.Publisher(
        "/debug/target_points", PointCloud2, queue_size=3, latch=True
    )
    debug_full_pcd = rospy.Publisher(
        "/debug/full_pcd", PointCloud2, queue_size=3, latch=True
    )
    debug_surface_pcd = rospy.Publisher(
        "/debug/surface_pcd", PointCloud2, queue_size=3, latch=True
    )

    ## init shared memory pointers ##
    shared_arrays = []
    shares = []
    for name in shared_arr_names:
        e_shm = shared_memory.SharedMemory(name=name)
        shared_arrays.append(np.ndarray(s_shape, dtype=s_dtype, buffer=e_shm.buf))
        shares.append(e_shm)

    ## init vars ##
    perception = robot.init_perception_interface()
    perception.old_perception = False
    iters = 0
    cam_inds = [0, 1]
    start_time = time.time()
    original_object_to_grasp = object_to_grasp
    if eval_xml_file != "nothing":
        mj_model = mujoco.MjModel.from_xml_path(
            eval_xml_file
        )  # This sometimes breaks so just do it once
    while not rospy.is_shutdown():
        used1 = perception.did_use_cam_pose_before(robot.camera[0])
        used2 = perception.did_use_cam_pose_before(robot.camera[1])
        if used1 and used2:
            continue
        # print('Perception Start', object_to_grasp, flush=True)

        ## simulate noisy detection perception module when we are using gt##
        if gt == True and iters % 5 == 0 and iters != 0:
            pass
            # object_to_grasp = 35 + random.randint(1, 3)
            # print(
            #     "[j_user DEBUG random object simulating noisy detection running]"
            # )
        else:
            object_to_grasp = original_object_to_grasp

        result = update_and_get_points(
            object_to_grasp,
            robot,
            perception,
            camera_inds=cam_inds,
            save_debug=SAVE_RECORDING,
            debug_number=iters,
        )
        iters += 1

        # print("After update and get points")

        if result is not None:
            _, tgt_pts, _, _, _, _ = result

        # Evaluate point cloud
        if eval_xml_file != "nothing" and iters % 2 == 0:
            if len(tgt_pts) == 0:
                print("len tgt_pts is zero so skipping eval", tgt_pts)
            else:
                # print("before eval")
                metric_1, percent_tp, num_tp, percent_fp, num_fp, pcd_gt = (
                    perception.evaluate_target_pcd(mj_model, body_name, tgt_pts)
                )
                precision = num_tp / (num_tp + num_fp) if num_tp > 0 else 0
                seg_eval_f.write(
                    f"RMSE:{metric_1}, percent_tp:{percent_tp}, num_tp:{num_tp}, percent fp:{percent_fp}, num fp: {num_fp}, precision {precision}\n"
                )
                print(
                    f"RMSE:{metric_1}, percent_tp:{percent_tp}, num_tp:{num_tp}, percent fp:{percent_fp}, num fp: {num_fp}\n"
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

            copy_content_to_shared_arrays(shared_arrays, result)
            # j_user: To prevent crashing due to no target points
            surface_pts, tgt_pts, _, all_pts, _, _ = result
            if len(tgt_pts) == 0:
                # print("[j_user DEBUG] No target points in perception pipeline")
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
                # print("Target pts published", len(tgt_pts))
                full_msg = PerceptionInterface.create_cloud(
                    all_pts,
                    np.linspace([0, 0, 0], [255, 255, 255], len(all_pts)),
                    255,
                    "world",
                    rospy.Time.now(),
                )
                debug_full_pcd.publish(full_msg)
                surface_msg = PerceptionInterface.create_cloud(
                    surface_pts,
                    np.linspace([0, 0, 0], [255, 255, 255], len(surface_pts)),
                    255,
                    "world",
                    rospy.Time.now(),
                )
                debug_surface_pcd.publish(surface_msg)
            # print('Write-Share Time:', time.time() - t0, flush=True)
            # print('Perception Sent', flush=True)

        else:
            # print('Perception Failed!', flush=True)
            pass
        perception_time = time.time() - start_time
        f.write(f"{iters}: {perception_time}\n")
        f.flush()
    f.close()
    seg_eval_f.close()
    for share in shares:
        share.close()


def grasping_pipeline(gcon, robot):
    rospy.init_node("planning_grasping_child")
    f = open("grasping_time.txt", "w")

    ## init shared memory pointers ##
    shared_arrays = []
    shares = []
    for name in shared_arr_names:
        e_shm = shared_memory.SharedMemory(name=name)
        shared_arrays.append(np.ndarray(s_shape, dtype=s_dtype, buffer=e_shm.buf))
        shares.append(e_shm)

    ## init modules ##
    from geometry_msgs.msg import Pose

    Pose.__lt__ = lambda a, b: a.position.z < b.position.z
    grasp_planner = GraspPlanner()
    # planner = robot.init_motion_planner()
    ik = TracIKSolver(robot.urdf, "base_link", robot.gripper_link)

    ## init vars ##
    grasp_heap = []
    iters = 0
    grasping_time = 0.0
    tgt_pts = []
    while not rospy.is_shutdown():
        # print("------------grasp running")
        iters += 1
        # print('Grasping Start', flush=True)
        t0 = time.time()

        result = copy_content_from_shared_arrays(shared_arrays)
        points, tgt_pts, tgt_rgb, all_pts, all_rgb, ds_all = result
        while len(tgt_pts) == 0:
            ## get latest perception info ##
            result = copy_content_from_shared_arrays(shared_arrays)
            points, tgt_pts, tgt_rgb, all_pts, all_rgb, ds_all = result

        # tgt_mesh = PerceptionInterface.get_shape_estimate(
        #     tgt_pts,
        # )
        # est_tgt_pts = tm.sample.sample_surface(tgt_mesh, 100)[0]
        # est_tgt_pts = np.concatenate([est_tgt_pts, tgt_pts])
        # est_tgt_rgb = np.ones((est_tgt_pts.shape[0], 3))
        t1 = time.time()
        # print('Read-Share Time:', t1 - t0, flush=True)

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
            ik.ik,
            collision_voxel_tuple=(points, all_pts, 0.005),
            # octomap_reset=planner.reset_octomap,
            # octomap_set=planner.set_planning_scene,
            # visualize=True,
        )
        t2 = time.time()

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

        t3 = time.time()
        # print('GPD Time:', t2 - t1, flush=True)
        # print('Score Sorting Time:', t3 - t2, flush=True)
        # print('Total Grasp Planning Time:', t3 - t0, flush=True)
        gcon.send(grasp_heap)
        # print('Send-Grasp Time:', time.time() - t3, flush=True)
        grasping_time += t3 - t1
        f.write(f"{iters}: {grasping_time}\n")
        f.flush()
    f.close()
    for share in shares:
        share.close()


def viewpoint_pipeline(vcon, robot):
    rospy.init_node("planning_viewpoint_child")
    f = open("viewpoint_time.txt", "w")

    # print('******Viewpoint 1', flush=True)
    ## init shared memory pointers ##
    shared_arrays = []
    shares = []
    for name in shared_arr_names:
        e_shm = shared_memory.SharedMemory(name=name)
        shared_arrays.append(np.ndarray(s_shape, dtype=s_dtype, buffer=e_shm.buf))
        shares.append(e_shm)

    # print('******Viewpoint 2', flush=True)
    ## init modules ##
    grasp_planner = GraspPlanner()
    planner = robot.init_motion_planner()
    ik = TracIKSolver(robot.urdf, "base_link", robot.camera_link)

    # print('******Viewpoint 3', flush=True)
    ## init vars ##
    cam_info = rospy.wait_for_message(
        robot.camera[1] + "/color/camera_info", CameraInfo
    )
    cam_intr = np.array(cam_info.K).reshape((3, 3))
    iters = 0
    viewpoint_time = 0.0
    tgt_pts = []

    # print('******Viewpoint 4', flush=True)
    while not rospy.is_shutdown():
        iters += 1
        # print('******Viewpoint Start', flush=True)

        ## get latest perception info ##
        result = copy_content_from_shared_arrays(shared_arrays)
        points, tgt_pts, tgt_rgb, all_pts, all_rgb, ds_all = result
        if len(tgt_pts) == 0:
            tgt_pts = np.mean(all_pts, axis=0).reshape((1, 3))

        planner.reset_octomap()
        planner.set_planning_scene(points=all_pts, update_moveit=True)

        t0 = time.time()
        color_mask = np.concatenate(
            [
                [1, 0, 0] * np.ones_like(points),
                [0, 0, 1] * np.ones((len(all_pts) - len(points), 3)),
            ]
        )
        center = np.mean(tgt_pts, axis=0)
        radius = 0.5 * (max(tgt_pts[:, 2]) - min(tgt_pts[:, 2]))  # + 0.28
        # radii = [radius + 0.28]
        radii = np.linspace(0.28, 0.56, 2) + radius
        angles = [15, 30, 45, 60, 75]
        for adj_radius in radii:
            views = PerceptionInterface.sample_views(center, [adj_radius], angles)
            # print('Num sampled views: ', len(views), flush=True)

            ik_valid_views = []
            t1 = time.time()
            for view in views:
                pose0 = pose_to_matrix(view)
                for i, j, k, l in [
                    [1.0, 1.0, 0.0, 0.0],
                    [-1, -1, 0.0, 0.0],
                    [0.0, 0.0, -1, 1.0],
                    [0.0, 0.0, 1.0, -1],
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
                    ee_pose = matrix_to_pose(pose1)
                    if ik.ik(pose1) is not None:
                        ik_valid_views.append(view)
                        break
            # print('Ik time: ', time.time() - t1, flush=True)
            # print('Num ik valid views: ', len(ik_valid_views), flush=True)

            c_valid_views = []
            t1 = time.time()
            for view in ik_valid_views:
                pose0 = pose_to_matrix(view)
                for i, j, k, l in [
                    [1.0, 1.0, 0.0, 0.0],
                    [-1, -1, 0.0, 0.0],
                    [0.0, 0.0, -1, 1.0],
                    [0.0, 0.0, 1.0, -1],
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
                    ee_pose = matrix_to_pose(pose1)
                    if not grasp_planner.ik_collision(
                        ee_pose,
                        robot.camera_link,
                    ):
                        c_valid_views.append(view)
                        break
                    # vis_view(view)
                    # input('rotate')
            # print('BioIk time: ', time.time() - t1, flush=True)
            print("Num collision valid views: ", len(c_valid_views), flush=True)
            if len(c_valid_views) > 0:
                break

        t1 = time.time()
        scores = PerceptionInterface.get_info_gain(
            points, all_pts, cam_intr, c_valid_views
        )
        # print('Score time: ', time.time() - t1, flush=True)
        # c_valid_views = [pose_to_matrix(view) for view in c_valid_views]

        t1 = time.time()
        print("Viewpoint Time:", t1 - t0, flush=True)
        vcon.send(list(zip(scores, c_valid_views)))
        viewpoint_time += t1 - t0
        f.write(f"{iters}: {viewpoint_time}\n")
        f.flush()
    f.close()
    for share in shares:
        share.close()


def move_to_target(shared_arrays, gcon, vcon, robot, planner, goal_type="look"):
    debug_pub = rospy.Publisher(
        "/debug/target_points", PointCloud2, queue_size=3, latch=True
    )
    br = tf2_ros.TransformBroadcaster()

    def vis_view(pose, name="TEST"):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "world"
        t.child_frame_id = name
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation.x = pose.orientation.x
        t.transform.rotation.y = pose.orientation.y
        t.transform.rotation.z = pose.orientation.z
        t.transform.rotation.w = pose.orientation.w
        br.sendTransform(t)

    ## ik/fk structures ##
    ik = planner.ik_for_ees[robot.gripper_link]
    ik2 = TracIKSolver(robot.urdf, "base_link", "arm_right_link_7_t")
    ik3 = TracIKSolver(robot.urdf, "base_link", "arm_right_link_6_b")
    ik_cam = TracIKSolver(robot.urdf, "base_link", robot.camera_link)
    joint_state = rospy.wait_for_message("/joint_states_all", JointState)
    ind_js2ik = list(map(joint_state.name.index, ik.joint_names))
    ind_js2ik2 = list(map(joint_state.name.index, ik2.joint_names))
    ind_js2ik3 = list(map(joint_state.name.index, ik3.joint_names))
    ind_js2ik_cam = list(map(joint_state.name.index, ik_cam.joint_names))

    ## joint limits ##
    speed = rospy.get_param("/robot/vel_ang_lim", 5) * np.pi / 180
    accel = rospy.get_param("/robot/acc_ang_lim", 10) * np.pi / 180
    jerk = 100 * np.pi / 180
    prev_vel = np.zeros(len(joint_state.name))
    prev_acc = np.zeros(len(joint_state.name))
    duration = 0
    total_dur = 0

    ## init vars ##
    grasps = []
    randomize = False
    if goal_type in ("look", "sample"):
        wait_for_grasp = False
    else:
        wait_for_grasp = True
    prev_score = (0, 0)
    prev_apts = []
    prev_tpts = []
    tgt_pts = []
    view_points = []
    num_iters = 0
    info_rate = np.inf
    ta = rospy.Time.now().to_sec()
    tb = rospy.Time.now().to_sec()
    ti = rospy.Time.now().to_sec()

    total_motion_planning_time = 0.0
    total_motion_planning_iter = 0
    while not rospy.is_shutdown():
        total_motion_planning_iter += 1
        t0 = rospy.Time.now().to_sec()
        result = copy_content_from_shared_arrays(shared_arrays)
        points, tgt_pts, tgt_rgb, all_pts, all_rgb, ds_all = result
        while len(tgt_pts) == 0 and wait_for_grasp:
            ## get latest perception info ##
            result = copy_content_from_shared_arrays(shared_arrays)
            points, tgt_pts, tgt_rgb, all_pts, all_rgb, ds_all = result
        # d_tgt_pts = 100 * abs(len(tgt_pts) - len(prev_tpts)) / len(tgt_pts)
        # d_all_pts = 100 * abs(len(all_pts) - len(prev_apts)) / len(all_pts)
        # info_rate = (d_tgt_pts + d_all_pts) / (rospy.Time.now().to_sec() - ti)

        if len(tgt_pts) > 0:
            ## visualize ##
            msg = PerceptionInterface.create_cloud(
                tgt_pts,
                np.linspace([0, 0, 0], [255, 255, 255], len(tgt_pts)),
                255,
                "world",
                rospy.Time.now(),
            )
            debug_pub.publish(msg)
            ## end visualize ##
        else:
            tgt_pts = np.mean(all_pts, axis=0).reshape((1, 3))

        ti = rospy.Time.now().to_sec()
        not_equal = len(prev_apts) == 0
        not_equal = not_equal or ds_all.shape != prev_apts.shape
        not_equal = not_equal or (ds_all != prev_apts).any()
        prev_apts = copy.deepcopy(ds_all)
        # prev_tpts = copy.deepcopy(tgt_pts)
        t1 = time.time()
        # print('Read-Share Time:', t1 - t0, flush=True)

        ## get latest grasps ##
        if gcon.poll() or wait_for_grasp:
            num_iters += 1
        while gcon.poll() or wait_for_grasp:
            grasps = gcon.recv()
        wait_for_grasp = False
        t2 = time.time()
        # print('Read-Grasp Time:', t2 - t1, flush=True)

        t3 = time.time()
        ## debug distance ##
        kdtree = KDTree(all_pts)
        # joint_values = np.array(joint_state.position)[ind_js2ik]
        # joint_values2 = np.array(joint_state.position)[ind_js2ik2]
        # joint_values3 = np.array(joint_state.position)[ind_js2ik3]
        joint_values_cam = np.array(joint_state.position)[ind_js2ik_cam]
        # ee_pose = ik.fk(joint_values)
        # arm7_pose = ik2.fk(joint_values2)
        # arm6_pose = ik3.fk(joint_values3)
        cam_pose = ik_cam.fk(joint_values_cam)
        # positions = [ee_pose[:3, 3], arm7_pose[:3, 3], arm6_pose[:3, 3]]
        # dists, inds = kdtree.query(positions, k=1)
        # print('kdtree dist: ', dists, flush=True)
        # collision_weight = 500.0 / min(dists)
        # collision_distance = max(0.14 - min(dists), 0.07)
        # planner.set_collision_params(collision_weight, collision_distance)

        ## get closest grasp ##
        if False:
            i_closest = None
            joint_values = np.array(joint_state.position)[ind_js2ik]
            ee_pose = ik.fk(joint_values)
            ee_position = ee_pose[:3, 3]
            min_dist = np.inf
            i = 0
            for s, g, pg, smp in grasps:
                pg_pose = pose_to_matrix(pg)
                pg_position = [pg.position.x, pg.position.y, pg.position.z]
                distT = np.linalg.norm(pg_position - ee_position)
                distR = np.linalg.norm(pg_pose[:3, :3] - ee_pose[:3, :3])
                distR = min(distR, abs(distR - np.pi))
                if distT + distR < min_dist:
                    min_dT = distT
                    min_dR = distR
                    min_dist = distT + distR
                    i_closest = i
                i += 1

        ## update high score ##
        if grasps:
            ss, gs, pgs, smps = zip(*grasps)
            max_score = max(ss)
            if max_score > prev_score:
                if max_score[0] - prev_score[0] > 0.05:
                    num_iters = 0
                prev_score = max_score

        # check termnination conditions
        still_time = rospy.Time.now().to_sec() - tb - duration
        # stop_cond = num_iters >= 10 and prev_score[0] > 0.6
        stop_cond = info_rate < 1000 and prev_score[0] > 0.6
        stop_cond |= info_rate < 500 and prev_score[0] > 0.5
        print("Num Iters:", num_iters, flush=True)
        print("Max grasp score:", prev_score, flush=True)
        print("Info Rate:", info_rate, flush=True)
        # print('Still Time:', still_time, flush=True)
        if stop_cond and len(grasps) > 0:
            break

        # update planning scene if different
        # print('0 Not Equal?:', not_equal, flush=True)
        if not_equal:
            t000 = rospy.Time.now().to_sec()
            # print('Update Planning Scene...', flush=True)
            planner.reset(update_moveit=False)
            planner.set_planning_scene(
                points=ds_all,
                update_moveit=False,
                # visualize=True,
            )
            # print(
            #     'Update Planning Scene Time 0:',
            #     rospy.Time.now().to_sec() - t000,
            #     flush=True
            # )
            # planner.reset_octomap()
            # planner.set_planning_scene(
            #     points=all_pts,
            #     update_moveit=True,
            # )
            # print(
            #     'Update Planning Scene Time 1:',
            #     rospy.Time.now().to_sec() - t000,
            #     flush=True
            # )

        ## define planning goals ##
        # t000 = rospy.Time.now().to_sec()
        target = None
        look_link = None
        lookat_point = None
        lookat_axis = None
        cone_direction = None
        cone_angle = None
        cone_point = None
        end_effector = robot.gripper_link
        if goal_type == "look":
            l_points, l_norms, l_scores = planner.make_exploration_goal(
                tgt_pts, None, 1
            )
            if randomize:
                ind_l = np.random.randint(0, len(l_scores))
            else:
                ind_l = np.argmax(l_scores)
            l_norm = l_norms[ind_l]
            l_point = l_points[ind_l]
            l_score = l_scores[ind_l]
            look_rand = False
            # print('0 Look Score:', l_score, flush=True)
            look_link = robot.camera_link
            lookat_point = l_point
            lookat_axis = robot.camera_axis
            cone_direction = l_norm
            cone_angle = np.pi / 6
        elif goal_type == "sample":
            # t001 = rospy.Time.now().to_sec()
            while vcon.poll() or len(view_points) == 0:
                views = vcon.recv()
                view_scores, view_points = zip(*views)
                print("Max Viewpoint Score:", max(view_scores), flush=True)
            # print(
            #     'Viewpoint Time Read:',
            #     rospy.Time.now().to_sec() - t001,
            #     flush=True
            # )
            if randomize:
                i_select = np.random.randint(0, len(view_points))
            else:
                i_select = np.argmax(view_scores)
            pose = pose_to_matrix(view_points[i_select])
            # pose = view_points[i_select]
            target = pose[:3, 3]  # just set position goal
            # dxyz = np.subtract(pose[:3, 3], cam_pose[:3, 3])
            # ndxyz = np.linalg.norm(dxyz)
            # if ndxyz < 0.05:
            #     target = pose[:3, 3]
            # else:
            #     target = cam_pose[:3, 3] + 0.05 * dxyz / ndxyz
            # vis_view(Pose(Point(*target), Quaternion(0, 0, 0, 1)), 't')
            vis_view(view_points[i_select])
            # vis_view(matrix_to_pose(view_points[i_select]))
            end_effector = robot.camera_link
            # set lookat goal
            look_link = robot.camera_link
            lookat_point = np.mean(tgt_pts, axis=0)
            lookat_axis = robot.camera_axis
            # cone_direction = pose[:3, 2]
            # cone_angle = np.pi / 8
            # cone_point = pose[:3, 3]
            info_rate = max(view_scores)
        else:
            if grasps:
                if randomize:
                    i_select = np.random.randint(0, len(grasps))
                else:
                    ss, gs, pgs, smps = zip(*grasps)
                    i_select = np.argmax(ss)
                target = grasps[i_select][2]
        # print('Goal Time:', rospy.Time.now().to_sec() - t000, flush=True)

        t000 = rospy.Time.now().to_sec()
        randomize = False
        plan = planner.iter_ik_motion_plan(
            joint_state,
            target,
            robot.gripper_group,
            score=1,
            num_iters=1,
            speed=speed,
            min_duration=0.05,
            look_link=look_link,
            lookat_point=lookat_point,
            lookat_axis=lookat_axis,
            cone_direction=cone_direction,
            cone_angle=cone_angle,
            # cone_point=cone_point,
            check_collisions=True,
            ee=end_effector,
            is_diff=False,
        )
        print("Plan Time:", rospy.Time.now().to_sec() - t000, flush=True)
        # planner.set_collision_params(200, 0.085)
        if plan is None:
            print("No plan found!", flush=True)
            planner.reset(reset_static=True, update_moveit=False)
            planner.set_planning_scene(points=ds_all, update_moveit=False)
            randomize = True
            t4 = rospy.Time.now().to_sec()
            total_motion_planning_time += t4 - t3
            continue
        plan.points.pop(0)

        inds = list(map(plan.joint_names.index, joint_state.name))
        vals = np.array(plan.points[-1].positions)[inds]
        disp = vals - joint_state.position
        dist = max(np.abs(disp))
        print("Displacement: ", dist, flush=True)
        if dist <= 0.02:
            planner.reset(reset_static=True, update_moveit=False)
            planner.set_planning_scene(points=ds_all, update_moveit=False)
            randomize = True
            t4 = rospy.Time.now().to_sec()
            total_motion_planning_time += t4 - t3
            continue
        if rospy.Time.now().to_sec() - tb - duration > 0:
            prev_acc = np.zeros(len(joint_state.name))
            prev_vel = np.zeros(len(joint_state.name))
        duration = plan.points[-1].time_from_start.to_sec()
        # print('******Duration Before!:', duration, flush=True)
        vel = plan.points[-1].velocities
        max_dv = max(np.abs(vel - prev_vel))
        if max_dv / duration > accel:
            duration = max_dv / accel
            vel = disp / duration
        acc = (vel - prev_vel) / duration
        max_da = max(np.abs(acc - prev_acc))
        if max_da / duration > jerk:
            duration = max_da / jerk
            vel = disp / duration
            acc = (vel - prev_vel) / duration
        plan.points[-1].velocities = tuple(vel)
        # print('******Duration After!:', duration, flush=True)
        total_dur += duration
        plan.points[-1].time_from_start = rospy.Duration(total_dur)
        # plan.points[-1].velocities = tuple(disp / duration)
        t4 = rospy.Time.now().to_sec()
        total_motion_planning_time += t4 - t3

        print("Dur, total, time", duration, total_dur, t4 - t3, flush=True)
        tb = rospy.Time.now().to_sec()
        sleep_time = max(duration, 0.2)  # avg plan time vs next duration
        while (tb - ta) < (total_dur - sleep_time):
            tb = rospy.Time.now().to_sec()
        is_stream_new, _wp = execute(plan, window=duration, wait=False, retime=False)
        if is_stream_new == 1:
            ta = tb
            total_dur = duration

        # update joint state for next iteration
        joint_state.position = vals
        prev_vel = np.array(plan.points[-1].velocities)
        prev_acc = acc

    print("total motion planning explore time: ", total_motion_planning_time)
    print("total motion planning explore iter: ", total_motion_planning_iter)
    ss, gs, pgs, smps = (list(x) for x in zip(*grasps))
    return gs, pgs, ss, joint_state


def move_to_target_2(gcon, robot, planner):
    mg = planner.move_groups[robot.gripper_group]
    ik = planner.ik_for_ees[robot.gripper_link]
    joint_state = rospy.wait_for_message("/joint_states_all", JointState)
    ind_js2ik = list(map(joint_state.name.index, ik.joint_names))

    # helper function:
    def plan_to_grasps():
        # update planning scene if different
        not_equal = prev_points is None
        not_equal = not_equal or all_points.shape != prev_points.shape
        not_equal = not_equal or (all_points != prev_points).all()
        if not_equal:
            planner.reset(reset_static=True, update_moveit=False)
            planner.set_planning_scene(
                points=all_points,
                # colors=all_colors,
                update_moveit=False,
                # visualize=True,
            )
        # trajectory = planner.pose_motion_plan(
        #     joint_state,
        #     # grasps[0][1],
        #     [g[1] for g in grasps],
        #     robot.gripper_group,
        #     is_diff=False,
        # )
        # ss, gs, pgs, smps = zip(*grasps)
        traj = planner.iter_ik_motion_plan(
            joint_state,
            closest_p_grasp,
            robot.gripper_group,
            score=10,
            num_iters=NUM_ITERS,
            speed=10 * np.pi / 180,
            look_link=robot.camera_link,
            lookat_point=np.mean(tgt_pts, axis=0),
            lookat_axis=robot.camera_axis,
            ee=robot.gripper_link,
            is_diff=False,
        )
        if traj:
            # print([p.time_from_start for p in traj.points], flush=True)
            times = []
            positions = []
            velocities = []
            for i in range(len(traj.points)):
                times.append(traj.points[i].time_from_start.to_sec())
                positions.append(traj.points[i].positions)
                velocities.append(traj.points[i].velocities)
            path = CubicHermiteSpline(times, positions, velocities)
            step = WINDOW / 10
            ts = np.arange(0, times[-1] + step / 2, step)
            pos = path(ts)
            vel = path(ts, 1)
            # pos, vel, ts, _ = MI.retime_trajectroy(traj.points, 0.1)
            traj.points.clear()
            for i in range(len(ts)):
                traj.points.append(
                    JointTrajectoryPoint(
                        positions=pos[i],
                        velocities=vel[i],
                        time_from_start=rospy.Duration(ts[i]),
                    )
                )
        return traj

    grasps = None
    prev_points = None
    closest_p_grasp = None
    highest_p_grasp = None
    window_points = None
    while not rospy.is_shutdown():
        t10 = rospy.Time.now().to_sec()
        while gcon.poll() or not grasps:
            grasps, tgt_pts, all_points = gcon.recv()
        t11 = rospy.Time.now().to_sec()
        # print('Get Grasps or Skip:', t11 - t10, flush=True)

        # check proximity to grasp
        joint_values = np.array(joint_state.position)[ind_js2ik]
        ee_pose = ik.fk(joint_values)
        ee_position = ee_pose[:3, 3]
        min_dist = np.inf
        max_score = 0
        for s, g, pg, smps in grasps:
            pg_pose = pose_to_matrix(pg)
            pg_position = [pg.position.x, pg.position.y, pg.position.z]
            distT = np.linalg.norm(pg_position - ee_position)
            distR = np.linalg.norm(pg_pose[:3, :3] - ee_pose[:3, :3])
            if distT + distR < min_dist:
                min_dT = distT
                min_dR = distR
                min_dist = distT + distR
                closest_p_grasp = pg
            if s > max_score:
                max_score = s
                highest_p_grasp = pg
        if min_dT < 0.05 and (min_dR < 0.1 or abs(min_dR - 3.14) < 0.1):
            break
        t12 = rospy.Time.now().to_sec()
        # print('Distances (T,R,ctime):', distT, distR, t12 - t11, flush=True)

        t00 = rospy.Time.now().to_sec()
        robot_status = rospy.wait_for_message("/robot_status", RobotStatus, timeout=0.1)
        t01 = rospy.Time.now().to_sec()
        # print('Get Robot Status:', t01 - t00, flush=True)
        if robot_status.in_motion.val == 0:
            # print('Planning from Stopped State!', flush=True)
            joint_state = rospy.wait_for_message(
                "/joint_states_all", JointState, timeout=0.1
            )
            # if plan is None:
            #     # mg.set_planner_id("LazyPRMStarSemiPers")
            #     mg.set_planning_time(10)
            # else:
            #     # mg.set_planner_id("LazyPRMSemiPers")
            #     mg.set_planning_time(WINDOW)
            plan = plan_to_grasps()
            prev_points = all_points
            if type(plan) is JointTrajectory:
                error, window_points = execute(
                    plan, window=WINDOW, wait=False, retime=False
                )
                t = rospy.Time.now().to_sec()
        else:
            if window_points is None or len(window_points.points) == 0:
                continue
            planning_point = window_points.points.pop(0)
            tft = planning_point.time_from_start
            # print('pop:', tft.to_sec(), flush=True)
            try:
                t0 = rospy.Time.now().to_sec()
                while tft.to_sec() < WINDOW * 0.2 + (t0 - t):
                    planning_point = window_points.points.pop(0)
                    tft = planning_point.time_from_start
                    # print('pop:', tft.to_sec(), flush=True)
            except IndexError:
                plan = None
                continue

            joint_state = JointState(
                name=window_points.joint_names,
                position=planning_point.positions,
                velocity=planning_point.velocities,
            )
            plan = plan_to_grasps()
            for i in range(len(plan.points)):
                plan.points[i].time_from_start += tft
            t1 = rospy.Time.now().to_sec()
            # print('Planning Time:', (t1 - t0), flush=True)

            # print('t1-t:', t1 - t, flush=True)
            # print('ppt:', tft.to_sec(), flush=True)
            # print('ppt > t1-t:', tft.to_sec() > t1 - t, flush=True)
            while tft.to_sec() > (WINDOW + (t1 - t)):
                # print('post loop', t1, flush=True)
                t1 = rospy.Time.now().to_sec()

            error, window_points = execute(
                plan, window=WINDOW, wait=False, retime=False
            )

    ss, gs, pgs, smps = zip(*grasps)
    return gs[pgs.index(closest_p_grasp)], closest_p_grasp


def closed_loop_pick_or_place(
    object_to_grasp, robot, gt, place=None, eval_xml_file="nothing"
):
    ## clean old shared memory ##
    for name in shared_arr_names:
        try:
            garbage = shared_memory.SharedMemory(name=name)
            garbage.unlink()
            garbage.close()
        except FileNotFoundError:
            pass

    ## initialize shared memory ##
    shared_arrays = []
    shares = []
    for name in shared_arr_names:
        shm = shared_memory.SharedMemory(name=name, create=True, size=s_size)
        shared_arrays.append(np.ndarray(s_shape, dtype=s_dtype, buffer=shm.buf))
        shares.append(shm)

    mp.set_start_method("spawn")
    # mp.set_start_method('forkserver')

    ## start perception pipeline ##
    perc_proc = Process(
        target=perception_pipeline,
        args=(object_to_grasp, robot, gt, eval_xml_file),
    )
    perc_proc.start()

    ## start grasping pipeline ##
    gconS, gconR = Pipe()
    grasp_proc = Process(
        target=grasping_pipeline,
        args=(gconS, robot),
    )
    grasp_proc.start()

    ## start viewpoint pipeline ##
    vconS, vconR = Pipe()
    viewpoint_proc = Process(
        target=viewpoint_pipeline,
        args=(vconS, robot),
    )
    viewpoint_proc.start()

    t0 = time.time()
    ## plan and execute loop ##
    planner = robot.init_motion_planner()
    grasps, pre_grasps, scores, js = move_to_target(
        shared_arrays,
        gconR,
        vconR,
        robot,
        planner,
        goal_type="sample",
    )

    ## switch to idle mode after point streaming ##
    t1 = time.time()
    closed_loop_execution_time = t1 - t0
    print("Closed-loop Execution Time:", closed_loop_execution_time)

    ## get latest perception info ##
    result = copy_content_from_shared_arrays(shared_arrays)
    points, tgt_pts, tgt_rgb, all_pts, all_rgb, ds_all = result

    ## stop subprocesses ##
    viewpoint_proc.kill()
    viewpoint_proc.join()
    vconS.close()
    vconR.close()
    # print('Stopping Grasping subprocesses ...')
    grasp_proc.kill()
    grasp_proc.join()
    gconS.close()
    gconR.close()
    # print('Stopping Perception subprocesses...')
    perc_proc.kill()
    perc_proc.join()
    # print('Perception Stopped!')
    # input('Press Enter to continue...')
    openloop_motion_planning_time = time.time()

    ik = planner.ik_for_ees[robot.gripper_link]
    ik2 = TracIKSolver(robot.urdf, "base_link", "arm_right_link_7_t")
    ik3 = TracIKSolver(robot.urdf, "base_link", "arm_right_link_6_b")
    grasp_planner = GraspPlanner()
    # for g, s in zip(grasps, scores):
    #     invalid = grasp_planner.ik_collision(g, visualize=True)
    #     input(f'Score: {s}, Invalid: {invalid}, ...')

    planner.reset(update_moveit=True)
    ind = None
    while len(grasps) > 0:
        t1 = time.time()
        # filter all_pts by robot state
        all_pts_filtered = copy.deepcopy(all_pts)
        ind_js2ik = list(map(js.name.index, ik.joint_names))
        ind_js2ik2 = list(map(js.name.index, ik2.joint_names))
        ind_js2ik3 = list(map(js.name.index, ik3.joint_names))
        joint_values = np.array(js.position)[ind_js2ik]
        joint_values2 = np.array(js.position)[ind_js2ik2]
        joint_values3 = np.array(js.position)[ind_js2ik3]
        ee_pose = ik.fk(joint_values)
        arm7_pose = ik2.fk(joint_values2)
        arm6_pose = ik3.fk(joint_values3)
        positions = [ee_pose[:3, 3], arm7_pose[:3, 3], arm6_pose[:3, 3]]
        kdtree = KDTree(positions)
        dists, inds = kdtree.query(all_pts_filtered, k=1)
        all_pts_filtered = all_pts_filtered[dists > 0.09]
        all_pts_filtered = np.concatenate([all_pts_filtered, tgt_pts])
        print("Dists:", min(dists), max(dists), flush=True)
        ## plan to target ##
        planner.reset_octomap()
        planner.set_planning_scene(
            # points=all_pts,
            points=all_pts_filtered,
            # colors=all_rgb,
            # visualize=True,
        )
        # call bio_ik for goal joint_state
        ind = scores.index(max(scores))
        print("Chosen Grasp Score:", scores[ind])
        c_iter = 0
        while c_iter < 10:
            c_iter += 1
            goal_js, colliding = grasp_planner.ik_solve(
                pre_grasps[ind],
                visualize=True,
            )
            if not colliding:
                break
        ind_gjs = list(map(goal_js.name.index, ik.joint_names))
        goal_js.name = ik.joint_names
        goal_js.position = np.array(goal_js.position)[ind_gjs]
        plan1 = planner.joint_motion_plan(
            js,
            goal_js,
            robot.gripper_group,
            is_diff=False,
        )
        # plan1 = planner.pose_motion_plan(
        #     js,
        #     pre_grasps[ind],
        #     robot.gripper_group,
        #     is_diff=False,
        # )
        t10 = time.time()
        print("Plan1 Time:", t10 - t1)
        if type(plan1) is not JointTrajectory:
            pre_grasps.pop(ind)
            grasps.pop(ind)
            scores.pop(ind)
            ind = None
            continue

        ## compute pre-grasp distance ##
        js2 = JointState()
        js2.name = plan1.joint_names
        js2.position = plan1.points[-1].positions
        grasp_position = pose_to_matrix(grasps[ind])[:3, 3]
        pre_grasp_pose = pose_to_matrix(pre_grasps[ind])[:3, 3]
        dist = np.linalg.norm(grasp_position - pre_grasp_pose)
        # print('New Pre-grasp Distance:', dist)
        planner.reset_octomap()
        f2, plan2 = planner.cartesian_motion(
            js2,
            (0, 0, dist),
            robot.gripper_group,
            robot.gripper_link,
            xyz_is_relative=True,
            avoid_collisions=True,
            is_diff=False,
        )
        t11 = time.time()
        print("Plan2 Time:", t11 - t10)
        if f2 < 1.0:
            # print('Plan2 Error:', f2)
            # remove the current grasp pose at index ind
            pre_grasps.pop(ind)
            grasps.pop(ind)
            scores.pop(ind)
            ind = None
            continue

        ## plan lift ##
        js3 = JointState()
        js3.name = plan2.joint_names
        js3.position = plan2.points[-1].positions
        grasp_state = js3
        f3, plan3 = planner.cartesian_motion(
            js3,
            (0, 0, lift_height),
            robot.gripper_group,
            robot.gripper_link,
            xyz_is_relative=False,
            avoid_collisions=True,
            is_diff=False,
        )
        t12 = time.time()
        print("Plan3 Time:", t12 - t11)
        if f3 <= 0.01:
            # print('Plan3 Error:', f3)
            pre_grasps.pop(ind)
            grasps.pop(ind)
            scores.pop(ind)
            ind = None
            continue

        if place:
            ## plan to place ##
            # mg = planner.move_groups[robot.gripper_group]
            # mg.set_planning_time(10)
            js4 = JointState()
            js4.name = plan3.joint_names
            js4.position = plan3.points[-1].positions

            # filter all_pts by robot state
            all_pts_filtered = copy.deepcopy(all_pts)
            ind_js2ik = list(map(js4.name.index, ik.joint_names))
            ind_js2ik2 = list(map(js4.name.index, ik2.joint_names))
            ind_js2ik3 = list(map(js4.name.index, ik3.joint_names))
            joint_values = np.array(js4.position)[ind_js2ik]
            joint_values2 = np.array(js4.position)[ind_js2ik2]
            joint_values3 = np.array(js4.position)[ind_js2ik3]
            ee_pose = ik.fk(joint_values)
            arm7_pose = ik2.fk(joint_values2)
            arm6_pose = ik3.fk(joint_values3)
            positions = [ee_pose[:3, 3], arm7_pose[:3, 3], arm6_pose[:3, 3]]
            kdtree = KDTree(positions)
            dists, inds = kdtree.query(all_pts_filtered, k=1)
            all_pts_filtered = all_pts_filtered[dists > 0.09]
            print("Dists:", min(dists), max(dists), flush=True)

            plan4 = plan_place(
                js4,
                place,
                grasp_state,
                tgt_pts,
                # all_pts,
                all_pts_filtered,
                planner,
                robot,
                # tgt_rgb=tgt_rgb,
                # all_rgb=all_rgb,
            )
            t13 = time.time()
            print("Plan4 Time:", t13 - t12)
            if type(plan4) is not JointTrajectory:
                pre_grasps.pop(ind)
                grasps.pop(ind)
                scores.pop(ind)
                ind = None
                continue
            plans = [plan1, plan2, plan3, plan4]
        else:
            plans = [plan1, plan2, plan3]
        break

    openloop_motion_planning_time = time.time() - openloop_motion_planning_time
    print("open loop motion planning time: ", openloop_motion_planning_time)

    ## wait till robot stops ##
    wait_till(stop=True)
    execute(JointTrajectory())

    t0 = time.time()
    for i, plan in enumerate(plans):
        if i == 2:
            ee_close()
        if type(plan) is JointTrajectory:
            # print('Plan success!')
            execute(plan, window=0.1, wait=True)
            # execute(plan, window=0)
            # rospy.sleep(0.5)
        else:
            # print('Plan', i, 'error:', plan)
            break

    t1 = time.time()
    open_loop_execution_time = t1 - t0
    print("Open-loop Execution Time:", open_loop_execution_time)
    for share in shares:
        share.close()
        share.unlink()


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
    object_to_grasp = (
        sys.argv[3] if len(sys.argv) > 3 else 38
    )  # 38 corresponds to tomato_soup_can
    place = [0.6, 0, 1.1, 0.5, -0.5, -0.5, -0.5] if len(sys.argv) > 4 else None
    eval_xml_file = sys.argv[5] if len(sys.argv) > 5 else "nothing"
    body_name = sys.argv[6] if len(sys.argv) > 6 else False
    run_only_perception = sys.argv[7] if len(sys.argv) > 7 else False

    print(
        "Args: is_sim (s or r), gt (g or r), object_to_grasp (id sim or name real), place (p), xml_file (path or 'nothing'), body_name (name of object->> if xml file is nothing this is irrelevant), run_only_perception (True or blank)"
    )

    ## init perception and planning interfaces ##
    rospy.init_node("planning")
    t0 = time.time()
    from motoman import MotomanSDA10F

    robot = MotomanSDA10F(
        is_sim, gt
    )  # To use the real camera messages, set the first argument is_sim = r
    print("after made robot")
    t1 = time.time()
    # print('Init Time:', t1 - t0)

    if run_only_perception != False:
        print("Only perception!")
        mp.set_start_method("spawn")
        ## clean old shared memory ##
        for name in shared_arr_names:
            try:
                garbage = shared_memory.SharedMemory(name=name)
                garbage.unlink()
                garbage.close()
            except FileNotFoundError:
                pass

        ## initialize shared memory ##
        shared_arrays = []
        shares = []
        for name in shared_arr_names:
            shm = shared_memory.SharedMemory(name=name, create=True, size=s_size)
            shared_arrays.append(np.ndarray(s_shape, dtype=s_dtype, buffer=shm.buf))
            shares.append(shm)
        # mp.set_start_method('forkserver')
        ## start perception pipeline ##
        perc_proc = Process(
            target=perception_pipeline,
            args=(object_to_grasp, robot, gt, eval_xml_file, body_name),
        )
        perc_proc.start()
        # perception_pipeline(object_to_grasp, robot, gt, eval_xml_file)
    else:
        closed_loop_pick_or_place(
            object_to_grasp, robot, gt, place, eval_xml_file
        )  # python closed_loop.py s g 36 p "../../xmls/adjusted_pos_ycb_boxes.xml"
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

    To run tests:
    1. Choose rosbag
    2. Look to find the right item
    3. Put correct item in the adjust_object.py file
    4. Run experiment with the right item and adjust
    5. Save the object position in the xml
    5. Put the xml and object into the tmux closed_loop.py
    6. Run the experiment (voting and no voting in closed_loop.py)
    7. Save the text file and image in the results folder
    """

    # killall -9 -u j_user

    input("Open?")
    ee_open()
