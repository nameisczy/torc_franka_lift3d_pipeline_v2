import os
import sys
import copy
import time
import heapq
import threading
import multiprocessing as mp
from multiprocessing import Process, Pipe, shared_memory

import torch
import numpy as np
import open3d as o3d
import transformations as tf
from scipy.spatial import KDTree
from tracikpy import TracIKSolver
from scipy.interpolate import CubicHermiteSpline

from curobo.rollout.rollout_base import Goal
from curobo.types.robot import JointState as JointState_CU

import rospy
import tf2_ros
import sensor_msgs.point_cloud2 as pc2
from industrial_msgs.msg import RobotStatus
from sensor_msgs.msg import CameraInfo, JointState, PointCloud2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TransformStamped, Pose, Point, Quaternion
from lab_vbnpm.srv import GetNextPlanningPoint, ExecuteTrajectoryResponse

from perception.perception_fast import PerceptionInterface
from grasp_planner.curobo_grasp_planner import GraspPlanner
from task_planner.curobo_open_loop import update_and_get_points
from execution_scene.motoman_interface import MotomanInterface as MI
from task_planner.eutils import (
    ee_open,
    ee_close,
    execute,
    wait_till,
    get_experiment_result,
)
from utils.conversions import pose_to_matrix, matrix_to_pose, float_to_ros_duration
import mujoco
try:
    from curobo.rollout.cost.ray_cost import RayCost
except ModuleNotFoundError:
    RayCost = None
import torch
from scipy.spatial.transform import Rotation as R

# import trimesh as tm # must import after perception to avoid conflict
# TODO: why is above true‽

SAVE_RECORDING = False
# SAVE_RECORDING = 'video'
# SAVE_RECORDING = 'pcd'
# SAVE_RECORDING = 'all'

TIME_OUT = 300  # seconds

num_pts_to_sample = 10000
grasp_heap_size = 50
lift_height = 0.01  # meters

target_end_effector_velocity = 1.0  # meters/second
control_step_dt = 1.0  # seconds

shared_arr_names = (
    "pts",
    "tgt_pts",
    "tgt_rgb",
    "all_pts",
    "all_rgb",
)
s_shape = (3000000, 3)
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
import os


def perception_pipeline(
    object_to_grasp, robot, gt, eval_xml_file="nothing", body_name="004_sugar_box"
):
    rospy.init_node("planning_perception_child")
    f = open("perception_time.txt", "w")
    seg_eval_f = open("segmentation_evaluation.txt", "w")
    log_fd = open("perception_pipeline.log", "a")
    os.dup2(log_fd.fileno(), 1)  # Redirect stdout
    os.dup2(log_fd.fileno(), 2)
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
    if eval_xml_file != "nothing":
        mj_model = mujoco.MjModel.from_xml_path(
            eval_xml_file
        )  # This sometimes breaks so just do it once
    while not rospy.is_shutdown():
        used1 = perception.did_use_cam_pose_before(robot.camera[0])
        used2 = perception.did_use_cam_pose_before(robot.camera[1])
        if used1 and used2:
            continue
        print("Perception Start", object_to_grasp, flush=True)

        result = update_and_get_points(
            object_to_grasp,
            robot,
            perception,
            camera_inds=cam_inds,
            save_debug=SAVE_RECORDING,
            debug_number=iters,
        )
        object_to_grasp = ""  # track image rather than grounding again
        iters += 1

        print("After update and get points")

        if result is not None:
            _, tgt_pts, _, _, _ = result

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

        if len(cam_inds) > 1:
            cam_inds.pop(0)
        if result is not None:
            # t0 = time.time()

            copy_content_to_shared_arrays(shared_arrays, result[:-1])
            # j_user: To prevent crashing due to no target points
            surface_pts, tgt_pts, _, all_pts, _ = result
            print("length all points", len(all_pts))
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
            print("Perception Sent", flush=True)

        else:
            print("Perception Failed!", flush=True)
            pass
        perception_time = time.time() - start_time
        f.write(f"{iters}: {perception_time}\n")
        f.flush()
        print("Perception time", perception_time)
        print(
            "now waiting for a new image to be seen from a new pose to have a reason to run more calculations"
        )
    f.close()
    seg_eval_f.close()
    for share in shares:
        share.close()


def grasping_pipeline(gcon, robot, world_config):
    rospy.init_node("planning_grasping_child")
    f = open("grasping_time.txt", "w")
    log_fd = open("grasping_pipeline.log", "a")
    os.dup2(log_fd.fileno(), 1)  # Redirect stdout
    os.dup2(log_fd.fileno(), 2)

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
    grasp_planner = GraspPlanner(
        robot.curobo_config,
        world_config,
        robot.urdf,
        ignore_collision_ee_links=robot.ignore_collision_ee_links,
    )
    # planner = robot.init_motion_planner(planner='curobo', warmup=False)
    ik = TracIKSolver(robot.urdf, "base_link", robot.gripper_link)

    ## init vars ##
    grasp_heap = []
    pre_grasp_heap = []
    iters = 0
    grasping_time = 0.0
    tgt_pts = []
    while not rospy.is_shutdown():
        print("------------grasp running")
        iters += 1
        # print('Grasping Start', flush=True)
        t0 = time.time()

        result = copy_content_from_shared_arrays(shared_arrays)
        points, tgt_pts, tgt_rgb, all_pts, all_rgb = result
        while len(tgt_pts) == 0 and len(all_pts) == 0:
            ## get latest perception info ##
            result = copy_content_from_shared_arrays(shared_arrays)
            points, tgt_pts, tgt_rgb, all_pts, all_rgb = result

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
                np.reshape([g[5] for g in grasp_heap], (-1, 3)),
            ]
        )
        colors_to_sample = np.concatenate(
            [
                np.array(tgt_rgb)[indices],
                np.zeros((len(grasp_heap), 3)),
            ]
        )
        grasp_result = grasp_planner.get_grasp_poses(
            points_to_sample,
            colors_to_sample,
            points,
            all_pts,
            visualize=True,
        )
        if len(grasp_result) == 7:
            grasps = grasp_result[0]
            pre_grasps = grasp_result[1]
            grasp_js = grasp_result[2]
            p_grasp_js = grasp_result[3]
            scores = grasp_result[4]
            samples = grasp_result[5]
            failure = False
            if max(scores) < 0.3:
                failure = True
        else:
            # failure is one of these cases:
            # 'NO_GRASPS'
            # 'IK_INFEASIBLE'
            # 'IN_COLLISION_TARGET'
            # 'IN_COLLISION_SCENE'
            # 'IK_INFEASIBLE_PRE_GRASPS'
            # 'IN_COLLISION_PRE_GRASPS'
            failure, grasps, grasp_js, scores = grasp_result

        if failure:
            if len(grasp_heap) > 0:
                # technically should not happen but continue just in case
                gcon.send(grasp_heap)
                t3 = time.time()
                print("Total Grasp Planning Time:", t3 - t0, flush=True)
                grasping_time += t3 - t1
                f.write(f"{iters}: {grasping_time}\n")
                f.flush()
                continue
            if failure == "NO_GRASPS":
                continue

            pre_grasp_result = grasp_planner.find_nearest_grasp_retraction(
                grasps,
                scores,
                # visualize=True,
            )
            pre_grasps, pre_grasp_joints, pre_scores = pre_grasp_result
            i = 0
            pre_heap_size = grasp_heap_size
            while len(pre_grasp_heap) < pre_heap_size and i < len(pre_grasps):
                heapq.heappush(
                    pre_grasp_heap,
                    (
                        pre_scores[i],
                        None,
                        pre_grasps[i],
                        None,
                        pre_grasp_joints[i],
                        None,
                    ),
                )
                i += 1
            while i < len(pre_grasps):
                heapq.heappushpop(
                    pre_grasp_heap,
                    (
                        pre_scores[i],
                        None,
                        pre_grasps[i],
                        None,
                        pre_grasp_joints[i],
                        None,
                    ),
                )
                i += 1

            if len(pre_grasp_heap) > 0:
                unzip = list(zip(*pre_grasp_heap))
                g_color = [0] * len(pre_grasp_heap)
                g_color[np.argmax(unzip[0])] = -1
                # grasp_planner.plotter.draw_grasps()
                grasp_planner.plotter.draw_grasps(unzip[2], g_color)
                gcon.send(pre_grasp_heap)
            t3 = time.time()
            print("Total Grasp Planning Time:", t3 - t0, flush=True)
            grasping_time += t3 - t1
            f.write(f"{iters}: {grasping_time}\n")
            f.flush()
            continue

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
                (
                    scores[j],
                    grasps[j],
                    pre_grasps[j],
                    grasp_js[j],
                    p_grasp_js[j],
                    sample,
                ),
            )
            i += 1

        # pushpop new grasps and maintain heap size
        while i < len(best_for_sample):
            sample, j = best_samples[i]
            heapq.heappushpop(
                grasp_heap,
                (
                    scores[j],
                    grasps[j],
                    pre_grasps[j],
                    grasp_js[j],
                    p_grasp_js[j],
                    sample,
                ),
            )
            # consider heapq.heapreplace to gauranty new items are considered
            i += 1

        t3 = time.time()
        print("Grasp Service Time:", t2 - t1, flush=True)
        print("Score Sorting Time:", t3 - t2, flush=True)
        print("Total Grasp Planning Time:", t3 - t0, flush=True)
        if len(grasp_heap) > 0:
            # visualize grasps
            unzip = list(zip(*grasp_heap))
            grasp_planner.plotter.draw_grasps()
            grasp_planner.plotter.draw_grasps(unzip[1], unzip[0])
            g_color = [0] * len(grasp_heap)
            g_color[np.argmax(unzip[0])] = -1
            grasp_planner.plotter.draw_grasps(unzip[2], g_color)
            # ind=np.argmax(unzip[0])
            # grasp_planner.plotter.draw_grasps([unzip[2][ind]],[1])
        gcon.send(grasp_heap)
        # print('Send-Grasp Time:', time.time() - t3, flush=True)
        grasping_time += t3 - t1
        f.write(f"{iters}: {grasping_time}\n")
        f.flush()
    f.close()
    for share in shares:
        share.close()


def move_to_target(
    shared_arrays,
    gcon,
    robot,
    planner,
    goal_type="proposed",
    poses=None,
    out_file=sys.stderr,
):
    # get start joint position
    joint_state = rospy.wait_for_message("/joint_states_all", JointState)

    # initialize publishers and callbacks
    def update_mpc_scene_callback(msg):
        all_pts = list(pc2.read_points(msg, ("x", "y", "z"), True))
        planner.set_planning_scene(
            points=all_pts,
            send_to_gpu=False,
            save_scene_file="/tmp/world_config.pth",
        )
        # planner.visualize_rviz()

    rospy.Subscriber("/debug/full_pcd", PointCloud2, update_mpc_scene_callback)

    print("Goal type:", goal_type)

    if goal_type == "proposed":

        def update_ray_scene_callback(msg):
            surface_pts = list(pc2.read_points(msg, ("x", "y", "z"), True))
            # print("surface pts type", type(surface_pts))
            surface_pts = np.asarray(surface_pts)
            z_min = np.min(surface_pts[:, 2])
            surface_pts = surface_pts[surface_pts[:, 2] > (z_min + 0.005)]
            surface_pts = surface_pts.tolist()
            planner.set_planning_scene(
                points=surface_pts,
                send_to_gpu=False,
                raytrace_version=True,
            )
            # planner.visualize_rviz(raytrace=True)

        rospy.Subscriber("/debug/surface_pcd", PointCloud2, update_ray_scene_callback)
        wait_for_grasp = False
        planner.ray_cost_enabled = True
    elif goal_type == "baseline":
        wait_for_grasp = True
        planner.ray_cost_enabled = False
    elif goal_type == "pose":
        wait_for_grasp = False
        planner.ray_cost_enabled = False
        pose = poses.pop(0)
    else:
        print("Invalid Goal Type:", goal_type)
        return None, None, None, None, None, None, None, None

    # initialize vars
    speed = rospy.get_param("/robot/vel_ang_lim")
    accel = rospy.get_param("/robot/acc_ang_lim")
    total_dur = 0
    t_call_execute = 0
    grasps = []
    old_points = None
    prev_vel = 0
    tES = 0
    t00 = time.time()
    tgt_ind = -1
    max_score = 0

    # Control Loop
    while not rospy.is_shutdown():
        t0 = time.time()
        result = copy_content_from_shared_arrays(shared_arrays)
        points, tgt_pts, tgt_rgb, all_pts, all_rgb = result
        # print('Num Target Points:', len(tgt_pts), flush=True)
        # print("Copy Perception Info")
        while len(tgt_pts) == 0 and wait_for_grasp:
            ## get latest perception info ##
            result = copy_content_from_shared_arrays(shared_arrays)
            points, tgt_pts, tgt_rgb, all_pts, all_rgb = result
        # print("Found target points from the perception pipeline")
        t1 = time.time()
        # print('Read-Share Time:', t1 - t0, flush=True)

        # print("wait for grasps", wait_for_grasp)
        ## get latest grasps ##
        if goal_type != "pose":
            while gcon.poll() or wait_for_grasp:
                grasps = gcon.recv()
                if len(grasps) > 0:
                    wait_for_grasp = False
                    planner.point_cost_enabled = True
                    ss, gs, pgs, gjs, pgjs, smps = zip(*grasps)
                    # set grasp target
                    if gs[0] is not None:
                        max_grasp = pgs[np.argmax(ss)]
                    else:
                        if time.time() - t00 > TIME_OUT:
                            print("Grasps Found", 0, sep=",", file=out_file)
                            print("Timeout", True, sep=",", file=out_file)
                            return None, None, None, None, None, None, None, None
                            # median as target
                        now_max = max(ss)
                        if now_max > max_score:
                            max_score = now_max
                            tgt_ind = -1
                        max_grasp = pgs[np.argsort(ss)[tgt_ind]]
                    print("Current Best Score:", max_score, flush=True)
                    target = max_grasp
                    goal_pos = (
                        np.array(
                            [
                                max_grasp.position.x,
                                max_grasp.position.y,
                                max_grasp.position.z,
                            ]
                        )
                        .astype(float)
                        .tolist()
                    )
                    goal_quat = (
                        np.array(
                            [
                                max_grasp.orientation.w,
                                max_grasp.orientation.x,
                                max_grasp.orientation.y,
                                max_grasp.orientation.z,
                            ]
                        )
                        .astype(float)
                        .tolist()
                    )
                    rospy.set_param("/target_position", goal_pos)
                    rospy.set_param("/target_quaternion", goal_quat)
                    break
                else:
                    if time.time() - t00 > TIME_OUT:
                        print("Grasps Found", 0, sep=",", file=out_file)
                        print("Timeout", True, sep=",", file=out_file)
                        return None, None, None, None, None, None, None, None
            if len(grasps) == 0:
                planner.point_cost_enabled = False
            else:
                if tES == 0:
                    print("Grasps Found", 1, sep=",", file=out_file)
        else:
            target = pose
            goal_pos = pose[0:3]
            goal_quat = pose[3:7]
            planner.point_cost_enabled = True
        t2 = time.time()
        print("Grasp found!, Read-Grasp Time:", t2 - t1, flush=True)

        ## motion planning ##
        # plan, raw_plan = planner.mpc_step_wrapper(
        #     joint_state=joint_state, shift_steps=10, max_attempts=1
        # )

        test_grasps = False
        while test_grasps == True:
            plan, mgr = planner.pose_motion_plan(
                joint_state, target, return_all=True, grasp=planner.point_cost_enabled
            )  # If not enabled then not using grasps
            start_time = rospy.Time.now().to_sec()
            current_time = rospy.Time.now().to_sec()
            while (current_time - start_time) < 5.0:
                current_time = rospy.Time.now().to_sec()
            print("--------------Replanning now")

        if False and old_points is not None:
            if len(to_plan_from) > 1:
                # TODO index propoerly rather than hard code
                to_plan_from[1].positions[-1], to_plan_from[1].positions[6] = (
                    to_plan_from[1].positions[6],
                    to_plan_from[1].positions[-1],
                )
                to_plan_from[2].positions[-1], to_plan_from[2].positions[6] = (
                    to_plan_from[2].positions[6],
                    to_plan_from[2].positions[-1],
                )
                to_plan_from[3].positions[-1], to_plan_from[3].positions[6] = (
                    to_plan_from[3].positions[6],
                    to_plan_from[3].positions[-1],
                )
                init_pos_1 = torch.from_numpy(to_plan_from[1].positions[6:-1]).cuda()
                init_pos_2 = torch.from_numpy(to_plan_from[2].positions[6:-1]).cuda()
                init_pos_3 = torch.from_numpy(to_plan_from[3].positions[6:-1]).cuda()
                print(
                    "init poses",
                    to_plan_from[1].positions,
                    init_pos_1,
                    init_pos_2,
                    init_pos_3,
                )
            else:
                init_pos_1 *= 0.0
                init_pos_2 *= 0.0
                init_pos_3 *= 0.0
            planner.set_init_points(
                planner.motion_gen.trajopt_solver, init_pos_1, init_pos_2, init_pos_3
            )
            planner.set_init_points(
                planner.motion_gen.finetune_trajopt_solver,
                init_pos_1,
                init_pos_2,
                init_pos_3,
            )

        valid = False
        js = copy.deepcopy(joint_state)
        end = planner.parse_joint_state(js)
        kinematics = planner.motion_gen.compute_kinematics(end)
        ee_pos = kinematics.ee_pos_seq.cpu().numpy()[-1]
        ee_quat = kinematics.ee_quat_seq.cpu().numpy()[-1]
        dist_pos = np.linalg.norm(np.array(goal_pos) - ee_pos)
        dist_quat = 1 - np.dot(goal_quat, ee_quat)
        if dist_pos < 0.01 and dist_quat < 0.001:
            # print('Already at goal!', flush=True)
            # print('Len Grasps:', len(grasps), tgt_ind, flush=True)
            tgt_ind -= 1
            if tgt_ind < -len(grasps):
                tgt_ind = -1
            continue
        if dist_pos < 0.25 and dist_quat < np.pi / 2:
            if old_points is not None and len(old_points) > 0:
                js.position = old_points[-1].positions
                end = planner.parse_joint_state(js)
                kinematics = planner.motion_gen.compute_kinematics(end)
                ee_pos = kinematics.ee_pos_seq.cpu().numpy()[-1]
                ee_quat = kinematics.ee_quat_seq.cpu().numpy()[-1]
                dist_pos = np.linalg.norm(np.array(goal_pos) - ee_pos)
                dist_quat = 1 - np.dot(goal_quat, ee_quat)
                if dist_pos < 0.01 and dist_quat < 0.001:
                    # force using old points if going to same goal
                    valid = True
                    plan0 = None
        # planner.update_world_motion_gen()
        # plan0, success = planner.cartesian_motion(
        #     joint_state,
        #     target,
        #     precise=False,
        #     return_all=True,
        # )
        # if success:
        #     valid = planner.validate_trajectory(plan0)
        plan0, mgr0 = planner.pose_motion_plan(
            joint_state,
            target,
            return_all=True,
            path_constraint="short",
        )
        valid = plan0 is not None
        if not valid:
            plan0, mgr0 = planner.pose_motion_plan(
                joint_state,
                target,
                return_all=True,
                # var_steps_scale=var_steps_scale,
            )
            # trim first point
            if plan0 is not None:
                plan0.points = plan0.points[1:]
        else:
            print("*****Shortcutting!", flush=True)
            # while len(plan0.points) > 32:
            #     plan0.points = plan0.points[:-1:2] + [plan0.points[-1]]
        t20 = time.time()
        step_dt = 0.125
        # step_dt = 0.1
        if plan0 is not None:
            plan0.points[-1].time_from_start = plan0.points[0].time_from_start
            plan0.points[-1].time_from_start += float_to_ros_duration(0)
            print("*****prev_vel:", prev_vel, speed, flush=True)
            pos, vel, ts, jt = MI.retime_trajectroy(
                plan0.points,
                step_dt,
                speed,
                accel,
                # 8*((speed*np.pi/180)**2),
                (prev_vel**2) / 8,
            )
            if ts is None:
                # debugging stuff:
                # for i in range(len(plan0.points)):
                #     plan0.points[i].time_from_start = float_to_ros_duration(i/50.)
                # planner.visualize_traj_rviz(plan0)
                # input('Vis')
                # execute(plan0, window=step_dt, retime=True, stream=True, wait=False)
                # input('runned')
                plan0 = None
            else:
                plan0.points.clear()
                for i in range(len(ts)):
                    plan0.points.append(
                        JointTrajectoryPoint(
                            positions=pos[i],
                            velocities=vel[i],
                            time_from_start=float_to_ros_duration(ts[i]),
                        )
                    )
        print("Retime time:", time.time() - t20, flush=True)
        if plan0 is None:
            # var_steps_scale = np.clip(var_steps_scale+2.0, 0.0, 6.0)
            if not valid:
                if tES == 0:
                    tES = -1
                if old_points is None or len(old_points) == 0:
                    print("Planning failed! Retrying...", flush=True)
                    continue
                now = rospy.Time.now().to_sec()
                if (now - t_call_execute) < (control_step_dt / 2):
                    print(
                        "Planning failed! But there is time to try again...", flush=True
                    )
                    continue
                print("Planning failed! Using old points...", flush=True)
            else:
                print("Using old points!", flush=True)
            plan.points = old_points
        else:
            print("New usable plan!")
            # var_steps_scale = np.clip(var_steps_scale-1.0, 0.0, 6.0)
            plan = plan0
            # mgr = mgr0
            # plan = planner.joint_trajectory_from_curobo(mgr0.interpolated_plan, step_dt, joint_state, True)
        num_steps = int(round(control_step_dt / step_dt)) + 1
        print("Num Steps:", num_steps, flush=True)

        # Plan
        old_points = copy.deepcopy(plan.points[num_steps:])
        to_plan_from = copy.deepcopy(plan.points[num_steps : num_steps + 3])
        if len(to_plan_from) == 0:
            to_plan_from = [copy.deepcopy(plan.points[-1])]
        plan.points = plan.points[1:num_steps]
        t3 = time.time()
        print("Motion Plan Time:", t3 - t2, flush=True)

        # update next joint state to plan from
        joint_state.position = to_plan_from[0].positions
        joint_state.velocity = to_plan_from[0].velocities
        joint_state.effort = to_plan_from[0].accelerations
        prev_vel = np.linalg.norm(joint_state.velocity)
        # joint_state.position = plan.points[-1].positions
        # joint_state.velocity = plan.points[-1].velocities

        # update durations
        for i in range(len(plan.points)):
            total_dur += step_dt
            plan.points[i].time_from_start = float_to_ros_duration(total_dur)
            print("Dur:", i, total_dur)

        # sleep until ready to execute plan
        # Once the plan is finishing running, just run the next one
        # (When time elapsed since last plan is greater than the time we should run it - time new motion plan needs)
        now = rospy.Time.now().to_sec()
        while (now - t_call_execute) < (control_step_dt):  # x2 only in sim!
            # print('NOW!', now - t_call_execute, control_step_dt, flush=True)
            now = rospy.Time.now().to_sec()

        t_call_execute = now
        res = execute(plan, window=step_dt, retime=False, stream=True, wait=False)
        if res[0] == ExecuteTrajectoryResponse.RESTARTED_STREAM:
            total_dur = control_step_dt
        if tES <= 0:
            tES = time.time()
            print("Time till execution start", tES - t00, sep=",", file=out_file)

        # check grasp achieved
        print(
            "----------------------------------planner.point_cost_enabled",
            planner.point_cost_enabled,
        )
        if planner.point_cost_enabled:
            end = planner.parse_joint_state(joint_state)
            kinematics = planner.motion_gen.compute_kinematics(end)
            ee_pos = kinematics.ee_pos_seq.cpu().numpy()[-1]
            ee_quat = kinematics.ee_quat_seq.cpu().numpy()[-1]
            dist_pos = np.linalg.norm(np.array(goal_pos) - ee_pos)
            dist_quat = 1 - np.dot(goal_quat, ee_quat)
            print("Dist Pos:", dist_pos)
            print("Dist Quat:", dist_quat)
            # if dist_pos < 0.001 and dist_quat < 0.31:
            # if dist_pos < 0.0001 and dist_quat < 0.0001:
            if dist_pos < 0.01 and dist_quat < 0.001:
                if goal_type == "pose":
                    input("Next?")
                    pose = poses.pop(0)
                    old_points = None
                else:
                    # only break if grasp is valid
                    if grasps[0][1] is not None:
                        break

    if grasps:
        ss, gs, pgs, gjs, pgjs, smps = (list(x) for x in zip(*grasps))
        return ss, gs, pgs, gjs, pgjs, smps, joint_state, tES
    else:
        return None, None, None, None, None, None, joint_state, tES


def find_ik_solutions(ik):
    ray_pose = rospy.get_param("/best_ray_pose", [])
    ray_pose = np.asarray(ray_pose)
    print("ray pose shape", ray_pose.shape)
    ray = rospy.get_param("/ray_largest_cone", [])
    if ray is None:
        return None
    ray = np.asarray(ray)
    torch_rays = torch.from_numpy(ray)

    # Get the ee pose for this:
    # joint_indices = list(map(joint_state.name.index, ik.joint_names))
    # joint_values = np.array(joint_state.position)[joint_indices]
    # ee_pose = ik.fk(joint_values)
    # ee_position = ee_pose[:3, 3]
    # rotation_matrix = ee_pose[:3, :3]
    # rotation = R.from_matrix(rotation_matrix)
    # #euler_angles = rotation.as_euler('xyz', degrees=True)  # Roll, Pitch, Yaw
    # quaternion = rotation.as_quat()
    # ee_position = torch.from_numpy(ee_position)

    ray_quaternion = PerceptionInterface.single_direction_to_quaternion(
        torch_rays[0] * -1.0
    )

    homogenous_matrix = tf.quaternion_matrix(ray_quaternion)
    homogenous_matrix[:3, 3] = ray_pose
    goal_joint_values = ik.ik(homogenous_matrix)

    point_tar = Point(x=ray_pose[0], y=ray_pose[1], z=ray_pose[2])
    print("ray quaternion", ray_quaternion, "ray pose", ray_pose)
    orientation_tar = Quaternion(
        x=ray_quaternion[0],
        y=ray_quaternion[1],
        z=ray_quaternion[2],
        w=ray_quaternion[3],
    )
    resulting_pose = Pose(position=point_tar, orientation=orientation_tar)

    if goal_joint_values is None:
        return None, None
    return goal_joint_values, resulting_pose


def closed_loop_pick_or_place(
    object_to_grasp,
    robot,
    gt,
    target=None,
    place=None,
    eval_xml_file="nothing",
    out_file=sys.stderr,
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

    try:
        mp.set_start_method("spawn", force=True)
        # mp.set_start_method('forkserver')
    except RuntimeError:
        print("start method already set")

    ## start perception pipeline ##
    perc_proc = Process(
        target=perception_pipeline,
        args=(object_to_grasp, robot, gt, eval_xml_file),
    )
    perc_proc.start()

    # initialize planner
    planner = robot.init_motion_planner(planner="curobo", warmup=False)

    ## start grasping pipeline ##
    gconS, gconR = Pipe()
    grasp_proc = Process(
        target=grasping_pipeline,
        args=(gconS, robot, planner.static_world_config),
    )
    grasp_proc.start()

    t0 = time.time()
    ## plan and execute loop ##
    execute(JointTrajectory())  # ensure robot is idle
    ss, gs, pgs, gjs, pgjs, smps, joint_state, tES = move_to_target(
        shared_arrays,
        gconR,
        robot,
        planner,
        # goal_type='proposed',
        # goal_type='pose',
        goal_type="baseline",
        poses=target,
        out_file=out_file,
    )
    if ss is None:
        return "timeout"

    ## switch to idle mode after point streaming ##
    t1 = time.time()
    closed_loop_execution_time = t1 - t0
    print("Closed-loop Execution Time:", closed_loop_execution_time)

    ## wait till robot stops ##
    wait_till(stop=True)
    execute(JointTrajectory())
    t22 = time.time()
    print("Time execution start to pre-grasp", t22 - tES, sep=",", file=out_file)

    ## get latest perception info ##
    result = copy_content_from_shared_arrays(shared_arrays)
    points, tgt_pts, tgt_rgb, all_pts, all_rgb = result

    ## stop subprocesses ##
    # print('Stopping Grasping subprocesses ...')
    grasp_proc.kill()
    # grasp_proc.terminate()
    grasp_proc.join()
    gconS.close()
    gconR.close()
    # print('Stopping Perception subprocesses...')
    perc_proc.kill()
    # perc_proc.terminate()
    perc_proc.join()
    # print('Perception Stopped!')

    if not robot.is_sim:
        print("Press Enter to Execute.")
        input()
        obj_name = object_to_grasp
    else:
        obj_name = 35 if type(object_to_grasp) is str else object_to_grasp

    if target is not None:
        return

    if len(gs) > 0:
        ind = np.argmax(ss)
        print("Grasp Score", ss[ind], sep=",", file=out_file)
        if sys.stderr.name != out_file.name:
            np.savetxt(out_file.name.replace(".csv", "_target_points.txt"), tgt_pts)

        ## plan to pre-grasp ##
        t0 = time.time()
        planner.set_planning_scene(
            all_pts,
            visualize=False,
            save_scene_file="/tmp/world_config.pth",
        )
        # planner.visualize_rviz()
        # planner.visualize_traj_rviz(plan1)
        # plan1 = planner.pose_motion_plan(
        #     joint_state,
        #     pgs[ind],
        # )
        # t1 = time.time()
        # print('Plan1 Time:', t1 - t0)
        # if plan1 is None:
        #     ss.pop(ind)
        #     gs.pop(ind)
        #     pgs.pop(ind)
        #     gjs.pop(ind)
        #     pgjs.pop(ind)
        #     return 'approach failed!'

        ## plan approach ##
        t2 = time.time()
        planner.set_planning_scene(None)
        # planner.visualize_rviz()
        # planner.visualize_spheres_rviz(joint_state)
        # planner.visualize_spheres_rviz(gjs[ind])
        # joint_state2 = JointState()
        # joint_state2.name = plan1.joint_names
        # joint_state2.position = plan1.points[-1].positions
        # plan2 = planner.pose_motion_plan(
        #     joint_state,
        #     gs[ind],
        #     path_constraint=[1, 1, 1, 1, 1, 0],
        #     constraint_in_goal_frame=True,
        # )
        plan2, success = planner.cartesian_motion(
            joint_state,
            gs[ind],
            # offset=[0, 0, 0.005, 1, 0, 0, 0],
            return_all=True,
        )
        t3 = time.time()
        print("Plan2 Time:", t3 - t2)
        if plan2 is None or not success:
            ss.pop(ind)
            gs.pop(ind)
            pgs.pop(ind)
            gjs.pop(ind)
            pgjs.pop(ind)
            return "approach failed!"

        ## plan lift ##
        joint_state3 = JointState()
        joint_state3.name = plan2.joint_names
        joint_state3.position = plan2.points[-1].positions
        grasp_state = joint_state3
        # plan3 = planner.pose_motion_plan(
        #     joint_state3,
        #     gs[ind],
        #     path_constraint=[1, 1, 1, 1, 1, 0],
        #     constraint_in_goal_frame=False,
        #     offset=[0, 0, lift_height, 1, 0, 0, 0],
        #     visualize=False,
        # )
        plan3, success = planner.cartesian_motion(
            joint_state3,
            gs[ind],
            offset=[0, 0, lift_height, 1, 0, 0, 0],
            constraint_in_goal_frame=False,
            return_all=True,
        )
        t4 = time.time()
        print("Plan3 Time:", t4 - t3)
        if plan3 is None or not success:
            ss.pop(ind)
            gs.pop(ind)
            pgs.pop(ind)
            gjs.pop(ind)
            pgjs.pop(ind)
            return "lift failed!"

        if place:
            ## plan to place ##
            joint_state4 = JointState()
            joint_state4.name = plan3.joint_names
            joint_state4.position = plan3.points[-1].positions

            ## get estimated target mesh ##
            tgt_mesh = PerceptionInterface.get_shape_estimate(
                tgt_pts,
                # tgt_rgb,
            )

            # filter all_pts by robot and attach object
            planner.set_planning_scene(
                all_pts,
                tgt_mesh,
                grasp_state,
                attach_zoffset=0,
                filter_js=joint_state4,
                visualize=False,
            )
            planner.visualize_rviz()
            planner.visualize_spheres_rviz(joint_state4)
            t0 = time.time()
            print("Plan4 Scene Time:", t0 - t4)

            grasp_rot = pose_to_matrix(gs[ind])[:3, :3]
            place_rot0 = tf.quaternion_matrix(place[3:7])[:3, :3]
            place_alt = np.multiply([1, 1, 1, 1, 1, -1, -1], place)
            place_rot1 = tf.quaternion_matrix(place_alt[3:7])[:3, :3]
            norm0 = np.linalg.norm(grasp_rot - place_rot0)
            norm1 = np.linalg.norm(grasp_rot - place_rot1)
            print("Dist to", place, " :", norm0)
            print("Dist to", place_alt, " :", norm1)
            min_ind = np.argmin([norm0, norm1])
            place = np.array([place, place_alt])[min_ind]

            plan4 = planner.pose_motion_plan(
                joint_state4,
                place,
            )
            t5 = time.time()
            print("Plan4 Time:", t5 - t4)
            if plan4 is None:
                ss.pop(ind)
                gs.pop(ind)
                pgs.pop(ind)
                gjs.pop(ind)
                pgjs.pop(ind)
                return "retract failed!"

            plans = [plan2, plan3, plan4]
        else:
            plans = [plan2, plan3]
    else:
        assert False  # branch should not execute

    t0 = time.time()
    for i, plan in enumerate(plans):
        if i == 1:
            ee_close()
        if type(plan) is JointTrajectory:
            # print('Plan success!')
            execute(plan, window=0.1, wait=True, retime=True)
            execute(JointTrajectory())
            # execute(plan, window=0)
        else:
            # print('Plan', i, 'error:', plan)
            break
        if i == 1:
            s, g, d = get_experiment_result(obj_name, sim=robot.is_sim)
            print("Grasp Success", s, sep=",", file=out_file)
            print(g)
            if len(g) == 0:
                g = [""]
            gs = '"' + ",".join(g) + '"'
            print("Grasped", gs, sep=",", file=out_file)

    t1 = time.time()
    open_loop_execution_time = t1 - t0
    print("Open-loop Execution Time:", open_loop_execution_time)
    print("Time pre-grasp to retract", t1 - t2, sep=",", file=out_file)

    s, g, d = get_experiment_result(obj_name, sim=robot.is_sim)
    print("Retract Success", s, sep=",", file=out_file)
    if len(d) == 0:
        d = [""]
    ds = '"' + ",".join(d) + '"'
    print("Dropped", ds, sep=",", file=out_file)
    print("Timeout", False, sep=",", file=out_file)
    return "finished"


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
    object_to_grasp = sys.argv[3] if len(sys.argv) > 3 else 35
    place = (
        None
        if len(sys.argv) > 4
        and sys.argv[4][0]
        in (
            "n",
            "N",
            "f",
            "F",
            "0",
        )
        else [0.5, 0, 1.2, 0.5, -0.5, 0.5, 0.5]
    )
    out_file = open(sys.argv[5], "a") if len(sys.argv) > 5 else sys.stderr
    eval_xml_file = sys.argv[6] if len(sys.argv) > 6 else "nothing"
    body_name = sys.argv[7] if len(sys.argv) > 7 else False
    run_only_perception = sys.argv[8] if len(sys.argv) > 8 else False

    print(
        "Args: is_sim (s or r), gt (g or r), object_to_grasp (id sim or name real), place (p), stats_file (path), xml_file (path or 'nothing'), body_name (name of object->> if xml file is nothing this is irrelevant), run_only_perception (True or blank)"
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
        result = closed_loop_pick_or_place(
            object_to_grasp, robot, gt, None, place, eval_xml_file, out_file
        )  # python closed_loop.py s g 36 p "../../xmls/adjusted_pos_ycb_boxes.xml"
        if result != "finished":
            print("Error", result, sep=",", file=out_file)
        out_file.flush()
        out_file.close()
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

    shares = []
    for name in shared_arr_names:
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()
    os.system(f"kill -9 {os.getpid()}")
