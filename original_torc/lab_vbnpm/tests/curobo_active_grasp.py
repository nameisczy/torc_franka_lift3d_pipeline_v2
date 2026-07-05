import sys
import copy
import time
import torch
import numpy as np
import open3d as o3d
import transformations as tf
from scipy.spatial import KDTree

import rospy
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import CameraInfo, JointState, PointCloud2

from utils.print_color import *
from utils.conversions import pose_to_matrix
from perception.perception_fast import PerceptionInterface
from grasp_planner.curobo_grasp_planner import GraspPlanner
from task_planner.curobo_open_loop import update_and_get_points
from task_planner.eutils import ee_open, ee_close, execute, get_experiment_result
from utils.conversions import pose_to_matrix, matrix_to_pose, float_to_ros_duration
import trimesh  # must import after perception to avoid conflict

# TODO: why is above true‽

from tracikpy import TracIKSolver

TIME_OUT = 300  # seconds
lift_height = 0.02


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


def active_grasp_views(
    object_to_grasp,
    robot,
    perception,
    planner,
    grasp_planner,
    tgt_pts,
    target_augment_scale=0,  # scale factor to augment convex hull of points
    visualize=False,
    downsample=1,
    bbox_size=np.array([0.3, 0.3, 0.3]),
    tsdf_resolution=0.0075,
):

    t0 = time.time()

    ik = TracIKSolver(robot.urdf, "base_link", robot.camera_link)

    tgt_mesh = PerceptionInterface.get_shape_estimate(tgt_pts)
    tgt_center = np.mean(tgt_mesh.vertices, axis=0)

    center = np.mean(tgt_pts, axis=0)
    center = tgt_center
    radius = 0.5 * (max(tgt_pts[:, 2]) - min(tgt_pts[:, 2]))  # + 0.28
    # radii = [radius + 0.28]
    GRIPPER_TO_CAMERA_LEN = 0.12  # 0.28
    num_radii = 4
    # radii = np.linspace(GRIPPER_TO_CAMERA_LEN, 0.56, num_radii) + radius
    radii = np.linspace(0.28, 0.56, 2) + radius
    angles = [7.5, 10, 15, 30, 45, 60, 75, 80, 85, 87.5]

    c_valid_views = []
    c_valid_poses = []

    t_views_start = time.time()
    for adj_radius in radii:
        views = PerceptionInterface.sample_views(center, [adj_radius], angles)
        print("[k_user] Num sampled views: ", len(views), flush=True)

        ik_valid_views = []
        t1 = time.time()
        for cam_view in views:
            pose0 = pose_to_matrix(cam_view)
            for z_rotation in [np.pi, 0, np.pi / 2, 3 * np.pi / 2]:
                z_quat = np.concatenate(
                    [[np.cos(z_rotation)], np.sin(z_rotation) * np.array([0, 0, 1])]
                )
                z_transform = tf.quaternion_matrix(z_quat)

                # Obtain rotated camera pose
                cam_view = pose0 @ z_transform

                if ik.ik(cam_view) is not None:
                    ik_valid_views.append(cam_view)
                    break

        print("[k_user] Num ik valid views:", len(ik_valid_views), flush=True)
        # c_valid_views = []
        # c_valid_poses = []
        t1 = time.time()
        for cam_view in ik_valid_views:
            # c_valid_views.append(view)
            # continue
            # cam_view = pose_to_matrix(view)

            # Convert rotated camera pose to end-effector pose
            # link7 wrt camera
            # end-effector wrt link7
            # ee_pose = cam_view @ perception.cam2link[perception.camera_prefixes[1]]
            # ee_pose = ee_pose @ np.array([
            #     [0, -1, 0, 0],
            #     [-1, 0, 0, 0],
            #     [0, 0, -1, 0.152],
            #     [0, 0, 0, 1]
            # ])

            rel = np.array(
                [
                    [
                        0.0,
                        -1.0,
                        0.0,
                        0.061,
                    ],
                    [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        1.0,
                        -0.112,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                ]
            )

            # ee_pose = cam_view @ rel

            ee_pose = cam_view @ np.linalg.pinv(rel)
            # ee_pose = ee_pose @ np.array([
            #     [0, 1, 0, 0],
            #     [-1, 0, 0, 0],
            #     [0, 0, 1, 0.152],
            #     [0, 0, 0, 1]
            # ])

            if grasp_planner.valid_pose(ee_pose):
                c_valid_views.append(matrix_to_pose(cam_view))
                c_valid_poses.append(matrix_to_pose(ee_pose))
                # break
                # vis_view(view)
                # input('rotate')
        # print('BioIk time: ', time.time() - t1, flush=True)
        print("[k_user] Num collision valid views: ", len(c_valid_views), flush=True)
        if len(c_valid_views) > 0:
            break

    # c_valid_views_arr = np.array([pose_to_matrix(view) for view in c_valid_views])
    # c_valid_poses_arr = np.array([pose_to_matrix(pose) for pose in c_valid_poses])
    # np.save("/home/lab/motoman_ws/src/lab_vbnpm/valid_views.npy", c_valid_views_arr)
    # np.save("/home/lab/motoman_ws/src/lab_vbnpm/valid_poses.npy", c_valid_poses_arr)

    t1 = time.time()

    print("[k_user] active_grasp: view sampling + validation took", t1 - t_views_start)

    # Using percentiles to be robust to outliers
    # bbox_min = np.percentile(tgt_pts, 1, axis=0)
    # bbox_max = np.percentile(tgt_pts, 99, axis=0)

    # Note: we can either use tgt_mesh.centroid or np.mean(tgt_mesh.vertices, axis=0)
    # Empirically, these points were around one centimeter apart
    # tgt_center = tgt_mesh.centroid
    bbox_min = tgt_center - (bbox_size / 2 - tsdf_resolution / 2)
    bbox_max = tgt_center + (bbox_size / 2 - tsdf_resolution / 2)

    print("[k_user] active_grasp: computing information gain")

    use_active_grasp_warmup = True
    active_grasp_warmup_time = 0
    if use_active_grasp_warmup and len(c_valid_views) > 0:
        t_warmup0 = time.time()
        perception.get_info_gain2(
            robot.camera[1],
            [c_valid_views[0]],
            bbox=(bbox_min, bbox_max),
            downsample=downsample,
            ACTIVE_GRASP_VOXEL_SIZE=tsdf_resolution,
        )
        active_grasp_warmup_time = time.time() - t_warmup0

    t00 = time.time()

    scores, min_max = perception.get_info_gain2(
        robot.camera[1], c_valid_views, bbox=(bbox_min, bbox_max), downsample=downsample
    )
    t01 = time.time()
    print("[k_user] active_grasp: computed information gain in", t01 - t00, "seconds")
    # scores = perception.get_info_gain3(
    #     points, all_pts, cam_intr, c_valid_views,
    # )

    print(
        f"[k_user] active_grasp baseline took {t01 - t0 - active_grasp_warmup_time} seconds total."
    )

    return scores, c_valid_views, c_valid_poses, min_max


def sense_plan_act(
    object_to_grasp,
    robot,
    perception,
    planner,
    grasp_planner,
    target_augment_scale=0,  # scale factor to augment convex hull of points
    visualize=False,
    out_file=sys.stderr,
):
    debug_target_pub = rospy.Publisher(
        "/debug/target_points", PointCloud2, queue_size=3, latch=True
    )
    debug_full_pcd = rospy.Publisher(
        "/debug/full_pcd", PointCloud2, queue_size=3, latch=True
    )
    debug_surface_pcd = rospy.Publisher(
        "/debug/surface_pcd", PointCloud2, queue_size=3, latch=True
    )
    t000 = time.time()

    print("sense_plan_act: Sensing the environment!")
    ## sense the environment ##
    result = update_and_get_points(object_to_grasp, robot, perception)
    points, tgt_pts, tgt_rgb, all_pts, all_rgb, _ = result
    if len(tgt_pts) == 0:
        tgt_pts = np.mean(all_pts, axis=0).reshape((1, 3))

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
        np.linspace([0, 0, 0], [255, 255, 255], len(all_pts)),
        255,
        "world",
        rospy.Time.now(),
    )
    debug_full_pcd.publish(full_msg)

    ## augment target points ##
    if target_augment_scale:
        t5 = time.time()
        t05 = time.time()
        if visualize:
            trimesh.points.PointCloud(tgt_pts, tgt_rgb).show()
        t06 = time.time()
        tgt_mesh = perception.get_shape_estimate(
            tgt_pts,
            tgt_rgb,
            scale=target_augment_scale,
        )
        tgt_pts = trimesh.sample.sample_surface_even(tgt_mesh, 1000)[0]
        tgt_rgb = 100 * np.ones((tgt_pts.shape[0], 3))
        t6 = time.time()
        print("Augmentation Time:", (t6 - t5) - (t06 - t05))

    t01 = time.time()
    if visualize:
        if len(tgt_pts) > 0:
            trimesh.points.PointCloud(tgt_pts, tgt_rgb).show()
        trimesh.points.PointCloud(all_pts, all_rgb).show()
        # input('Continue?')
    t02 = time.time()

    joint_state = rospy.wait_for_message("/joint_states_all", JointState, timeout=5)

    planner.set_planning_scene(
        all_pts,
        visualize=False,
    )

    predefined_pose = False
    if predefined_pose:
        goal_pose = (
            torch.tensor([0.8, -0.2, 1.25]),  # Position
            torch.tensor([0, 0.923879533, 0, 0.382683432]),  # Quaternion
        )

        goal = goal_pose
    else:
        grasp_planner.set_collision_scene(all_pts)

        scores, views, poses, min_max = active_grasp_views(
            object_to_grasp,
            robot,
            perception,
            planner,
            grasp_planner,
            tgt_pts,
            downsample=10,
        )
        num_views_to_plan = 1

        from grasp_planner.grasp_plotter import GraspPlotter

        plotter = GraspPlotter()
        plotter.draw_grasps(poses, scores, normalize=True, min_max=min_max)
        # view_arr = np.array([pose_to_matrix(view) for view in views])
        # np.save("/data/local/kc1317/workspace/views.npy", view_arr)

        print("Views:", len(views))
        print("Poses:", len(poses))
        sorted_score_idx = np.argsort(scores)[::-1]
        views_to_plan_idx = sorted_score_idx[:num_views_to_plan]
        views_to_plan = [poses[i] for i in views_to_plan_idx]
        goal = views_to_plan

    t0 = time.time()
    plan = planner.pose_motion_plan(
        joint_state,
        goal,
    )
    t1 = time.time()
    print("Sense-Plan-Act Time:", t1 - t0)
    if plan is None:
        raise KeyboardInterrupt()

    print("Time till execution start 1", t1 - t000, sep=",", file=out_file)

    if type(plan) is JointTrajectory:
        # print('Plan success!')
        if visualize:
            planner.visualize_traj_rviz(plan)
        # input('run?')
        execute(plan, window=0.1, wait=True, retime=True)
        # execute(plan, window=0)
        # input('Continue?')

    t9 = time.time()
    print("Time execution start 1 to view point", t9 - t1, sep=",", file=out_file)


def open_loop_pick_or_place(
    object_to_grasp,
    robot,
    perception,
    planner,
    grasp_planner,
    target_augment_scale=0,  # scale factor to augment convex hull of points
    target_augment_scale_dim=2,  # consider distane in 2 or 3 dimensions?
    place=None,
    visualize=False,
    save_debug=False,
    out_file=sys.stderr,
):
    debug_target_pub = rospy.Publisher(
        "/debug/target_points", PointCloud2, queue_size=3, latch=True
    )
    debug_full_pcd = rospy.Publisher(
        "/debug/full_pcd", PointCloud2, queue_size=3, latch=True
    )
    debug_surface_pcd = rospy.Publisher(
        "/debug/surface_pcd", PointCloud2, queue_size=3, latch=True
    )
    ## NOTE: t0 is used for timeout calculation when finding grasps
    ## thus, we exclude sense-plan-act from t0, as it is not directly related to the grasp planning
    ## open-loop baseline: move to visibility pose
    sense_plan_act(
        object_to_grasp,
        robot,
        perception,
        planner,
        grasp_planner,
        target_augment_scale=target_augment_scale,
        visualize=visualize,
        out_file=out_file,
    )

    # print('Staring!')
    t0 = time.time()
    grasp_time = 0
    mplan_time = 0

    ## sense the environment ##
    result = update_and_get_points(object_to_grasp, robot, perception)
    points, tgt_pts, tgt_rgb, all_pts, all_rgb, _ = result

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
        np.linspace([0, 0, 0], [255, 255, 255], len(all_pts)),
        255,
        "world",
        rospy.Time.now(),
    )
    debug_full_pcd.publish(full_msg)

    ## augment target points ##
    if target_augment_scale:
        t5 = time.time()
        t05 = time.time()
        if visualize:
            trimesh.points.PointCloud(tgt_pts, tgt_rgb).show()
        t06 = time.time()
        tgt_mesh = perception.get_shape_estimate(
            tgt_pts,
            tgt_rgb,
            scale=target_augment_scale,
        )
        tgt_pts = trimesh.sample.sample_surface_even(tgt_mesh, 1000)[0]
        tgt_rgb = 100 * np.ones((tgt_pts.shape[0], 3))
        t6 = time.time()
        print("Augmentation Time:", (t6 - t5) - (t06 - t05))

    t01 = time.time()
    if visualize:
        if len(tgt_pts) > 0:
            trimesh.points.PointCloud(tgt_pts, tgt_rgb).show()
        trimesh.points.PointCloud(all_pts, all_rgb).show()
        # input('Continue?')
    t02 = time.time()

    joint_state = rospy.wait_for_message("/joint_states_all", JointState, timeout=5)

    ## generate grasps ##
    grasp = []
    while len(grasp) == 0 or max(score) < 0.3:
        planner.set_planning_scene(
            all_pts,
            visualize=False,
        )
        # planner.visualize_rviz()
        t1 = time.time()
        grasp_res = grasp_planner.get_grasp_poses(
            tgt_pts,
            tgt_rgb,
            points,
            all_pts,
            visualize=True,
            # rviz_func=planner.visualize_spheres_rviz,
        )
        if len(grasp_res) == 7:
            grasp, p_grasp, grasp_js, p_grasp_js, score, sample, _ = grasp_res
            if grasp:
                grasp_planner.plotter.draw_grasps()
                grasp_planner.plotter.draw_grasps(grasp, score)
                g_color = [0.0] * len(score)
                g_color[np.argmax(score)] = -1.0
                grasp_planner.plotter.draw_grasps(p_grasp, g_color)
        t2 = time.time()
        grasp_time += t2 - t1
        print("Grasp Planning Time:", t2 - t1)

        if len(grasp) == 0 or max(score) < 0.3:
            if time.time() - t0 > TIME_OUT:
                print("Grasps Found", 0, sep=",", file=out_file)
                return "timeout"
            else:
                continue

        print("Grasps Found", len(grasp), sep=",", file=out_file)

        ind = None
        while len(grasp) > 0:
            if time.time() - t0 > TIME_OUT:
                return "timeout"

            t1 = time.time()
            ## plan to target ##
            planner.set_planning_scene(
                all_pts,
                visualize=False,
            )
            # planner.visualize_rviz()
            t101 = time.time()
            print("Plan1 Scene Time:", t101 - t1)
            ind = np.argmax(score)
            grasp_planner.plotter.draw_grasps()
            grasp_planner.plotter.draw_grasps(grasp, score)
            g_color = [0.0] * len(score)
            g_color[ind] = -1.0
            grasp_planner.plotter.draw_grasps(p_grasp, g_color)
            # plan1 = planner.joint_motion_plan(
            #     joint_state,
            #     p_grasp_js[ind],
            # )
            plan1 = planner.pose_motion_plan(
                joint_state,
                p_grasp[ind],
            )
            t10 = time.time()
            print("Plan1 Time:", t10 - t101)
            if plan1 is None:
                # js = JointState()
                # js.name = planner.motion_gen.joint_names
                # js.position = p_grasp_js[ind]
                # planner.visualize_spheres_rviz(js)
                # input('Continue?')
                grasp.pop(ind)
                p_grasp.pop(ind)
                grasp_js.pop(ind)
                p_grasp_js.pop(ind)
                score.pop(ind)
                ind = None
                t2 = time.time()
                mplan_time += t2 - t1
                continue

            ## get estimated target mesh ##
            tgt_mesh = PerceptionInterface.get_shape_estimate(
                tgt_pts,
                # tgt_rgb,
            )

            ## plan approach ##
            joint_state2 = JointState()
            joint_state2.name = plan1.joint_names
            joint_state2.position = plan1.points[-1].positions
            # grasp_position = pose_to_matrix(grasps[ind])[:3, 3]
            # pre_grasp_pose = pose_to_matrix(pre_grasps[ind])[:3, 3]
            # dist = np.linalg.norm(grasp_position - pre_grasp_pose)
            # print('New Pre-grasp Distance:', dist)

            # planner.set_planning_scene(
            #     None,
            #     tgt_mesh,
            #     grasp_js[ind],
            #     attach_zoffset=0.01,
            # )
            # planner.visualize_spheres_rviz(grasp_js[ind])
            # input('Continue?')

            # planner.visualize_rviz()
            # plan2 = planner.joint_motion_plan(
            #     joint_state2,
            #     grasp_js[ind],
            #     path_constraint=[1, 1, 1, 1, 1, 0],
            #     constraint_in_goal_frame=True,
            # )
            # plan2 = planner.pose_motion_plan(
            #     joint_state2,
            #     grasp[ind],
            #     path_constraint=[0.9, 0.9, 0.9, 0.9, 0.9, 0],
            #     # path_constraint=[1, 1, 1, 1, 1, 0],
            #     constraint_in_goal_frame=True,
            # )
            plan2, success = planner.cartesian_motion(
                joint_state2,
                grasp[ind],
                # offset=[0, 0, 0.005, 1, 0, 0, 0],
                return_all=True,
            )
            # planner.visualize_traj_rviz(plan2)
            t11 = time.time()
            print("Plan2 Time:", t11 - t10)
            # input('Continue?')
            if plan2 is None or not success:
                grasp.pop(ind)
                p_grasp.pop(ind)
                grasp_js.pop(ind)
                p_grasp_js.pop(ind)
                score.pop(ind)
                ind = None
                t2 = time.time()
                mplan_time += t2 - t1
                continue

            ## plan lift ##
            joint_state3 = JointState()
            joint_state3.name = plan2.joint_names
            joint_state3.position = plan2.points[-1].positions
            grasp_state = joint_state3
            # plan3 = planner.pose_motion_plan(
            #     joint_state3,
            #     grasp[ind],
            #     path_constraint=[0.9, 0.9, 0.9, 0.9, 0.9, 0],
            #     # path_constraint=[1, 1, 1, 1, 1, 0],
            #     constraint_in_goal_frame=False,
            #     offset=[0, 0, lift_height, 1, 0, 0, 0],
            #     visualize=False,
            # )
            plan3, success = planner.cartesian_motion(
                joint_state3,
                grasp[ind],
                offset=[0, 0, lift_height, 1, 0, 0, 0],
                constraint_in_goal_frame=False,
                return_all=True,
            )
            # planner.visualize_traj_rviz(plan3)
            t12 = time.time()
            print("Plan3 Time:", t12 - t11)
            if plan3 is None or not success:
                grasp.pop(ind)
                p_grasp.pop(ind)
                grasp_js.pop(ind)
                p_grasp_js.pop(ind)
                score.pop(ind)
                ind = None
                t2 = time.time()
                mplan_time += t2 - t1
                continue

            t2 = time.time()
            mplan_time += t2 - t1
            if place:
                t1 = time.time()
                ## plan to place ##
                joint_state4 = JointState()
                joint_state4.name = plan3.joint_names
                joint_state4.position = plan3.points[-1].positions

                # filter all_pts by robot and attach object
                planner.set_planning_scene(
                    all_pts,
                    tgt_mesh,
                    grasp_state,
                    attach_zoffset=0,
                    filter_js=joint_state4,
                    visualize=False,
                )
                t101 = time.time()
                print("Plan4 Scene Time:", t101 - t1)
                # planner.visualize_rviz()
                # planner.visualize_spheres_rviz(joint_state4)

                grasp_rot = pose_to_matrix(grasp[ind])[:3, :3]
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
                t20 = time.time()
                print("Plan4 Time:", t20 - t101)
                if plan4 is None:
                    grasp.pop(ind)
                    p_grasp.pop(ind)
                    grasp_js.pop(ind)
                    p_grasp_js.pop(ind)
                    score.pop(ind)
                    ind = None
                    t2 = time.time()
                    mplan_time += t2 - t1
                    continue

                t2 = time.time()
                mplan_time += t2 - t1
                plans = [plan1, plan2, plan3, plan4]
            else:
                plans = [plan1, plan2, plan3]
            break

    t1 = time.time()
    print("Chosen Grasp Score:", score[ind])
    print("Total Grasp Planning Time:", grasp_time)
    print("Total MotionPlanning Time:", mplan_time)
    print("Total Computation Time:", (t1 - t0) - (t02 - t01))
    print("Time view point to execution start 2", t1 - t0, sep=",", file=out_file)
    print("Grasp Score", score[ind], sep=",", file=out_file)
    if sys.stderr.name != out_file.name:
        np.savetxt(out_file.name.replace(".csv", "_target_points.txt"), tgt_pts)

    planner.set_planning_scene(
        all_pts,
        visualize=False,
        save_scene_file="/tmp/world_config.pth",
    )
    planner.visualize_rviz()
    # planner.visualize_traj_rviz(plan1)
    if not robot.is_sim:
        print("Press Enter to Execute.")
        input()
        obj_name = object_to_grasp
    else:
        obj_name = 35 if type(object_to_grasp) is str else object_to_grasp

    t0 = time.time()
    for i, plan in enumerate(plans):
        if i == 0:
            ee_open()
        elif i == 2:
            ee_close()

        if type(plan) is JointTrajectory:
            # print('Plan success!')
            planner.visualize_traj_rviz(plan)
            # input('run?')
            execute(plan, window=0.1, wait=True, retime=True)
            # execute(plan, window=0)
            # input('Continue?')
        else:
            # print('Plan', i, 'error:', plan)
            break

        if i == 0:
            t2 = time.time()
            print(
                "Time execution start 2 to pre-grasp", t2 - t0, sep=",", file=out_file
            )
        elif i == 2:
            s, g, d = get_experiment_result(obj_name, sim=robot.is_sim)
            print("Grasp Success", s, sep=",", file=out_file)
            if len(g) == 0:
                g = [""]
            gs = '"' + ",".join(g) + '"'
            print("Grasped", gs, sep=",", file=out_file)

    t1 = time.time()
    print("Total Execution Time:", t1 - t0)
    print("Time pre-grasp to retract", t1 - t2, sep=",", file=out_file)

    s, g, d = get_experiment_result(obj_name, sim=robot.is_sim)
    print("Retract Success", s, sep=",", file=out_file)
    if len(d) == 0:
        d = [""]
    ds = '"' + ",".join(d) + '"'
    print("Dropped", ds, sep=",", file=out_file)

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
        else [0.6, 0, 1.1, 0.5, -0.5, 0.5, 0.5]
    )
    # place = [0.6, 0, 1.1] if len(sys.argv) > 4 else None
    out_file = open(sys.argv[5], "a") if len(sys.argv) > 5 else sys.stderr

    print("is_sim: ", is_sim)
    print("gt: ", gt)
    ## init perception and planning interfaces ##
    rospy.init_node("planning")
    t0 = time.time()
    from task_planner.motoman import MotomanSDA10F

    robot = MotomanSDA10F(is_sim, gt)
    perception = robot.init_perception_interface()
    planner = robot.init_motion_planner(planner="curobo")
    grasp_planner = GraspPlanner(
        robot.curobo_config,
        planner.static_world_config,
        robot.urdf,
        ignore_collision_ee_links=robot.ignore_collision_ee_links,
    )
    # grasp_result = grasp_planner.get_grasp_poses([], None, [], [])
    t1 = time.time()
    print("Init Time:", t1 - t0)

    t00 = time.time()
    # test_pose = tf.quaternion_matrix(np.array([0, 0.923879533, 0, 0.382683432]))
    # test_pose[:3, 3] = [0.8, -0.2, 1.25]
    # perception.get_info_gain2(camera_name=robot.camera[1], camera_poses = [test_pose], downsample=20)
    # perception.compile(robot.camera[1])
    t001 = time.time()

    print("[k_user] active_grasp: warmed up in", t001 - t00, "seconds")

    result = open_loop_pick_or_place(
        object_to_grasp,
        robot,
        perception,
        planner,
        grasp_planner,
        target_augment_scale=0,
        target_augment_scale_dim=2,
        place=place,
        visualize=False,
        out_file=out_file,
    )

    if result == "timeout":
        print("Timeout", True, sep=",", file=out_file)
    else:
        print("Timeout", False, sep=",", file=out_file)
