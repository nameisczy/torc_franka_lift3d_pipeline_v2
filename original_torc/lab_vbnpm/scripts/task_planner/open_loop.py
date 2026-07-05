import sys
import copy
import time
import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from tracikpy import TracIKSolver

import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from utils.print_color import *
from utils.conversions import pose_to_matrix
from grasp_planner.grasp_planner import GraspPlanner
# from grasp_planner.grasp_planner_collision_checking import GraspPlannerHPPFCL as GraspPlanner
from task_planner.eutils import ee_open, ee_close, execute
from perception.perception_fast import PerceptionInterface

lift_height = 0.05


def update_and_get_points(
    object_to_grasp,
    robot,
    perception,
    filter_points=True,
    camera_inds=[0, 1],
    save_debug=False,
    debug_number=0,
):
    t0 = time.time()
    # print('0', flush=True)
    for i in camera_inds:
        perception.updated_fused_points(robot.camera[i], object_to_grasp)
    # print('2', flush=True)
    points, colors = perception.get_fused_point_cloud()
    # print('3', flush=True)
    #bg_points, bg_colors = perception.get_fused_bg_point_cloud()
    # print('4', flush=True)
    t2 = time.time()

    perception.rays_from_occlusion()

    # print('5', flush=True)
    for i in camera_inds:
        perception.update_occlusion(robot.camera[i])
    # print('6', flush=True)
    occluded_points = perception.get_occlusion_points()
    occluded_colors = np.zeros((*occluded_points.shape[:-1], 3))
    occluded_colors[:] = [1, 0, 1]
    t3 = time.time()
    # print('7', flush=True)

    # filter combined point cloud
    all_pts = np.concatenate([points, occluded_points])
    all_rgb = np.concatenate([colors, occluded_colors])
    if filter_points:
        # print('f7', flush=True)
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(all_pts)
        pcl.colors = o3d.utility.Vector3dVector(all_rgb)
        # pcl, ind = pcl.remove_radius_outlier(10, 0.004)
        pcl, ind = pcl.remove_radius_outlier(10, 0.008)
        pcl, ind = pcl.remove_statistical_outlier(20, 3)
        all_pts = np.array(pcl.points)
        all_rgb = np.array(pcl.colors)
        if perception.old_perception == True:
            tgt_pts, tgt_rgb = perception.get_fused_target_point_cloud()
        else:
            tgt_pts, tgt_rgb = perception.get_largest_target_cluster_simple()
            #perception.get_cluster_on_best_spot()
            #get_largest_target_cluster()#.get_fused_target_point_cloud()
        #tgt_pts, tgt_rgb = perception.get_fused_target_point_cloud()
    else:
        # print('f8', flush=True)
        if perception.old_perception == True:
            tgt_pts, tgt_rgb = perception.get_fused_target_point_cloud()
        else:
            tgt_pts, tgt_rgb = perception.get_largest_target_cluster_simple()
            #perception.get_cluster_on_best_spot()
        #tgt_pts, tgt_rgb = perception.get_large_enough_target_clusters(5)#get_largest_target_cluster()#.get_fused_target_point_cloud()
        #tgt_pts, tgt_rgb = perception.get_fused_target_point_cloud()

    # down sample
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    resolution = 0.1
    downpcd = pcd.voxel_down_sample(resolution)
    ds_all_pts = np.array(downpcd.points)

    t1 = time.time()

    if save_debug == 'video':
        cameras = [robot.camera[i] for i in camera_inds]
        perception.save_image('/tmp/recording_0000', cameras, debug_number)
    elif save_debug == 'pcd' and len(tgt_pts) > 0:
        perception.save_fusion(
            '/tmp/pcd_0000', tgt_pts, tgt_rgb, 'target', debug_number
        )
        perception.save_fusion(
            '/tmp/pcd_0000', points, colors, 'surface', debug_number
        )
        perception.save_fusion(
            '/tmp/pcd_0000', all_pts, all_rgb, 'all', debug_number
        )

    # print('Fusion Time:', t2 - t0, flush=True)
    # print('Occlusion Time:', t3 - t2, flush=True)
    # print('Filtering Time:', t1 - t3, flush=True)
    # print('Total Perception Time:', t1 - t0, flush=True)
    return points, tgt_pts, tgt_rgb, all_pts, all_rgb, ds_all_pts


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
):
    # print('Staring!')
    t0 = time.time()
    grasp_time = 0
    mplan_time = 0

    ## sense the environment ##
    result = update_and_get_points(object_to_grasp, robot, perception)
    points, tgt_pts, tgt_rgb, all_pts, all_rgb, ds_all_pts = result

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
        print('Augmentation Time:', (t6 - t5) - (t06 - t05))

    t01 = time.time()
    if visualize:
        if len(tgt_pts) > 0:
            trimesh.points.PointCloud(tgt_pts, tgt_rgb).show()
        trimesh.points.PointCloud(all_pts, all_rgb).show()
        # input('Continue?')
    t02 = time.time()

    joint_state = rospy.wait_for_message(
        '/joint_states_all', JointState, timeout=5
    )

    ik = planner.ik_for_ees[robot.gripper_link]
    ik2 = TracIKSolver(robot.urdf, "base_link", "arm_right_link_7_t")
    ik3 = TracIKSolver(robot.urdf, "base_link", "arm_right_link_6_b")
    ## generate grasps ##
    grasps = []
    while len(grasps) == 0:
        t1 = time.time()
        planner.reset()
        indices = np.random.choice(
            len(tgt_pts), min(100, len(tgt_pts)), replace=False
        )
        grasps, pre_grasps, scores, samples = grasp_planner.get_grasp_poses(
            tgt_pts[indices],
            tgt_rgb[indices],
            planner.ik_for_ees[robot.gripper_link].ik,
            collision_voxel_tuple=(points, all_pts, 0.005),
            # collision_voxel_tuple=(tgt_pts, all_pts, 0.01),
            # octomap_reset=planner.reset_octomap,
            # octomap_set=planner.set_planning_scene,
            # visualize=True,
        )
        t2 = time.time()
        grasp_time += t2 - t1
        print('Grasp Planning Time:', t2 - t1)

        # for g, s in zip(grasps, scores):
        #     invalid = grasp_planner.ik_collision(g, visualize=True)
        #     input(f'Score: {s}, Invalid: {invalid}, ...')

        ind = None
        while len(grasps) > 0:
            t1 = time.time()
            ## plan to target ##
            planner.reset_octomap()
            planner.set_planning_scene(
                points=all_pts,
                colors=all_rgb,
                # visualize=True,
            )
            # call bio_ik for goal joint_state
            ind = scores.index(max(scores))
            print('Chosen Grasp Score:', scores[ind])
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
                joint_state,
                goal_js,
                robot.gripper_group,
                is_diff=False,
            )
            t10 = time.time()
            print('Plan1 Time:', t10 - t1)
            if type(plan1) is not JointTrajectory:
                pre_grasps.pop(ind)
                grasps.pop(ind)
                scores.pop(ind)
                ind = None
                t2 = time.time()
                mplan_time += t2 - t1
                continue

            ## compute pre-grasp distance ##
            joint_state2 = JointState()
            joint_state2.name = plan1.joint_names
            joint_state2.position = plan1.points[-1].positions
            grasp_position = pose_to_matrix(grasps[ind])[:3, 3]
            pre_grasp_pose = pose_to_matrix(pre_grasps[ind])[:3, 3]
            dist = np.linalg.norm(grasp_position - pre_grasp_pose)
            # print('New Pre-grasp Distance:', dist)
            planner.reset_octomap()
            f2, plan2 = planner.cartesian_motion(
                joint_state2,
                (0, 0, dist),
                robot.gripper_group,
                robot.gripper_link,
                xyz_is_relative=True,
                avoid_collisions=True,
                is_diff=False,
            )
            t11 = time.time()
            print('Plan2 Time:', t11 - t10)
            if f2 < 1.0:
                print('Plan2 Error:', f2)
                # remove the current grasp pose at index ind
                pre_grasps.pop(ind)
                grasps.pop(ind)
                scores.pop(ind)
                ind = None
                t2 = time.time()
                mplan_time += t2 - t1
                continue

            ## plan lift ##
            joint_state3 = JointState()
            joint_state3.name = plan2.joint_names
            joint_state3.position = plan2.points[-1].positions
            grasp_state = joint_state3
            f3, plan3 = planner.cartesian_motion(
                joint_state3,
                (0, 0, lift_height),
                robot.gripper_group,
                robot.gripper_link,
                xyz_is_relative=False,
                avoid_collisions=True,
                is_diff=False,
            )
            t12 = time.time()
            print('Plan3 Time:', t12 - t11)
            if f3 <= 0.01:
                print('Plan3 Error:', f3)
                pre_grasps.pop(ind)
                grasps.pop(ind)
                scores.pop(ind)
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

                # filter all_pts by robot state
                all_pts_filtered = copy.deepcopy(all_pts)
                ind_js2ik = list(map(joint_state4.name.index, ik.joint_names))
                ind_js2ik2 = list(map(joint_state4.name.index, ik2.joint_names))
                ind_js2ik3 = list(map(joint_state4.name.index, ik3.joint_names))
                joint_values = np.array(joint_state4.position)[ind_js2ik]
                joint_values2 = np.array(joint_state4.position)[ind_js2ik2]
                joint_values3 = np.array(joint_state4.position)[ind_js2ik3]
                ee_pose = ik.fk(joint_values)
                arm7_pose = ik2.fk(joint_values2)
                arm6_pose = ik3.fk(joint_values3)
                positions = [ee_pose[:3, 3], arm7_pose[:3, 3], arm6_pose[:3, 3]]
                kdtree = KDTree(positions)
                dists, inds = kdtree.query(all_pts_filtered, k=1)
                all_pts_filtered = all_pts_filtered[dists > 0.09]
                print('Dists:', min(dists), max(dists), flush=True)

                plan4 = plan_place(
                    joint_state4,
                    place,
                    grasp_state,
                    tgt_pts,
                    # all_pts,
                    all_pts_filtered,
                    planner,
                    robot,
                    # tgt_rgb=tgt_rgb if visualize else None,
                    # all_rgb=all_rgb if visualize else None,
                )
                if type(plan4) is not JointTrajectory:
                    pre_grasps.pop(ind)
                    grasps.pop(ind)
                    scores.pop(ind)
                    ind = None
                    t2 = time.time()
                    mplan_time += t2 - t1
                    continue

                t2 = time.time()
                mplan_time += t2 - t1
                print('Plan4 Time:', t2 - t1)
                plans = [plan1, plan2, plan3, plan4]
            else:
                plans = [plan1, plan2, plan3]
            break

    t1 = time.time()
    print('Chosen Grasp Score:', scores[ind])
    print('Total Grasp Planning Time:', grasp_time)
    print('Total MotionPlanning Time:', mplan_time)
    print('Total Computation Time:', (t1 - t0) - (t02 - t01))

    input('execute?')

    t0 = time.time()
    for i, plan in enumerate(plans):
        if i == 0:
            ee_open()
        elif i == 2:
            ee_close()

        if type(plan) is JointTrajectory:
            # print('Plan success!')
            execute(plan, window=0.1, wait=True)
            # execute(plan, window=0)
        else:
            # print('Plan', i, 'error:', plan)
            break

    t1 = time.time()
    print('Total Execution Time:', t1 - t0)


def plan_place(
    joint_state,
    place,
    grasp_state,
    tgt_pts,
    all_pts,
    planner,
    robot,
    tgt_rgb=None,
    all_rgb=None,
):
    ## get estimated target mesh ##
    tgt_mesh = PerceptionInterface.get_shape_estimate(
        tgt_pts,
        tgt_rgb,
    )

    ## add target mesh to scene ##
    planner.reset()
    planner.set_planning_scene(
        points=all_pts,
        target_mesh=tgt_mesh,
        colors=all_rgb,
    )

    ## plan to place ##
    # place_mat = list_to_matrix(place)
    # ik_solver = planner.ik_for_ees[robot.gripper_link]
    # joint_vals = ik_solver.ik(place_mat)
    # place = dict(zip(ik_solver.joint_names, joint_vals))
    # plan = planner.joint_motion_plan(
    plan = planner.pose_motion_plan(
        joint_state,
        place,
        robot.gripper_group,
        attach_objects=['target'],
        grasp_state=grasp_state,
        ee=robot.gripper_link,
        ee_links=robot.ignore_collision_ee_links,
        is_diff=False,
    )
    return plan


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
    place = [0.6, 0, 1.1, 0.5, -0.5, -0.5, -0.5] if len(sys.argv) > 4 else None
    # place = [0.6, 0, 1.1] if len(sys.argv) > 4 else None

    print('is_sim: ', is_sim)
    print('gt: ', gt)
    ## init perception and planning interfaces ##
    rospy.init_node("planning")
    t0 = time.time()
    from motoman import MotomanSDA10F
    robot = MotomanSDA10F(is_sim, gt)
    perception = robot.init_perception_interface()
    planner = robot.init_motion_planner()
    grasp_planner = GraspPlanner()
    t1 = time.time()
    print('Init Time:', t1 - t0)

    open_loop_pick_or_place(
        object_to_grasp,
        robot,
        perception,
        planner,
        grasp_planner,
        target_augment_scale=0,
        target_augment_scale_dim=2,
        place=place,
        visualize=True,
    )

    input('Open?')
    ee_open()
