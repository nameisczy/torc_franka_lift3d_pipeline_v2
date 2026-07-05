import sys
import cv2
import time
import trimesh
import numpy as np
import open3d as o3d
import transformations as tf
from scipy.spatial import KDTree

import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from moveit_commander import conversions as conv

from utils import conversions as conv2
from grasp_planner.grasp_planner import GraspPlanner
from task_planner.eutils import ee_open, ee_close, execute
from perception.perception_fast import PerceptionInterface

pre_grasp_dist = 0.05
lift_height = 0.05
tolerance = 200


def update_and_get_points(
    object_to_grasp,
    robot,
    perception,
    filter_points=True,
    save_debug=False,
):
    t0 = time.time()
    print('0', flush=True)
    perception.updated_fused_points(
        robot.camera[0], object_to_grasp, filter_robot=False
    )
    # print('1', flush=True)
    perception.updated_fused_points(
        robot.camera[1], object_to_grasp, filter_robot=False
    )
    # print('2', flush=True)
    points, colors = perception.get_fused_point_cloud()
    # print('3', flush=True)
    bg_points, bg_colors = perception.get_fused_bg_point_cloud()
    # print('4', flush=True)

    t2 = time.time()
    if save_debug:
        perception.save_fusion('/tmp/pc_data', object_to_grasp)
    t3 = time.time()

    # print('5', flush=True)
    perception.update_occlusion(robot.camera[0])
    perception.update_occlusion(robot.camera[1])
    # print('6', flush=True)
    occluded_points = perception.get_occlusion_points()
    occluded_colors = np.zeros((*occluded_points.shape[:-1], 3))
    occluded_colors[:] = [1, 0, 1]
    t4 = time.time()
    # print('7', flush=True)

    trimesh.points.PointCloud(points, colors).show()
    trimesh.points.PointCloud(occluded_points, occluded_colors).show()

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
        tgt_pts, tgt_rgb = perception.get_largest_target_cluster()
    else:
        # print('f8', flush=True)
        tgt_pts, tgt_rgb = perception.get_fused_target_point_cloud()

    # down sample
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    resolution = 0.025
    downpcd = pcd.voxel_down_sample(resolution)
    ds_all_pts = np.array(downpcd.points)

    t1 = time.time()
    print('Fusion Time:', t2 - t0, flush=True)
    print('Occlusion Time:', t4 - t3, flush=True)
    print('Filtering Time:', t1 - t4, flush=True)
    print('Total Perception Time:', (t1 - t0) - (t3 - t2), flush=True)

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
    print('Staring!')
    t0 = time.time()

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

        occl_vol, ocolor_vol = perception.occl_vol.get_volume()
        tsdf_vol, color_vol, mask = perception.tsdf_vol.get_volume()
        # print(np.histogram(occl_vol.flatten(), bins=10))
        occl_en = trimesh.voxel.encoding.DenseEncoding(occl_vol >= 0)
        tsdf_en = trimesh.voxel.encoding.DenseEncoding(
            np.logical_and(tsdf_vol > -0.5, tsdf_vol < 0.5)
        )
        mask_en = trimesh.voxel.encoding.DenseEncoding(mask != 0)
        # trimesh.voxel.base.VoxelGrid(tsdf_en).show()
        # trimesh.voxel.base.VoxelGrid(occl_en).show()
        # trimesh.voxel.base.VoxelGrid(mask_en).show()
        # trimesh.voxel.base.VoxelGrid(mask_en).as_boxes(
        #     colors=color_vol[mask != 0]
        # ).show()
        # input('Continue?')

        points, colors = perception.get_fused_point_cloud()
        occluded_points = perception.get_occlusion_points()
        spc = o3d.geometry.PointCloud()
        spc.points = o3d.utility.Vector3dVector(points)
        spc.colors = o3d.utility.Vector3dVector(
            [1, 0, 0] * np.ones_like(points)
        )
        opc = o3d.geometry.PointCloud()
        opc.points = o3d.utility.Vector3dVector(occluded_points)
        opc.colors = o3d.utility.Vector3dVector(
            [0, 0, 1] * np.ones_like(occluded_points)
        )
        o3d.visualization.draw_geometries([spc, opc])
        min_bnds, max_bnds = perception.bounds.T
        svgd = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            spc, 0.005, min_bnds, max_bnds
        )
        ovgd = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            opc, 0.005, min_bnds, max_bnds
        )
        o3d.visualization.draw_geometries([svgd])
        o3d.visualization.draw_geometries([ovgd])

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(
            np.concatenate([points, occluded_points])
        )
        all_rb = np.concatenate(
            [
                [1, 0, 0] * np.ones_like(points),
                [0, 0, 1] * np.ones_like(occluded_points)
            ]
        )
        pc.colors = o3d.utility.Vector3dVector(all_rb)
        # o3d.visualization.draw_geometries([pc])
        # vgd = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        #     pc, 0.005, min_bnds, max_bnds
        # )
        # o3d.visualization.draw_geometries([vgd])
        tpc = o3d.t.geometry.PointCloud()
        tpc.point.points = o3d.core.Tensor(
            np.concatenate([points, occluded_points]), o3d.core.float32
        )
        tpc.point.colors = o3d.core.Tensor(all_rb, o3d.core.float32)

        x = 0.14
        y = 0
        z = 1.4
        qx = -0.541573133
        qy = 0.541573133
        qz = -0.454641112
        qw = 0.454641112

        x = 0.5540672918892274
        y = -0.7395636283945333
        z = 1.1163768752298142
        qx = -0.6960636074531736
        qy = 0.25641677746189684
        qz = -0.23943406181315297
        qw = 0.6264321357170187

        cam_intr = np.array(list(perception.cam_info.values())[0].K).reshape(
            (3, 3)
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        fx = cam_intr[0, 0]
        fy = cam_intr[1, 1]
        cx = cam_intr[0, 2]
        cy = cam_intr[1, 2]
        intrinsic.set_intrinsics(640, 480, fx, fy, cx, cy)

        R = tf.quaternion_matrix([qw, qx, qy, qz])[:3, :3]
        t = np.array([y, z, -x])
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R.T
        extrinsic[:3, 3] = np.array([y, z, x])

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        ctr = vis.get_view_control()
        vis.add_geometry(pc)
        vis.update_geometry(pc)
        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.intrinsic = intrinsic
        camera_params.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(camera_params, True)
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer()
        o3d.visualization.draw_geometries([img])
        img = np.array(img)
        print(img[:, :, 0], img[:, :, 2])
        vis.destroy_window()
        cv2.imshow('Red', img[:, :, 0])
        cv2.imshow('Blue', img[:, :, 2])
        cv2.waitKey()
        cv2.destroyAllWindows()

        input('Continue?')
        pcd_all = trimesh.points.PointCloud(all_pts, all_rgb)
        pcd = trimesh.points.PointCloud(tgt_pts, tgt_rgb)
        pcd.show()
        tgt_mesh = perception.get_shape_estimate(
            tgt_pts,
            # tgt_rgb,
            scale=1,
        )
        fp = tgt_mesh.triangles_center
        ft = tgt_mesh.face_normals

        thresh0 = np.percentile(tgt_pts[:, 2], 5)
        print(thresh0)
        ft = ft[fp[:, 2] > thresh0]
        fp = fp[fp[:, 2] > thresh0]
        # nv = list(zip(fp.tolist(), (ft + fp).tolist()))
        # norms = trimesh.load_path(nv)
        # trimesh.Scene([pcd, tgt_mesh, norms]).show()

        dd, i = KDTree(tgt_pts).query(fp)
        thresh = np.percentile(dd, 95)
        print(np.histogram(dd, bins=10))
        print(thresh)

        fp = fp[dd > thresh]
        ft = ft[dd > thresh]
        nv = list(zip(fp.tolist(), (ft + fp).tolist()))
        norms = trimesh.load_path(nv)
        trimesh.Scene([pcd, tgt_mesh, norms]).show()

        crosses = []
        ncrosses = []
        max_cross = None
        max_d = 0
        for i in range(len(fp)):
            p = fp[i]
            n = ft[i]
            c = np.cross(n, [0, 0, 1])
            d = -c
            c[2] = np.linalg.norm(c[:-1])
            d[2] = np.linalg.norm(c[:-1])
            c = 0.25 * c / np.linalg.norm(c)
            d = 0.25 * d / np.linalg.norm(d)
            crosses.append((p, p + c))
            ncrosses.append((p, p + d))
            if dd[i] > max_d:
                max_d = dd[i]
                max_cross = trimesh.load_path([(p, p + c), (p, p + d)])
        cross = trimesh.load_path(crosses)
        ncross = trimesh.load_path(ncrosses)
        trimesh.Scene([pcd, tgt_mesh, cross, ncross]).show()
        trimesh.Scene([pcd, tgt_mesh, max_cross]).show()

        avg_norm = np.sum(ft, axis=0)
        avg_norm = avg_norm / np.linalg.norm(avg_norm)
        avg_norm[2] = 0
        anorm = trimesh.load_path([pcd.centroid, pcd.centroid + avg_norm])
        trimesh.Scene([pcd, anorm]).show()
        input('Continue?')

        a_occluded_points = all_pts[(all_rgb[:, 0] == 1) & (all_rgb[:, 2] == 1)]
        nums = KDTree(tgt_pts).query_ball_point(
            a_occluded_points, 0.005, return_length=True
        )
        print(np.histogram(nums, bins=range(max(nums) + 1)))
        occluded_points = a_occluded_points[np.where(nums == 1)]
        occ_pcd = trimesh.points.PointCloud(occluded_points)

        opc = o3d.geometry.PointCloud()
        opc.points = o3d.utility.Vector3dVector(occluded_points)
        opc.estimate_normals()
        normals = np.asarray(opc.normals)
        avg_norm = np.sum(normals, axis=0)
        avg_norm = avg_norm / np.linalg.norm(avg_norm)
        nv = list(
            zip(
                occluded_points.tolist(),
                (normals + occluded_points).tolist(),
            )
        )
        norms = trimesh.load_path(nv)
        anorm = trimesh.load_path(
            [occ_pcd.centroid, occ_pcd.centroid + avg_norm]
        )

        trimesh.Scene([pcd, norms, occ_pcd]).show()
        trimesh.Scene([pcd, anorm, occ_pcd]).show()
        # trimesh.points.PointCloud(tgt_pts, tgt_rgb).show()

        opc = o3d.geometry.PointCloud()
        opc.points = o3d.utility.Vector3dVector(tgt_pts)
        opc.estimate_normals()
        normals = np.asarray(opc.normals)
        avg_norm = np.sum(normals, axis=0)
        avg_norm = avg_norm / np.linalg.norm(avg_norm)
        nv = list(zip(
            tgt_pts.tolist(),
            (tgt_pts - normals).tolist(),
        ))
        norms = trimesh.load_path(nv)
        anorm = trimesh.load_path([pcd.centroid, pcd.centroid - avg_norm])
        trimesh.Scene([pcd, norms]).show()
        trimesh.Scene([pcd, anorm]).show()
        input('Continue?')
    t02 = time.time()

    ## generate grasps ##
    t1 = time.time()
    grasps, pre_grasps, scores, configs = grasp_planner.get_grasp_poses(
        tgt_pts,
        tgt_rgb,
        planner.ik_for_ees[robot.gripper_link].ik,
        # pre_grasp_dist=pre_grasp_dist,
        # collision_voxel_tuple=(points, all_pts, 0.005),
        collision_voxel_tuple=(tgt_pts, all_pts, 0.01),
        # input_frame=f'{robot.camera[0]}_color_optical_frame',
        # visualize=True,
        # filter_outliers=(20, 0.1),
    )
    t2 = time.time()
    print('Grasp Planning Time:', t2 - t1)

    ## rescore grasps ##
    t1 = time.time()
    grasps2, pre_grasps2, scores2, configs2 = grasp_planner.rescore_grasps(
        configs,
        tgt_pts,
        tgt_rgb,
        planner.ik_for_ees[robot.gripper_link].ik,
        # pre_grasp_dist=pre_grasp_dist,
        # collision_voxel_tuple=(points, all_pts, 0.005),
        collision_voxel_tuple=(tgt_pts, all_pts, 0.01),
        # input_frame=f'{robot.camera[0]}_color_optical_frame',
        # visualize=True,
        # filter_outliers=(20, 0.1),
    )
    t2 = time.time()
    print('ReScore Time:', t2 - t1)

    ## rescore grasps ##
    t1 = time.time()
    grasps3, pre_grasps3, scores3, configs3 = grasp_planner.rescore_grasps(
        configs,
        tgt_pts,
        tgt_rgb,
        planner.ik_for_ees[robot.gripper_link].ik,
        # pre_grasp_dist=pre_grasp_dist,
        # collision_voxel_tuple=(points, all_pts, 0.005),
        collision_voxel_tuple=(tgt_pts, all_pts, 0.01),
        # input_frame=f'{robot.camera[0]}_color_optical_frame',
        # visualize=True,
        # filter_outliers=(20, 0.1),
    )
    t2 = time.time()
    print('ReScore Time:', t2 - t1)

    all_equal = True
    for config in configs:
        for key in config.__slots__:
            print(key, v1 == v2, v1 == v3, v2 == v3)
            if key == 'score':
                continue
            v1 = getattr(configs[0], key)
            v2 = getattr(configs2[0], key)
            v3 = getattr(configs3[0], key)
            all_equal &= v1 == v2 and v1 == v3 and v2 == v3
    print('All Equal:', all_equal)
    ranks = list(sorted(range(len(scores)), key=lambda x: scores[x]))
    ss1 = np.array(scores)[ranks]
    ss2 = np.array(scores2)[ranks]
    ss3 = np.array(scores3)[ranks]
    print('Scores:', list(sorted(range(len(ss1)), key=lambda x: ss1[x])))
    print('Scores2:', list(sorted(range(len(ss2)), key=lambda x: ss2[x])))
    print('Scores3:', list(sorted(range(len(ss3)), key=lambda x: ss3[x])))
    t03 = time.time()
    input('Continue?')
    t04 = time.time()

    t3 = time.time()
    ## plan to target ##
    planner.reset(update_moveit=False)
    planner.set_planning_scene(
        points=ds_all_pts,
        # target_mesh=tgt_mesh,
        update_moveit=False,
        visualize=True,
    )
    t4 = time.time()

    joint_state = rospy.wait_for_message(
        '/joint_states_all', JointState, timeout=5
    )

    plan1 = planner.iter_ik_motion_plan(
        joint_state,
        # pre_grasps,
        pre_grasps[np.argmax(scores)],
        robot.gripper_group,
        # score=scores,
        score=10,
        num_iters=50,
        speed=20 * np.pi / 180,
        look_link=robot.camera_link,
        lookat_point=np.mean(tgt_pts, axis=0),
        lookat_axis=robot.camera_axis,
        ee=robot.gripper_link,
        is_diff=False,
    )

    t10 = time.time()
    print('Plan1', plan1)
    print('Plan1 Points:', len(plan1.points))
    print('Plan1 Time:', t10 - t3)

    input('Continue?')
    execute(plan1, wait=True, retime=False)
    return

    ## compute pre-grasp distance ##
    joint_state2 = JointState()
    joint_state2.name = plan1.joint_names
    joint_state2.position = plan1.points[-1].positions

    ik = planner.ik_for_ees[robot.gripper_link]
    joint_indices = list(map(joint_state2.name.index, ik.joint_names))
    joint_values = np.array(joint_state2.position)[joint_indices]
    pre_pose = ik.fk(joint_values)
    pre_pose[:3, 3]

    min_dist = np.inf
    ind = None
    for i, g in enumerate(pre_grasps):
        g_pose = conv2.pose_to_matrix(g)
        g_position = [g.position.x, g.position.y, g.position.z]
        distT = np.linalg.norm(g_position - pre_pose[:3, 3])
        distR = np.linalg.norm(g_pose[:3, :3] - pre_pose[:3, :3])
        if distT + distR < min_dist:
            min_dist = distT + distR
            ind = i

    grasp_position = conv2.pose_to_matrix(grasps[ind])[:3, 3]
    dist = np.linalg.norm(grasp_position - pre_pose[:3, 3])
    print('New Pre-grasp Distance:', dist)

    f2, plan2 = planner.cartesian_motion(
        joint_state2,
        (0, 0, dist),
        robot.gripper_group,
        robot.gripper_link,
        xyz_is_relative=True,
        avoid_collisions=False,
        is_diff=False,
    )

    t11 = time.time()
    print('Plan2 Time:', t11 - t10)

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
        avoid_collisions=False,
        is_diff=False,
    )

    t12 = time.time()
    print('Plan3 Time:', t12 - t11)

    plans = [plan1, plan2, plan3]

    if place:
        ## plan to place ##
        joint_state4 = JointState()
        joint_state4.name = plan3.joint_names
        joint_state4.position = plan3.points[-1].positions
        plan4 = plan_place(
            joint_state4,
            place,
            grasp_state,
            tgt_pts,
            all_pts,
            planner,
            robot,
            # tgt_rgb=tgt_rgb if visualize else None,
            # all_rgb=all_rgb if visualize else None,
        )
        plans.append(plan4)

        t13 = time.time()
        print('Plan4 Time:', t13 - t12)

    t1 = time.time()
    print('Planning Scene Building Time:', (t4 - t3) * 2)
    print('Total Motion Planning Time:', t1 - t3)
    print('Total Computation Time:', (t1 - t0) - (t02 - t01) - (t04 - t03))

    input('execute?')

    t0 = time.time()
    for i, plan in enumerate(plans):
        if i == 0:
            ee_open()
        elif i == 2:
            ee_close()

        if type(plan) is JointTrajectory:
            print('Plan success!')
            execute(plan, wait=True)
        else:
            print('Plan', i, 'error:', plan)
            break

    t1 = time.time()
    print('Total Execution Time:', t1 - t0)

    input('Open?')
    ee_open()


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
        frame_id='world',  #f'{robot.camera[0]}_color_optical_frame',
        colors=all_rgb,
        # filter_outliers=(20, 2),
    )

    ## plan to place ##
    # place_mat = conv2.pose_to_matrix(conv.list_to_pose(place))
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
    # place = [0.6, 0, 1.1, 0.5, -0.5, -0.5, -0.5] if len(sys.argv) > 4 else None
    place = [0.6, 0, 1.1] if len(sys.argv) > 4 else None

    ## init perception and planning interfaces ##
    rospy.init_node("planning")
    t0 = time.time()
    from task_planner.motoman import MotomanSDA10F
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
        visualize=False,
    )
