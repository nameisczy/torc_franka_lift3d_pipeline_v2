import copy
import numpy as np
import open3d as o3d
import trimesh as tm
from scipy.spatial import KDTree

import rospy
from moveit_commander import conversions as conv

import bio_ik_msgs.msg as bik
from bio_ik_msgs.srv import GetIK
from geometry_msgs.msg import Pose, Point
from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from motion_planner.moveit_planner import MoveitPlanner
from perception.perception_fast import PerceptionInterface


class BioIkPlanner(MoveitPlanner):

    def __init__(
        self,
        robot_file,
        end_effector_links,
        move_groups,
        commander_args=[],
        is_sim=True,
        collision_links=[],
    ):
        super().__init__(
            robot_file,
            end_effector_links,
            move_groups,
            commander_args,
            is_sim,
        )
        self.collision_links = collision_links
        self.pose_weight = 20.0
        self.r_scale = 1.0
        self.look_weight = 20.0
        self.cone_weight = 40.0
        self.c_w = 500.0
        self.d = 0.13#0.085
        self.o_c_w = 500.0
        self.o_d = 0.15#0.085
        ## Init IK request ##
        self.ik_req = bik.IKRequest()
        self.ik_req.timeout = rospy.Duration(0.1)
        self.ik_req.avoid_collisions = False
        self.ik_req.approximate = True
        # Try to ensure joint limits
        self.ik_req.avoid_joint_limits_goals.append(bik.AvoidJointLimitsGoal())
        self.ik_req.avoid_joint_limits_goals[-1].weight = 1.0
        self.ik_req.avoid_joint_limits_goals[-1].primary = True
        # Try to ensure proximity to previous iteration
        self.ik_req.minimal_displacement_goals.append(
            bik.MinimalDisplacementGoal()
        )
        self.ik_req.minimal_displacement_goals[-1].weight = 10.0
        self.ik_req.minimal_displacement_goals[-1].primary = True
        self.reset(reset_static=True, update_moveit=False)

    def reset(self, reset_static=False, update_moveit=True):
        if update_moveit:
            super().reset()
        else:
            if reset_static:
                self.points = []
                self.static_points = []
                self.ik_req.min_distance_goals.clear()
                density = 50
                obj_d = self.scene.get_objects()
                pose_d = self.scene.get_object_poses(list(obj_d.keys()))
                for name, obj in obj_d.items():
                    #TODO process objects other than boxes
                    extents = obj.primitives[0].dimensions
                    pos_msg = pose_d[name].position
                    translation = (pos_msg.x, pos_msg.y, pos_msg.z)
                    box = tm.creation.box(extents)
                    box.apply_translation(translation)
                    num_samples = round(density * box.area)
                    samples = tm.sample.sample_surface_even(box, num_samples)[0]
                    self.static_points.extend(samples)
                    # Try to avoid collisions for collision_links
                    for link in self.collision_links:
                        for x, y, z in self.static_points:
                            self.ik_req.min_distance_goals.append(
                                bik.MinDistanceGoal()
                            )
                            self.ik_req.min_distance_goals[-1].link_name = link
                            self.ik_req.min_distance_goals[-1].target.x = x
                            self.ik_req.min_distance_goals[-1].target.y = y
                            self.ik_req.min_distance_goals[-1].target.z = z
                            self.ik_req.min_distance_goals[-1].distance = self.d
                            self.ik_req.min_distance_goals[-1].weight = self.c_w
            else:
                for i in range(len(self.points)):
                    self.ik_req.min_distance_goals.pop()

            # print(len(self.static_points))
            # tm.points.PointCloud(self.static_points).show()

    def set_planning_scene(
        self,
        points,
        target_mesh=None,
        frame_id='world',
        colors=None,
        filter_outliers=False,
        down_sample=False,
        update_moveit=True,
        visualize=False,
    ):
        if update_moveit:
            super().set_planning_scene(
                points, target_mesh, frame_id, colors, filter_outliers
            )
        else:
            if down_sample:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                resolution = 0.1  #rospy.get_param('/move_group/octomap_resolution', 0.01)
                downpcd = pcd.voxel_down_sample(resolution)
                self.points = list(downpcd.points)
            else:
                self.points = points
            # Try to avoid collisions for collision_links
            if visualize:
                tm.points.PointCloud(
                    list(self.static_points) + list(self.points)
                ).show()
            for link in self.collision_links:
                for x, y, z in self.points:
                    self.ik_req.min_distance_goals.append(bik.MinDistanceGoal())
                    self.ik_req.min_distance_goals[-1].link_name = link
                    self.ik_req.min_distance_goals[-1].target.x = x
                    self.ik_req.min_distance_goals[-1].target.y = y
                    self.ik_req.min_distance_goals[-1].target.z = z
                    self.ik_req.min_distance_goals[-1].distance = self.o_d
                    self.ik_req.min_distance_goals[-1].weight = self.o_c_w

    def make_exploration_goal(self, tgt_pts, tgt_rgb=None, look_dir=1):
        # estimate target shape
        tgt_mesh = PerceptionInterface.get_shape_estimate(
            tgt_pts,
            tgt_rgb,
            scale=1,
        )

        # get face centers and normals
        fp = tgt_mesh.triangles_center
        fn = tgt_mesh.face_normals

        # remove points below 5th percentile
        bottom = np.percentile(tgt_pts[:, 2], 5)
        fn = fn[fp[:, 2] > bottom]
        fp = fp[fp[:, 2] > bottom]

        # select points furthest from the visible surface
        d, i = KDTree(tgt_pts).query(fp)
        thresh = np.percentile(d, 95)
        fp = fp[d > thresh]
        fn = fn[d > thresh]

        # average normals and find average furthest point
        # avg_fn = np.sum(fn, axis=0)
        # avg_fn = avg_fn / np.linalg.norm(avg_fn)
        # avg_fn[2] = 1.5
        # avg_fn = -avg_fn / np.linalg.norm(avg_fn)
        # avg_fp = np.mean(fp, axis=0)

        for i in range(len(fn)):
            fn[i] = np.cross(fn[i], (0, 0, look_dir))
            fn[i][2] = np.linalg.norm(fn[i][:-1])
            fn[i] = -fn[i] / np.linalg.norm(fn[i])

        # visualize if desired
        if tgt_rgb is not None:
            nv = list(zip(fp.tolist(), (fn + fp).tolist()))
            norms = tm.load_path(nv)
            pcd = tm.points.PointCloud(tgt_pts, tgt_rgb)
            tm.Scene([pcd, tgt_mesh, norms]).show()
            anorm = tm.load_path([avg_fp, avg_fp - avg_fn])
            tm.Scene([pcd, anorm]).show()

        return fp, fn, d[d > thresh]
        # return avg_fp, avg_fn, max(d)

    def set_collision_params(self, collision_weight, collision_distance):
        self.c_w = collision_weight
        self.o_c_w = collision_weight
        self.d = collision_distance
        self.o_d = collision_distance

    def iter_ik_motion_plan(
        self,
        start_jnt,
        goal,  # pose as list, pose msg, or list of either
        move_group,
        score,  # one per goal
        num_iters=10,
        speed=20,
        min_duration=0.2,
        look_link=None,
        lookat_point=None,  # point (x,y,z)
        lookat_axis=(0, 0, 1),  # axis (x,y,z)
        cone_direction=None, # vector (x,y,z),
        cone_angle=np.pi,
        check_collisions=False,
        attach_objects=[],
        grasp_state=None,
        ee=None,
        ee_links=[],
        is_diff=False
    ):
        r_start_state = self.make_robot_state_msg(
            start_jnt, attach_objects, grasp_state, ee, ee_links, is_diff
        )
        trac_ik = self.ik_for_ees[ee]

        self.ik_req.group_name = f'{move_group}_bioik'
        self.ik_req.robot_state = r_start_state
        self.ik_req.avoid_collisions = check_collisions

        # Prevent end-effector from moving too far from
        self.ik_req.max_distance_goals.clear()
        self.ik_req.max_distance_goals.append(bik.MaxDistanceGoal())
        self.ik_req.max_distance_goals[-1].link_name = ee
        self.ik_req.max_distance_goals[-1].distance = 0.05
        self.ik_req.max_distance_goals[-1].weight = 100.0

        # Try to look at lookat_point
        self.ik_req.look_at_goals.clear()
        self.ik_req.cone_goals.clear()
        if look_link is not None and lookat_point is not None:
            self.ik_req.look_at_goals.append(bik.LookAtGoal())
            self.ik_req.look_at_goals[-1].link_name = look_link
            self.ik_req.look_at_goals[-1].target.x = lookat_point[0]
            self.ik_req.look_at_goals[-1].target.y = lookat_point[1]
            self.ik_req.look_at_goals[-1].target.z = lookat_point[2]
            self.ik_req.look_at_goals[-1].axis.x = lookat_axis[0]
            self.ik_req.look_at_goals[-1].axis.y = lookat_axis[1]
            self.ik_req.look_at_goals[-1].axis.z = lookat_axis[2]
            self.ik_req.look_at_goals[-1].weight = self.look_weight

            if cone_direction is not None:
                self.ik_req.cone_goals.append(bik.ConeGoal())
                self.ik_req.cone_goals[-1].link_name = look_link
                self.ik_req.cone_goals[-1].axis.x = lookat_axis[0]
                self.ik_req.cone_goals[-1].axis.y = lookat_axis[1]
                self.ik_req.cone_goals[-1].axis.z = lookat_axis[2]
                cone_point = lookat_point - 0.35 * np.array(cone_direction)
                self.ik_req.cone_goals[-1].position.x = cone_point[0]
                self.ik_req.cone_goals[-1].position.y = cone_point[1]
                self.ik_req.cone_goals[-1].position.z = cone_point[2]
                self.ik_req.cone_goals[-1].angle = cone_angle
                self.ik_req.cone_goals[-1].weight = self.cone_weight

        # Set pose or position goal(s)
        self.ik_req.pose_goals.clear()
        self.ik_req.position_goals.clear()
        # print(goal, hasattr(goal, '__iter__'))
        if goal is not None:
            if hasattr(goal, '__iter__') and len(goal) > 0:
                if not hasattr(goal[0], 'as_integer_ratio'):
                    for i, g in enumerate(goal):
                        if type(g) is list:
                            self.ik_req.pose_goals.append(bik.PoseGoal())
                            self.ik_req.pose_goals[-1].link_name = ee
                            self.ik_req.pose_goals[-1].pose = conv.list_to_pose(
                                g
                            )
                            self.ik_req.pose_goals[-1].weight = self.pose_weight
                            self.ik_req.pose_goals[-1].weight *= score[i]
                            self.ik_req.pose_goals[
                                -1].rotation_scale = self.r_scale
                        elif type(g) is Pose:
                            self.ik_req.pose_goals.append(bik.PoseGoal())
                            self.ik_req.pose_goals[-1].link_name = ee
                            self.ik_req.pose_goals[-1].pose = g
                            self.ik_req.pose_goals[-1].weight = self.pose_weight
                            self.ik_req.pose_goals[-1].weight *= score[i]
                            self.ik_req.pose_goals[
                                -1].rotation_scale = self.r_scale
                        else:
                            print('Invalid goal item type:', g, type(g))
                            return None
                elif len(goal) == 3:
                    self.ik_req.position_goals.append(bik.PositionGoal())
                    self.ik_req.position_goals[-1].link_name = ee
                    self.ik_req.position_goals[-1].position.x = goal[0]
                    self.ik_req.position_goals[-1].position.y = goal[1]
                    self.ik_req.position_goals[-1].position.z = goal[2]
                    self.ik_req.position_goals[-1].weight = self.pose_weight
                    self.ik_req.position_goals[-1].weight *= score
                elif len(goal) in (6, 7):
                    self.ik_req.pose_goals.append(bik.PoseGoal())
                    self.ik_req.pose_goals[-1].link_name = ee
                    self.ik_req.pose_goals[-1].pose = conv.list_to_pose(goal)
                    self.ik_req.pose_goals[-1].weight = self.pose_weight
                    self.ik_req.pose_goals[-1].weight *= score
                    self.ik_req.pose_goals[-1].rotation_scale = self.r_scale
                else:
                    print('Invalid goal:', goal)
                    return None
            elif type(goal) is Pose:
                self.ik_req.pose_goals.append(bik.PoseGoal())
                self.ik_req.pose_goals[-1].link_name = ee
                self.ik_req.pose_goals[-1].pose = goal
                self.ik_req.pose_goals[-1].weight = self.pose_weight
                self.ik_req.pose_goals[-1].weight *= score
                self.ik_req.pose_goals[-1].rotation_scale = self.r_scale
            else:
                print('Invalid goal type:', type(goal))
                return None

        # Initialize joint trajectory
        traj = JointTrajectory()
        traj.joint_names = start_jnt.name
        traj.points.append(JointTrajectoryPoint())
        traj.points[-1].positions = start_jnt.position
        traj.points[-1].velocities = (0, ) * len(start_jnt.position)
        traj.points[-1].time_from_start = rospy.Duration(0)

        # Run num_iters iterations of IK and add points to joint trajectory
        request = self.ik_req
        total_dur = 0
        for i in range(num_iters):
            # request = copy.deepcopy(ik_req)
            if i > 0:
                request.robot_state = ik_sol
            # set max distance goal target
            jnt_state = request.robot_state.joint_state
            jnt_indices = list(map(jnt_state.name.index, trac_ik.joint_names))
            jnt_values = np.array(jnt_state.position)[jnt_indices]
            ee_pose = trac_ik.fk(jnt_values)
            # move away from the current joint angle
            self.ik_req.max_distance_goals[-1].target.x = ee_pose[0, 3]
            self.ik_req.max_distance_goals[-1].target.y = ee_pose[1, 3]
            self.ik_req.max_distance_goals[-1].target.z = ee_pose[2, 3]

            rospy.wait_for_service("/bio_ik/get_bio_ik")
            get_bio_ik = rospy.ServiceProxy("/bio_ik/get_bio_ik", GetIK)
            response = get_bio_ik(request).ik_response
            ik_sol = response.solution
            ik_state = ik_sol.joint_state
            # print(
            #     'Stats:',
            #     response.error_code,
            #     response.solution_fitness,
            #     flush=True
            # )
            if response.error_code.val != 1:
                print('IK Failed:', response.error_code.val, flush=True)
                if num_iters == 1:
                    return None
                continue
            traj.points.append(JointTrajectoryPoint())
            inds = list(map(ik_state.name.index, start_jnt.name))
            positions = np.array(ik_state.position)[inds]
            traj.points[-1].positions = positions
            disp = traj.points[-1].positions - traj.points[-2].positions
            # dist = np.linalg.norm(disp)
            dist = max(np.abs(disp))
            duration = dist / speed
            duration = max(duration, min_duration)
            total_dur += duration
            traj.points[-1].time_from_start = rospy.Duration(total_dur)
            traj.points[-1].velocities = tuple(disp / duration)
            # print('Speed, disp, dist:', speed, disp, dist, flush=True)

        # print(
        #     'Planned Trajectory Length:',
        #     traj.points[-1].time_from_start
        # )

        # self.fill_joint_trajectory_from_state(traj, start_jnt)
        display = DisplayTrajectory()
        display.trajectory_start.joint_state.name = traj.joint_names
        display.trajectory_start.joint_state.position = traj.points[0].positions
        display.trajectory.append(RobotTrajectory())
        display.trajectory[0].joint_trajectory = traj
        display_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            DisplayTrajectory,
            latch=True,
            queue_size=10
        )
        display_publisher.publish(display)
        return traj

    # def cartesian_motion(self, start, goal, ee):
    #     pass
