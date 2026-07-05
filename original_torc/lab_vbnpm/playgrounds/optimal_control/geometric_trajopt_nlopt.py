"""
trajector optimization on joint position waypoints.
objective:
- go to desired pose
- given a list of desired poses, go to one of them.
- track trajectory
constraints:
- collision avoidance
- position distance
collision avoidance:
- mesh vs point cloud (mujoco + hpp-fcl + open3d)
"""

from robot import Robot#MotomanRobot
from planning_scene import PlanningScene
import so3
from typing import Union
import numpy as np
import numpy.typing as npt
import scipy as sp
import time
from scipy.sparse import coo_array

"""
go to desired pose while avoiding collisions with the environment.
The environment is represented by a point cloud.
"""
class PoseTrajOpt():
    def __init__(self, robot: Robot, scene: PlanningScene,
                 start_q: Union[npt.NDArray[np.float32], npt.NDArray[np.float64]],
                 target_link: str,
                 target_pose: Union[npt.NDArray[np.float32], npt.NDArray[np.float64]],
                 n_waypoints: int,
                 safety_margin: float = 0.01, max_dist: float = 5*np.pi/180):
        """
        data: q0 (assumed to satisfy the constraints)
        variable: [q1, ..., qN], size: N*DOF
        objective: sum_i ||q_{i+1}-q_i||_2 + dist(q_N, target_pose)  (minimize)
            dist_p(q_N, target_pose) = ||x_N - target_position||_2
            dist_r(q_N, target_pose) = (log(R_target * R_N^T))_v
        constraints:
        - collision avoidance: g(q_i) = col_dist(q_i) >= safety_margin
        - position distance: ||q_{i+1}-g_i|| <= max_dist
        - joint limits: lb <= q_i <= ub
        """
        self.start_q = start_q
        # * set the lower bound and upper bound of the variables according to the joint limits
        lb = np.zeros((n_waypoints, len(robot.selected_joint_values)))
        lb += robot.selected_joint_limits[:, 0].reshape((1,-1))
        ub = np.zeros((n_waypoints, len(robot.selected_joint_values)))
        ub += robot.selected_joint_limits[:, 1].reshape((1,-1))
        lb = lb.flatten()
        ub = ub.flatten()
        self.lb = lb
        self.ub = ub

        # * set the lower bound and upper bound of the constraints
        # set the collision constraints
        self.safety_margin = safety_margin
        constr_safety_margin = np.zeros((n_waypoints, scene.col_pair_num)) + safety_margin
        constr_safety_margin = constr_safety_margin.flatten()
        self.constr_safety_margin = constr_safety_margin
        # set the position distance constraints
        self.max_dist = max_dist
        constr_max_dist = np.zeros((n_waypoints)) + max_dist
        constr_max_dist = constr_max_dist.flatten()
        self.constr_max_dist = constr_max_dist
        # set the joint limit constraints
        # shape: n_waypoints * ndof
        cl_joint_limits = np.zeros((n_waypoints, len(robot.selected_joint_values)))
        cu_joint_limits = np.zeros((n_waypoints, len(robot.selected_joint_values)))
        for i in range(n_waypoints):
            cl_joint_limits[i, :] = robot.selected_joint_limits[:, 0]
            cu_joint_limits[i, :] = robot.selected_joint_limits[:, 1]
        self.cl_joint_limits = cl_joint_limits
        self.cu_joint_limits = cu_joint_limits
        # cl += cl_joint_limits.flatten().tolist()
        # cu += cu_joint_limits.flatten().tolist()
        # NOTE: joint limit constraint is already specified by the lb and ub
        # adding terminal pose distance as a constriant

        self.robot = robot
        self.scene = scene
        self.target_link = target_link
        self.target_pose = target_pose
        self.ndof = len(robot.selected_joint_values)
        self.n_waypoints = n_waypoints
        self.n = n_waypoints * self.ndof

        self.waypoint_distance_objective_weight = 1.0
        self.terminal_pose_objective_weight = 10.0

    def set_start_q(self, start_q):
        self.start_q = start_q

    def waypoint_distance_objective(self, qs):
        """
        return the sum of waypoint distance: ||q_1-q_0||_2 + sum_i ||q_{i+1}-q_i||_2
        """
        # TODO: find more efficient ways to do this
        qs = np.array(qs).reshape(self.n_waypoints, self.ndof)
        # ||q1-q0||_2
        dist_0 = np.linalg.norm(qs[0] - self.start_q)
        diff = np.diff(qs, n=1, axis=0)
        return dist_0 + np.linalg.norm(diff, ord=2, axis=1).sum()

    def waypoint_distance_gradient(self, qs):
        """
        return the gradient of the sum of waypoint distance.
        dJ/dq1 = (q1-q0)/||q1-q0||_2 - (q2-q1)/||q2-q1||_2
        dJ/dqi = -(q_{i+1}-q_i)/||q_{i+1}-q_i||_2 + (q_i-q_{i-1})/||q_i-q_{i-1}||_2
        dJ/dqN = (qN-q_{N-1})/||qN-q_{N-1}||_2
        """
        qs = np.array(qs).reshape(self.n_waypoints, self.ndof)
        diff_0 = qs[0] - self.start_q
        diff = np.diff(qs, n=1, axis=0) # (n_waypoints-1) x ndof
        # gradient shape: n_waypoints x ndof
        gradient = np.zeros((self.n_waypoints, self.ndof))
        gradient[0,:] = diff_0 / np.linalg.norm(diff_0) - diff[0] / np.linalg.norm(diff[0])
        # gradient[0, :] = diff[0] / np.linalg.norm(diff[0]) * (-1)
        gradient[-1, :] = diff[-1] / np.linalg.norm(diff[-1])
        # NOTE: notice that we need to append a new axis
        gradient[1:-1, :] = (diff[:-1] / np.linalg.norm(diff[:-1], ord=2, axis=1)[:, np.newaxis] - 
                            diff[1:] / np.linalg.norm(diff[1:], ord=2, axis=1)[:, np.newaxis])
        return gradient.flatten()

    def waypoint_distance_trajopt_f(self, qs, grad):
        """
        set the objective function according to nlopt interface.
        """
        val = self.waypoint_distance_objective(qs)
        grad[:] = self.waypoint_distance_gradient(qs)
        return val

    def terminal_pose_objective(self, qs):
        """
        return the distance between the last waypoint and the target pose.
        """
        qs = np.array(qs).reshape(self.n_waypoints, self.ndof)
        robot = self.robot
        target_link = self.target_link
        target_pose = self.target_pose
        robot.set_selected_joint_values(qs[-1])
        pose = robot.get_link_pose(target_link)
        # get position difference
        position_diff = pose[:3,3] - target_pose[:3,3]
        # get orientation difference
        R = pose[:3, :3]
        rot_diff = so3.log_quaternion(np.dot(target_pose[:3, :3], R.T))
        # TODO: add weight for the position and rotation diff
        return np.linalg.norm(position_diff) + np.linalg.norm(rot_diff)

    def terminal_pose_gradient(self, qs):
        """
        return the gradient of the terminal pose objective.
        d J_p / d q = d ||x_p(q_N) - target_position|| / d q = 
                        (x_p(q_N) - target_position)/||x_p(q_N) - target_position||*J_p(q_N)
        d J_r / d q = d ||log(R_target * R_N^T)_V|| / d q = 
                        (log(R_target * R_N^T)_V)/||log(R_target * R_N^T)_V||*
                        d log(R_target * R_N^T)_V / d (R_target*R_N^T) * d (R_target*R_N^T) / d R_N * d R_N / d q
        """
        qs = np.array(qs).reshape(self.n_waypoints, self.ndof)
        robot = self.robot
        target_link = self.target_link
        target_pose = self.target_pose
        robot.set_selected_joint_values(qs[-1])
        pose = robot.get_link_pose(target_link)
        # get position difference
        position_diff = pose[:3,3] - target_pose[:3,3]
        # get the Jacobian of pose relative to joint angle
        link_jac = robot.get_link_analytical_jacobian(target_link)  # 4x4xnv
        # * get the gradient of the position difference. Shape: nv
        # TODO: currently this is wrong since when diff is zero, the derivative should be undfined.
        if np.linalg.norm(position_diff) == 0.0:
            axis = np.zeros((3))
        else:
            axis = position_diff / np.linalg.norm(position_diff)
        # axis = position_diff # NOTE: for the case of square of error
        position_diff_gradient = np.tensordot(axis, link_jac[:3, 3, :], axes=1)

        # * get the gradient of the orientation difference. Shape: nv
        # get orientation difference
        R = pose[:3, :3]
        rot_diff = target_pose[:3,:3].dot(R.T)
        w_diff = so3.log_quaternion(rot_diff)  # shape: 3
        jac1 = so3.Dx_log_x_quaternion(rot_diff).reshape((3,3,3)) # shape: 3x9
        # d (R_target * R_N^T) / d R_N
        # TODO: verify the following
        jac2 = np.zeros((3,3,3,3))
        jac2[:,0,0,:] = target_pose[:3,:3]
        jac2[:,1,1,:] = target_pose[:3,:3]
        jac2[:,2,2,:] = target_pose[:3,:3]
        # TODO: currently this is wrong since when the axis is zero, the derivative should be undefined.
        if np.linalg.norm(w_diff) == 0.0:
            axis = np.zeros((3))
        else:
            axis = w_diff / np.linalg.norm(w_diff)
        # axis = w_diff # NOTE for the case of square of error
        jac_r = np.tensordot(axis, jac1, axes=1)  # shape: 3x3  
        # NOTE: notice that np.dot has weird behaviors taht do not use the first axis of the second array
        jac_r = np.tensordot(jac_r, jac2, axes=2) # shape: 3x3
        rotation_diff_gradient = np.tensordot(jac_r, link_jac[:3,:3,:], axes=2) # shape: nv
        # TODO: sparse gradient
        gradient = np.zeros((self.n_waypoints*self.ndof))
        gradient[-self.ndof:] = position_diff_gradient + rotation_diff_gradient
        return gradient

    def terminal_pose_trajopt_f(self, qs, grad):
        """
        set the objective function according to nlopt interface.
        """
        if grad.size > 0:
            grad[:] = self.terminal_pose_gradient(qs)
        val = self.terminal_pose_objective(qs)
        print('terminal pose distance value: ', val)
        return val

    def objective(self, qs):
        """
        return the scalar value of the objective given the trajectory qs.
        """
        waypoint_distance = self.waypoint_distance_objective(qs)
        terminal_pose = self.terminal_pose_objective(qs)
        print('waypoint distance objective: ', waypoint_distance)
        print('terminal pose distance value: ', terminal_pose)
        return self.waypoint_distance_objective_weight*waypoint_distance + self.terminal_pose_objective_weight*terminal_pose
        # return waypoint_distance
    

    def gradient(self, qs):
        return self.waypoint_distance_objective_weight*self.waypoint_distance_gradient(qs) + self.terminal_pose_objective_weight*self.terminal_pose_gradient(qs)
        # return self.waypoint_distance_gradient(qs)

    def objective_trajopt_f(self, qs, grad):
        """
        set the objective function according to nlopt interface.
        """
        if grad.size > 0:
            grad[:] = self.gradient(qs)
        val = self.objective(qs)
        print('objective value: ', val)
        return val

    def compute_collision_constraints(self, qs):
        qs = np.array(qs).reshape(self.n_waypoints, self.ndof)
        robot = self.robot
        scene = self.scene
        cc_margin = 0.05
        constrs = []
        jacs = []  # jacobian shape: n_waypoints*n_cols x n_waypoints*ndof
        row = []
        col = []
        for i in range(self.n_waypoints):
            # start_time = time.time()
            robot.set_selected_joint_values(qs[i])
            # handle self collision
            distance_results = scene.compute_collision_min_dist_total(dist_upper_bound=cc_margin, security_margin=self.safety_margin, full=True)
            # collision_results = scene.compute_collision_total(security_margin=cc_margin, full=True)
            # print('compute collision time: ', time.time()-start_time)

            # start_time = time.time()  # time for computing jacobian
            # if not in collision, then return the min distance
            for col_i in range(len(distance_results)):
                link1, link2, obj_idx1, obj_idx2, distance_result = distance_results[col_i]
                # obtain the pose of the two links
                pose1 = robot.get_link_pose(link1)
                pose1_inv = np.linalg.inv(pose1)
                pose2 = np.eye(4)
                pose2_inv = np.eye(4)
                if link2 != "scene":
                    pose2 = robot.get_link_pose(link2)
                    pose2_inv = np.linalg.inv(pose2)

                jac = np.zeros((len(robot.selected_joint_dofids)))
                if distance_result['distance'] >= cc_margin:
                    dist = cc_margin
                    # constrs.append(cc_margin)
                    # continue  # need to compute jacobian as well
                else:
                    # dist = 0.0 # at least one contact. So we compute the mean of the contact distance
                    dist = distance_result['distance']
                    jac = np.zeros((len(robot.selected_joint_dofids)))
                    p1 = distance_result['p1']
                    p2 = distance_result['p2']
                    p1_local = pose1_inv[:3,:3]@p1 + pose1_inv[:3,3]
                    p2_local = pose2_inv[:3,:3]@p2 + pose2_inv[:3,3]
                    normal = distance_result['normal']
                    # normal = normal / np.linalg.norm(normal)
                    if np.linalg.norm(normal) <= 1e-10:
                        normal = np.array([1.0, 0.0, 0.0])
                    else:
                        normal = normal / np.linalg.norm(normal)
                    if link2 != "scene":
                        jac1 = robot.get_point_on_link_spatial_jacobian(link1, p1_local)
                        jac2 = robot.get_point_on_link_spatial_jacobian(link2, p2_local)
                        jac = np.tensordot(normal, jac2 - jac1, axes=1)
                        # jac_i = normal@(jac1 - jac2)
                    else:
                        # collision with environment:
                        # d g(q) / d q = 1/g(q)*J_p1(q)
                        jac1 = robot.get_point_on_link_spatial_jacobian(link1, p1_local)
                        jac = np.tensordot(normal, -jac1, axes=1)
                        # jac_i = normal@jac1

                constrs.append(dist)
                jacs.append(jac)  # shape: ndof
                row_i = np.zeros(jac.shape) + i*len(distance_results)+col_i # this is the index for the collisions
                                                                             # shape: ndof
                column_i = np.arange(len(jac)) + i*self.ndof # this is the index for the joint angles
                row.append(row_i)
                col.append(column_i)
            # print('compute jacobian time: ', time.time()-start_time)
        constrs = np.array(constrs) # self.n_waypoints*len(collision_results)
        jacs_data = np.array(jacs).flatten()  # shape: n_waypoints*n_cols x ndof
        row = np.array(row).flatten()
        col = np.array(col).flatten()
        jacs = sp.sparse.coo_array((jacs_data, (row, col)), shape=(self.n_waypoints*len(distance_results),
                                                                   self.n_waypoints*self.ndof))
        # jacs = jacs_data
        self.prev_qs = qs.flatten()
        self.prev_collision_constrs = constrs
        self.prev_collision_jacs = jacs

    def compute_collision_constraints_prev(self, qs):
        qs = np.array(qs).reshape(self.n_waypoints, self.ndof)
        robot = self.robot
        scene = self.scene
        cc_margin = 0.05
        constrs = []
        jacs = []  # jacobian shape: n_waypoints*n_cols x n_waypoints*ndof
        row = []
        col = []
        for i in range(self.n_waypoints):
            # start_time = time.time()
            robot.set_selected_joint_values(qs[i])
            # handle self collision
            collision_results = scene.compute_collision_total(security_margin=cc_margin, full=True)
            # print('compute collision time: ', time.time()-start_time)

            # start_time = time.time()  # time for computing jacobian
            # if not in collision, then return the min distance
            for col_i in range(len(collision_results)):
                link1, link2, obj_idx1, obj_idx2, collision_result = collision_results[col_i]
                # obtain the pose of the two links
                pose1 = robot.get_link_pose(link1)
                pose1_inv = np.linalg.inv(pose1)
                pose2 = np.eye(4)
                pose2_inv = np.eye(4)
                if link2 != "scene":
                    pose2 = robot.get_link_pose(link2)
                    pose2_inv = np.linalg.inv(pose2)
                if not collision_result.isCollision():
                    dist = cc_margin
                    # constrs.append(cc_margin)
                    # continue  # need to compute jacobian as well
                else:
                    dist = 0.0 # at least one contact. So we compute the mean of the contact distance
                # otherwise, compute jacobian. use the mean of each contact point
                # jac = np.zeros((robot.nv, 1))
                jac = np.zeros((len(robot.selected_joint_dofids)))
                n_contacts = collision_result.numContacts()
                for contact_i in range(n_contacts):
                    contact = collision_result.getContact(contact_i)
                    dist += (-contact.penetration_depth)
                    # * compute jacobian
                    # self collision:
                    # d g(q) / d q = 1/g(q)*(J_p1(q) - J_p2(q))
                    # NOTE: the point p1 and p2 are in the world frame
                    # it seems that the contact point is the intermediate point between the two objects
                    normal = np.array(contact.normal)  # normal: p2 - p1
                    pt = np.array(contact.pos)
                    p1 = pt - normal * 0.5 * (-contact.penetration_depth)
                    p2 = pt + normal * 0.5 * (-contact.penetration_depth)
                    # get the point on the link
                    p1_local = pose1_inv[:3,:3]@p1 + pose1_inv[:3,3]
                    p2_local = pose2_inv[:3,:3]@p2 + pose2_inv[:3,3]
                    if link2 != "scene":
                        jac1 = robot.get_point_on_link_spatial_jacobian(link1, p1_local)
                        jac2 = robot.get_point_on_link_spatial_jacobian(link2, p2_local)
                        jac_i = np.tensordot(normal, jac2 - jac1, axes=1)
                        # jac_i = normal@(jac1 - jac2)
                    else:
                        # collision with environment:
                        # d g(q) / d q = 1/g(q)*J_p1(q)
                        jac1 = robot.get_point_on_link_spatial_jacobian(link1, p1_local)
                        jac_i = np.tensordot(normal, -jac1, axes=1)
                        # jac_i = normal@jac1
                    jac += jac_i

                if n_contacts > 0:
                    dist = dist / n_contacts
                    jac = jac / n_contacts
                constrs.append(dist)
                jacs.append(jac)  # shape: ndof
                row_i = np.zeros(jac.shape) + i*len(collision_results)+col_i # this is the index for the collisions
                                                                             # shape: ndof
                column_i = np.arange(len(jac)) + i*self.ndof # this is the index for the joint angles
                row.append(row_i)
                col.append(column_i)
            # print('compute jacobian time: ', time.time()-start_time)
        constrs = np.array(constrs) # self.n_waypoints*len(collision_results)
        jacs_data = np.array(jacs).flatten()  # shape: n_waypoints*n_cols x ndof
        row = np.array(row).flatten()
        col = np.array(col).flatten()
        jacs = sp.sparse.coo_array((jacs_data, (row, col)), shape=(self.n_waypoints*len(collision_results),
                                                                   self.n_waypoints*self.ndof))
        # jacs = jacs_data
        self.prev_qs = qs.flatten()
        self.prev_collision_constrs = constrs
        self.prev_collision_jacs = jacs

    def collision_constraints(self, qs):
        """
        return the collision distance constraints between each pairs of self links and environment.
        g(q_i) = col_dist(q_i) >= safety_margin
        g(q_i) = max(col_dist(q_i), margin) >= safety_margin
        shape: n_waypoints * n_pairs
        when the two objects distance are below some threshold, consider the collision as a cosntraint.
        Otherwise, the collision distance is at least teh threshold.
        TODO: (important) only compute jacobian info once. Need to store results after the first call.
        TODO: parallel computation of collision distance
        """
        if hasattr(self, 'prev_qs'):
            if np.allclose(qs, self.prev_qs, rtol=1e-8, atol=1e-10):
                return self.prev_collision_constrs
        self.compute_collision_constraints(qs)
        return self.prev_collision_constrs

    def collision_jacobian(self, qs):
        if hasattr(self, 'prev_qs'):
            if np.allclose(qs, self.prev_qs, rtol=1e-8, atol=1e-10):
                return self.prev_collision_jacs
        self.compute_collision_constraints(qs)
        return self.prev_collision_jacs

    def collision_constraint_nlopt_c(self, result, x, grad):
        """
        set the collision constraints according to nlopt interface.
        the constraint is:
        c <= 0
        For us it would be:
        -c + safety_margin <= 0
        """
        if grad.size > 0:
            grad[:] = -self.collision_jacobian(x).toarray()
        result[:] = -self.collision_constraints(x) + self.constr_safety_margin
        print('collision violation: ')
        print(np.sum(result[result > 0]))

    def position_distance_constraints(self, qs):
        """
        position distance: ||q_{i+1}-q_i|| <= max_dist
        pd_constrs[i] = ||q_{i}-q_{i-1}||  (i=0,1,...,n_waypoints-1)
        TODO: verify this
        """
        qs = np.array(qs).reshape(self.n_waypoints, self.ndof)
        diff0 = qs[0] - self.start_q
        norm_diff0 = np.linalg.norm(diff0)
        diff = np.diff(qs, n=1, axis=0)
        diff = np.linalg.norm(diff, ord=2, axis=1)
        diff = np.insert(diff, 0, norm_diff0, 0)
        return diff
    
    def position_distance_jacobian(self, qs):
        """
        return the jacobian of the position distance constraints.
        d_pd_constrs[0] / d q[0] = (q[0]-q0)/||q[0]-q0||
        d_pd_constrs[i] / d q[i] = (q[i]-q[i-1])/||q[i]-q[i-1]||
        d_pd_constrs[i] / d q[i-1] = (q[i]-q[i-1])/||q[i]-q[i-1]||*(-1)

        J[0, 0:ndof] = (q[0]-q0)/||q[0]-q0||
        J[i, i*ndof:(i+1)*ndof] = (q[i]-q[i-1])/||q[i]-q[i-1]||
        J[i, (i-1)*ndof:i*ndof] = (q[i]-q[i-1])/||q[i]-q[i-1]||*(-1)
        TODO: verify this is correct
        """
        qs = np.array(qs).reshape(self.n_waypoints, self.ndof)
        diff = np.diff(qs, n=1, axis=0)
        # shape: (self.n_waypoints-1) x self.ndof
        # diff[i] = (q_i-q_{i-1})/||q_{i}-q_{i-1}||
        diff0 = qs[0] - self.start_q
        diff0 = diff0 / np.linalg.norm(diff0)
        diff = diff / np.linalg.norm(diff, ord=2, axis=1)[:, np.newaxis] 
        indices_row = np.arange(self.n_waypoints)[:, np.newaxis]
        indices_col = np.arange(self.ndof).reshape((1,-1))
        row = np.zeros((self.n_waypoints, 2*self.ndof)) + indices_row
        col = np.zeros((self.n_waypoints, 2*self.ndof))
        col[1:, :self.ndof] = (row[1:, :self.ndof]-1) * self.ndof + indices_col
        col[:, self.ndof:] = (row[:, self.ndof:]) * self.ndof + indices_col
        data = np.zeros((self.n_waypoints, 2*self.ndof))
        data[0, self.ndof:] = diff0
        data[1:, self.ndof:] = diff
        data[1:, :self.ndof] = -diff
        # for the first waypoint, ignore the first ndof cols
        row = row.flatten()[self.ndof:]
        col = col.flatten()[self.ndof:]
        data = data.flatten()[self.ndof:]
        return sp.sparse.coo_array((data, (row, col)), shape=(self.n_waypoints, self.n_waypoints*self.ndof))

    def position_distance_constraint_nlopt_c(self, result, qs, grad):
        """
        set the position distance constraints according to nlopt interface.
        the constraint is:
        c <= 0
        For us it would be:
        c - max_dist <= 0
        """
        if grad.size > 0:
            grad[:] = self.position_distance_jacobian(qs).toarray()
        result[:] = self.position_distance_constraints(qs) - self.constr_max_dist
        print('position distance constraints violation: ')
        print(np.sum(result[result > 0]))

    # TODO: add constuction of each constraint lower and upper limits from class methods
    def joint_limit_constraints(self, qs):
        """
        joint limits: lb <= q[i] <= ub
        shape: n_waypoints x ndof
        """
        return qs

    def joint_limit_jacobian(self, qs):
        """
        jacobian is the identify matrix of size n_waypoints*ndof
        """
        # return sp.sparse.eye_array(len(qs))
        # return sp.sparse.identity(len(qs), format='dia')
        return np.ones((len(qs))).astype(int)

    def constraints(self, qs):
        """
        obtain all constraints and put them together
        """
        collision_constrs = self.collision_constraints(qs)
        position_constrs = self.position_distance_constraints(qs)
        # joint_limit_constrs = self.joint_limit_constraints(qs)
        # return np.concatenate([collision_constrs, position_constrs, joint_limit_constrs], axis=0)
        return np.concatenate([collision_constrs, position_constrs], axis=0)

    def jacobian(self, x):
        """
        obtain the jacobian of all constraints and put them together
        """
        collision_jac = self.collision_jacobian(x)
        position_jac = self.position_distance_jacobian(x)
        # joint_limit_jac = self.joint_limit_jacobian(x)
        # return sp.sparse.vstack([collision_jac, position_jac, joint_limit_jac])
        return np.concatenate([collision_jac, position_jac], axis=0)
