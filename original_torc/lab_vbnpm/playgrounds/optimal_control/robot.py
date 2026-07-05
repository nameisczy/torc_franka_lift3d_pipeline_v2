"""
specify the robot model for collision checking (self collision) and Jacobian computation.
implemented by mujoco, hpp-fcl, open3d (visualization)
"""

import so3, se3
import mujoco
import hppfcl
import numpy as np
import transformations as tf
import open3d as o3d
import copy
from typing import Union
import numpy.typing as npt
from collections import deque

"""
compute the kinematics trees. The structure is the following:
each node stores the following:
- link name
- link idx
- parent node
- children nodes
- Pi_i exp(xi_i*theta_i) up to the current node
- joints (order as it appears in the XML)
    - joint name
    - joint idx
    - xi_i at rest pose (theta=0)
    - Pi_i exp(xi_i*theta_i) up to the current node, and up to the current joint (since Mujoco joint computation is accoriding to
    the order of the joints in the XML file)
"""


class KinematicsNode:
    class KinematicJoint:
        def __init__(self, joint_idx, link_pose_zeroed, mj_model):
            """
            link_pose_zeroed: the pose of the link when the joint angles are zero
            """
            self.joint_idx = joint_idx
            self.joint_name = mj_model.joint(joint_idx).name
            self.qpos_adr = mj_model.jnt_qposadr[joint_idx]
            self.dof_adr = mj_model.jnt_dofadr[joint_idx]
            # obtain the axis for the joint
            jnt_pos = np.array(mj_model.jnt_pos[joint_idx])
            jnt_pos = link_pose_zeroed[:3,:3].dot(jnt_pos) + link_pose_zeroed[:3,3]
            jnt_axis = np.array(mj_model.jnt_axis[joint_idx])
            jnt_axis = jnt_axis / np.linalg.norm(jnt_axis)
            jnt_axis = link_pose_zeroed[:3,:3].dot(jnt_axis)
            # by default: assume to be hinge joint
            # TODO: add slider joint as well
            xi = se3.screw(jnt_axis, jnt_pos, 1.0, 0.0)  # this is the unit screw
            self.xi = xi
            self.theta = 0.0
            self.product_exp_xi_theta = None  # default: invalid value
            self.xi_transformed = None
            self.mj_model = mj_model

        def set_theta(self, theta):
            self.theta = theta

        def compute_product_exp_xi_theta(self, parent_product_exp_xi_theta):
            """
            compute the product of the exponential of xi*theta up to the current joint
            """
            self.product_exp_xi_theta = parent_product_exp_xi_theta @ se3.mat_exp(se3.hat(self.xi), self.theta)
            self.parent_product_exp_xi_theta = parent_product_exp_xi_theta
            return np.array(self.product_exp_xi_theta)

        def compute_spatial_jacobian_col(self):
            """
            compute the col in the spatial jacobian matrix.
            xi_transformed = adjoint(parent_product_exp_xi_theta) * xi
            NOTE: assume that the product_exp_xi_theta has been updated.
            """
            adjoint = se3.adjoint(self.parent_product_exp_xi_theta)
            xi_transformed = adjoint @ self.xi
            self.xi_transformed = xi_transformed
            return np.array(xi_transformed)

    def __init__(self, link_idx, parent_node, mj_model, mj_data_zeroed):
        """
        assume mj_data is zeroed, and the joint values are zeroed.
        """
        self.link_idx = link_idx
        self.link_name = mj_model.body(link_idx).name
        # self.link_idx = mj_model.body(link_name).id
        self.parent_node = parent_node
        self.children_nodes = []  # initialize as empty
        self.ancestor_nodes = []
        self.relavent_joint_indices = []  # relavent joints include the joints of the ancestor nodes and joints of this node
        self.relavent_joint_qpos_adrs = []
        self.relavent_joint_dofs = []
        # obtain the pose of the link from data
        link_pose = tf.quaternion_matrix([mj_data_zeroed.xquat[self.link_idx,0],
                                          mj_data_zeroed.xquat[self.link_idx,1],
                                          mj_data_zeroed.xquat[self.link_idx,2],
                                          mj_data_zeroed.xquat[self.link_idx,3]])
        link_pose[0,3] = mj_data_zeroed.xpos[self.link_idx,0]
        link_pose[1,3] = mj_data_zeroed.xpos[self.link_idx,1]
        link_pose[2,3] = mj_data_zeroed.xpos[self.link_idx,2]
        # self.link_pose = link_pose
        # set the data for each joint
        
        self.joints = []
        jnt_num = mj_model.body_jntnum[self.link_idx]
        jnt_adr = mj_model.body_jntadr[self.link_idx]
        for i in range(jnt_num):
            jnt_idx = i + jnt_adr
            joint_name = mj_model.joint(jnt_idx).name
            joint = self.KinematicJoint(jnt_idx, link_pose, mj_model)
            self.joints.append(joint)
        self.link_pose_q_zeroed = link_pose
        self.product_exp_xi_theta = None # this is the value at the last joint 
        self.mj_model = mj_model
        self._add_self_joints_to_relavent_joints()  # initialization. relavent joints include self joints

    def set_parent_node(self, parent_node):
        self.parent_node = parent_node

    def set_children_nodes(self, children_nodes):
        self.children_nodes = children_nodes

    def add_to_children(self, child_node: 'KinematicsNode'):
        self.children_nodes.append(child_node)

    def _add_self_joints_to_relavent_joints(self):
        self.relavent_joint_indices = [joint.joint_idx for joint in self.joints]
        self.relavent_joint_qpos_adrs = [joint.qpos_adr for joint in self.joints]
        self.relavent_joint_dofs = [joint.dof_adr for joint in self.joints]

    def set_ancestor_nodes(self, ancestor_nodes):
        self.ancestor_nodes = ancestor_nodes
        # set relavent joint indices to be this node's joints plus the ancestors
        self._add_self_joints_to_relavent_joints()
        for ancestor_node in self.ancestor_nodes:
            self.relavent_joint_indices.extend([joint.joint_idx for joint in ancestor_node.joints])
            self.relavent_joint_qpos_adrs.extend([joint.qpos_adr for joint in ancestor_node.joints])
            self.relavent_joint_dofs.extend([joint.dof_adr for joint in ancestor_node.joints])

    def add_to_ancestor(self, ancestor_node: 'KinematicsNode'):
        self.ancestor_nodes.append(ancestor_node)
        # add the joint of the ancestor node to the stored list
        self.relavent_joint_indices.extend([joint.joint_idx for joint in ancestor_node.joints])
        self.relavent_joint_qpos_adrs.extend([joint.qpos_adr for joint in ancestor_node.joints])
        self.relavent_joint_dofs.extend([joint.dof_adr for joint in ancestor_node.joints])

    def update_joint_values(self, mj_data: mujoco.MjData):
        """
        given the mujoco data, update the joint values of the joints in the node
        """
        for joint in self.joints:
            joint.set_theta(mj_data.qpos[joint.qpos_adr])

    def compute_product_exp_xi_theta(self, parent_product_exp_xi_theta):
        """
        compute the product of the exponential of xi*theta up to the current node
        for each of the joint
        return the product for the last joint of this body
        """
        product_exp_xi_theta = parent_product_exp_xi_theta
        for joint in self.joints:
            product_exp_xi_theta = joint.compute_product_exp_xi_theta(product_exp_xi_theta)
        self.product_exp_xi_theta = product_exp_xi_theta
        self.parent_product_exp_xi_theta = parent_product_exp_xi_theta
        return np.array(product_exp_xi_theta)

    def compute_spatial_jacobian_cols(self):
        """
        compute the twist column in the spatial jacobian matrix for each joint in this link.
        NOTE: assume that the product_exp_xi_theta has been updated.
        """
        cols = []
        for joint in self.joints:
            xi_transformed = joint.compute_spatial_jacobian_col()
            cols.append(xi_transformed)
        self.spatial_jacobian_cols = cols
        return cols


class KinematicsTree:
    def __init__(self, mj_model):
        """
        initialize the kinematics tree given the mujoco model and data.
        set the data to have zero joint values, and then obtain the rotation axis
        of each joint.
        Afterwards, reset the data to the input.
        """
        # * obtain the zeroed mj_data
        mj_data_zeroed = mujoco.MjData(mj_model)
        # * set the joint values to zero
        for i in range(mj_model.nq):
            mj_data_zeroed.qpos[i] = 0.0
        mujoco.mj_forward(mj_model, mj_data_zeroed)
        # * loop over each link, construct the kinematic node
        nodes = []
        roots = []
        for body_id in range(mj_model.nbody):
            node = KinematicsNode(body_id, None, mj_model, mj_data_zeroed)
            nodes.append(node)
        # * set the parent and children nodes
        for body_idx in range(mj_model.nbody):
            parent_id = mj_model.body_parentid[body_idx]
            if parent_id != body_idx:  # not root
                nodes[body_idx].set_parent_node(nodes[parent_id])
                nodes[parent_id].add_to_children(nodes[body_idx])
            else:
                roots.append(nodes[body_idx])
        self.nodes = nodes
        self.roots = roots
        self.mj_model = mj_model

        # * go through the kinematic tree to obtain the relavent joint indices for each link
        # this is useful for computing the spatial jacobian.
        q = deque()
        for node in self.roots:
            node.set_ancestor_nodes([])
            q.append(node)
            # set the product of the exponential of xi*theta to identity
        while len(q) > 0:
            node = q.popleft()
            for child in node.children_nodes:
                child.set_ancestor_nodes(node.ancestor_nodes + [node])  # this sets the relavent joints too
                q.append(child)

    def update_joint_values(self, mj_data: mujoco.MjData):
        for node in self.nodes:
            node.update_joint_values(mj_data)

    def compute_product_exp_xi_theta(self):
        """
        do a BFS search on the kinematic tree to compute the product of the 
        exponential of xi*theta up to the current node
        """
        q = deque()
        for node in self.roots:
            node.compute_product_exp_xi_theta(np.eye(4))
            q.append(node)
            # set the product of the exponential of xi*theta to identity
        while len(q) > 0:
            node = q.popleft()
            for child in node.children_nodes:
                child.compute_product_exp_xi_theta(node.product_exp_xi_theta)
                q.append(child)

    def compute_spatial_jacobian_cols(self, compute_product_exp_xi_theta: bool = False):
        """
        compute the twist column in the spatial jacobian matrix
        xi_jac_i = Ad_{Pi_{j=1}^{i-1}}xi_jac_i
        here the adjoint pose is the product of exponential of xi*theta until the previous joint of current joint.
        """
        if compute_product_exp_xi_theta:
            self.compute_product_exp_xi_theta()
        cols = np.zeros((6, self.mj_model.nv))
        for i in range(len(self.nodes)):
            cols_i = self.nodes[i].compute_spatial_jacobian_cols()
            # put the columns at their respective indices
            for j in range(len(cols_i)):
                cols[:,self.nodes[i].joints[j].dof_adr] = cols_i[j]
        self.spatial_jacobian_cols = cols
        return np.array(cols)

    def get_link_pose(self, link_name: str, compute_product_exp_xi_theta: bool = False) -> np.ndarray:
        """
        get the link pose given the computed product of the exponential of xi*theta
        the pose is given as: Pi_i exp(xi_i*theta_i) * g_i(0)
        using the current joint values stored in the kinematic tree
        NOTE: assume the joint values have been updated
        """
        if compute_product_exp_xi_theta:
            self.compute_product_exp_xi_theta()
        # get the node with the link name
        link_idx = self.mj_model.body(link_name).id
        node = self.nodes[link_idx]
        print('node.product_exp_xi_theta:', node.product_exp_xi_theta)
        print('node.link_pose_q_zeroed:', node.link_pose_q_zeroed)
        return node.product_exp_xi_theta @ node.link_pose_q_zeroed

    def get_link_spatial_jacobian(self, link_name: str, compute_spatial_jacobian_cols: bool = False,
                                  compute_product_exp_xi_theta: bool = False) -> np.ndarray:
        """
        get the spatial jacobian for the specific link
        """
        if compute_spatial_jacobian_cols:
            self.compute_spatial_jacobian_cols(compute_product_exp_xi_theta)
        cols = np.array(self.spatial_jacobian_cols)
        # * set irrelavent link cols to zero
        # obtain the relavent joint dofs
        link_idx = self.mj_model.body(link_name).id
        node = self.nodes[link_idx]
        relavent_dofs = node.relavent_joint_dofs
        irrelavent_dofs = set(range(self.mj_model.nv)) - set(relavent_dofs)
        irrelavent_dofs = np.array(list(irrelavent_dofs))
        cols[:,irrelavent_dofs] = 0.0
        return cols


class Robot:
    def __init__(self, model_path: str,
                 default_joint_value_dict: dict = None,
                 selected_joint_names: list = None,
                 disabled_collision_pairs: set = None,
                 compute_kinematics_tree: bool = False):
        """
        :param model_path: the path to the robot model xml file
        :param default_joint_value_dict: a dictionary of joint names and their default values. This is for full joint names
        :param selected_joint_names: a list of joint names that are selected for planning. 
                                     When None, it is full joint names extracted from XML file
        :param disabled_collision_pairs: a set of pairs of body ids that should not be checked for collision
        :param compute_kinematics_tree: whether to compute the kinematics structure and Jacobians manually
        """
        # * load robot model
        mj_model = mujoco.MjModel.from_xml_path(model_path)
        mj_data = mujoco.MjData(mj_model)
        self.mj_model = mj_model
        self.mj_data = mj_data
        
        # * handle the default values by setting them
        if default_joint_value_dict is not None:
            for joint_name, value in default_joint_value_dict.items():
                mj_data.qpos[mj_model.jnt_qposadr[mj_model.joint(joint_name).id]] = value
        mujoco.mj_forward(mj_model, mj_data)

        # * set the selected joint names
        robot_joint_names = []  # we keep a copy of all the joint names
        robot_joint_limits = []
        robot_joint_values = []  # TODO: when doing kinematics, need to store the vels and accs
        robot_joint_dofids = []
        for i in range(mj_model.njnt):
            joint_name = mj_model.joint(i).name
            robot_joint_names.append(joint_name)
            robot_joint_limits.append([mj_model.jnt_range[i,0], mj_model.jnt_range[i,1]])
            robot_joint_values.append(mj_data.qpos[mj_model.jnt_qposadr[i]])
            robot_joint_dofids.append(mj_model.jnt_dofadr[i])
        robot_joint_limits = np.array(robot_joint_limits)
        robot_joint_values = np.array(robot_joint_values)
        robot_joint_dofids = np.array(robot_joint_dofids)
        self.robot_joint_names = robot_joint_names
        self.robot_joint_limits = robot_joint_limits
        self.robot_joint_values = robot_joint_values
        self.robot_joint_dofids = robot_joint_dofids

        selected_joint_values = []
        selected_joint_limits = []
        selected_joint_dofids = []
        if selected_joint_names is None:
            selected_joint_names = copy.deepcopy(robot_joint_names)
            selected_joint_limits = np.array(robot_joint_limits)
            selected_joint_values = np.array(self.robot_joint_values)
            selected_joint_dofids = np.array(robot_joint_dofids)
        else:
            # when the value is not None, select joint names
            for i in range(len(selected_joint_names)):
                joint = mj_model.joint(selected_joint_names[i])
                selected_joint_limits.append([joint.range[0], joint.range[1]])
                selected_joint_values.append(mj_data.qpos[mj_model.jnt_qposadr[joint.id]])
                selected_joint_dofids.append(mj_model.jnt_dofadr[joint.id])
            selected_joint_limits = np.array(selected_joint_limits)
            selected_joint_values = np.array(selected_joint_values)
            selected_joint_dofids = np.array(selected_joint_dofids)
        self.selected_joint_names = selected_joint_names
        self.selected_joint_limits = selected_joint_limits
        self.selected_joint_values = selected_joint_values
        self.selected_joint_dofids = selected_joint_dofids

        # * store all the robot link names
        robot_link_names = []
        for i in range(mj_model.nbody):
            # print(mj_model.body(i).name)
            if mj_model.body(i).name == "world":
                continue
            robot_link_names.append(mj_model.body(i).name)
        self.robot_link_names = robot_link_names

        # * all the robot link ids
        robot_link_ids =[]
        for i in range(mj_model.nbody):
            if mj_model.body(i).name == "world":
                continue
            robot_link_ids.append(i)
        self.robot_link_ids = robot_link_ids

        # * construct the mapping from robot links to geoms
        # create the dictionary from robot_links to the list of collision meshes of geoms in the link
        # for robot links, the collision meshes start from "c_"
        # for pad box, they start from "c_", and are of box shape
        robot_link_name_to_geoms = {}
        for link in robot_link_names:
            robot_link_name_to_geoms[link] = []
            body_idx = mj_model.body(link).id
            for geom_idx in range(mj_model.body_geomadr[body_idx], mj_model.body_geomadr[body_idx]+mj_model.body_geomnum[body_idx]):
                if mj_model.geom(geom_idx).name.startswith("c_"):
                    geom = {}
                    if mj_model.geom_type[geom_idx] == mujoco.mjtGeom.mjGEOM_MESH:
                        # handle mesh here
                        mesh_idx = mj_model.geom_dataid[geom_idx]
                        vert_idx = mj_model.mesh_vertadr[mesh_idx]
                        vert_num = mj_model.mesh_vertnum[mesh_idx]
                        face_idx = mj_model.mesh_faceadr[mesh_idx]
                        face_num = mj_model.mesh_facenum[mesh_idx]
                        mesh_vertices = []
                        for i in range(vert_num):
                            vert = [mj_model.mesh_vert[vert_idx+i,0],mj_model.mesh_vert[vert_idx+i,1],mj_model.mesh_vert[vert_idx+i,2]]
                            mesh_vertices.append(vert)
                        mesh_faces = [] 
                        for i in range(face_num):
                            face = [mj_model.mesh_face[face_idx+i,0],mj_model.mesh_face[face_idx+i,1],mj_model.mesh_face[face_idx+i,2]]
                            mesh_faces.append(face)

                        mesh_vertices = np.array(mesh_vertices).astype(float)
                        mesh_faces = np.array(mesh_faces).astype(int)

                        # TODO: unsure if the vertices are scaled. Need to check
                        # TODO: unsure if the mesh_pos and mesh_quat fields need to be applied.
                        # TODO: unsure of the order to apply scale first or pose
                        mesh_pose = tf.quaternion_matrix([mj_model.mesh_quat[mesh_idx,0],
                                                          mj_model.mesh_quat[mesh_idx,1],
                                                          mj_model.mesh_quat[mesh_idx,2],
                                                          mj_model.mesh_quat[mesh_idx,3]])  # w,x,y,z
                        mesh_pose[0,3] = mj_model.mesh_pos[mesh_idx,0]
                        mesh_pose[1,3] = mj_model.mesh_pos[mesh_idx,1]
                        mesh_pose[2,3] = mj_model.mesh_pos[mesh_idx,2]

                        mesh_scale = [mj_model.mesh_scale[mesh_idx,0],
                                    mj_model.mesh_scale[mesh_idx,1],
                                    mj_model.mesh_scale[mesh_idx,2]]
                        mesh_scale = np.array(mesh_scale)
                        # mesh_vertices = mesh_vertices * mesh_scale
                        # mesh_vertices = mesh_pose[:3,:3].dot(mesh_vertices.T).T + mesh_pose[:3,3]
                        geom['type'] = 'mesh'
                        geom['vertices'] = mesh_vertices
                        geom['faces'] = mesh_faces
                    elif mj_model.geom_type[geom_idx] == mujoco.mjtGeom.mjGEOM_BOX:
                        # handle box here
                        half_size = [mj_model.geom_size[geom_idx,0],
                                    mj_model.geom_size[geom_idx,1],
                                    mj_model.geom_size[geom_idx,2]] # half-size
                        geom['size'] = np.array(half_size)
                        geom['type'] = 'box'

                    pose = tf.quaternion_matrix([mj_model.geom_quat[geom_idx,0],
                                                mj_model.geom_quat[geom_idx,1],
                                                mj_model.geom_quat[geom_idx,2],
                                                mj_model.geom_quat[geom_idx,3]])  # w,x,y,z
                    pose[0,3] = mj_model.geom_pos[geom_idx,0]
                    pose[1,3] = mj_model.geom_pos[geom_idx,1]
                    pose[2,3] = mj_model.geom_pos[geom_idx,2]
                    geom['pose'] = pose
                    robot_link_name_to_geoms[link].append(geom)
        self.robot_link_name_to_geoms = robot_link_name_to_geoms

        # * store the transformations of each link
        robot_link_name_to_transform = {}
        for link in robot_link_names:
            bid = mj_model.body(link).id
            link_pose = tf.quaternion_matrix([mj_data.xquat[bid,0], mj_data.xquat[bid,1], mj_data.xquat[bid,2], mj_data.xquat[bid,3]])
            link_pose[0,3] = mj_data.xpos[bid,0]
            link_pose[1,3] = mj_data.xpos[bid,1]
            link_pose[2,3] = mj_data.xpos[bid,2]
            robot_link_name_to_transform[link] = link_pose
        self.robot_link_name_to_transform = robot_link_name_to_transform

        # * construct the mapping from robot links to collision objects
        robot_link_name_to_fcl_objs = {}
        for link, geoms in robot_link_name_to_geoms.items():
            robot_link_name_to_fcl_objs[link] = []
            link_pose = robot_link_name_to_transform[link]
            for geom in geoms:
                if geom['type'] == 'mesh':
                    mesh = hppfcl.BVHModelOBBRSS()
                    mesh.beginModel(len(geom['faces']), len(geom['vertices']))
                    for vert in geom['vertices']:
                        mesh.addVertex(vert)
                    for face in geom['faces']:
                        mesh.addTriangle(geom['vertices'][face[0]], geom['vertices'][face[1]], geom['vertices'][face[2]])
                    mesh.endModel()
                    pose = link_pose@geom['pose']
                    T = hppfcl.Transform3f(pose[:3,:3], pose[:3,3])
                    mesh_obj = hppfcl.CollisionObject(mesh, T)
                    robot_link_name_to_fcl_objs[link].append(mesh_obj)
                elif geom['type'] == 'box':
                    # hppfcl Box is centered at (0,0,0), and spans -half_size to half_size
                    half_size = geom['size']
                    pose = link_pose@geom['pose']
                    T = hppfcl.Transform3f(pose[:3,:3], pose[:3,3])            
                    box = hppfcl.CollisionObject(hppfcl.Box(2*half_size[0], 2*half_size[1], 2*half_size[2]), T)
                    robot_link_name_to_fcl_objs[link].append(box)
        self.robot_link_name_to_fcl_objs = robot_link_name_to_fcl_objs

        # * construct the mapping from robot links to open3d geometries
        # store the open3d geometries current transform
        robot_link_name_to_open3d_geoms_transform = {}
        robot_link_name_to_open3d_geoms = {}
        for link, geoms in robot_link_name_to_geoms.items():
            link_pose = robot_link_name_to_transform[link]
            robot_link_name_to_open3d_geoms[link] = []
            robot_link_name_to_open3d_geoms_transform[link] = []
            for geom in geoms:
                if geom['type'] == 'mesh':
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(geom['vertices'])
                    mesh.triangles = o3d.utility.Vector3iVector(geom['faces'])
                    mesh.compute_vertex_normals()
                    mesh.paint_uniform_color([0.5, 0.5, 0.5])
                    mesh.transform(link_pose@geom['pose'])
                    robot_link_name_to_open3d_geoms[link].append(mesh)
                    robot_link_name_to_open3d_geoms_transform[link].append(link_pose@geom['pose'])
                elif geom['type'] == 'box':
                    half_size = geom['size']
                    box = o3d.geometry.TriangleMesh.create_box(width=2*half_size[0], height=2*half_size[1], depth=2*half_size[2])
                    box.compute_vertex_normals()
                    box.paint_uniform_color([0.5, 0.5, 0.5])
                    o3d_box_pose = np.eye(4)
                    o3d_box_pose[0,3] = o3d_box_pose[0,3] - geom['size'][0]
                    o3d_box_pose[1,3] = o3d_box_pose[1,3] - geom['size'][1]
                    o3d_box_pose[2,3] = o3d_box_pose[2,3] - geom['size'][2]
                    box.transform(link_pose@geom['pose']@o3d_box_pose)
                    robot_link_name_to_open3d_geoms[link].append(box)
                    robot_link_name_to_open3d_geoms_transform[link].append(link_pose@geom['pose']@o3d_box_pose)
        self.robot_link_name_to_open3d_geoms = robot_link_name_to_open3d_geoms
        self.robot_link_name_to_open3d_geoms_transform = robot_link_name_to_open3d_geoms_transform

        # * generate collision pairs for the robot
        if disabled_collision_pairs is None:
            disabled_collision_pairs = set()
        self.disabled_collision_pairs = disabled_collision_pairs
        collision_pairs = []
        for i in range(len(robot_link_names)):
            for j in range(i+1,len(robot_link_names)):
                link1 = robot_link_names[i]
                link2 = robot_link_names[j]
                # if the two bodies are parent-child, skip
                if mj_model.body(link1).parentid == mj_model.body(link2).id:
                    continue
                if mj_model.body(link2).parentid == mj_model.body(link1).id:
                    continue
                if (link1, link2) in disabled_collision_pairs:
                    continue
                if (link2, link1) in disabled_collision_pairs:
                    continue
                collision_pairs.append((link1, link2))
        self.collision_pairs = collision_pairs

        self._construct_collision_pair_data()

        # initialize the collision requests and collision results for collision with the env.
        # These are stored as a dictionary
        # from robot_link, geom_id to collision request and collision result
        robot_link_to_collision_req = {}
        robot_link_to_collision_res = {}
        robot_link_to_distance_req = {}
        robot_link_to_distance_res = {}
        for link in robot_link_names:
            for geom_i in range(len(robot_link_name_to_fcl_objs[link])):
                robot_link_to_collision_req[(link, geom_i)] = hppfcl.CollisionRequest()
                robot_link_to_collision_res[(link, geom_i)] = hppfcl.CollisionResult()
                robot_link_to_distance_req[(link, geom_i)] = hppfcl.DistanceRequest()
                robot_link_to_distance_res[(link, geom_i)] = hppfcl.DistanceResult()
        self.robot_link_to_collision_req = robot_link_to_collision_req
        self.robot_link_to_collision_res = robot_link_to_collision_res
        self.robot_link_to_distance_req = robot_link_to_distance_req
        self.robot_link_to_distance_res = robot_link_to_distance_res

        # * compute the kinematics structure if the option is ON
        self.compute_kinematics_tree = compute_kinematics_tree
        if self.compute_kinematics_tree:
            self._init_kinematics_tree()

    def _init_kinematics_tree(self):
        """
        compute the kinematics trees. The structure is the following:
        each node stores the following:
        - link name
        - link idx
        - parent node
        - children nodes
        - Pi_i exp(xi_i*theta_i) up to the current node
        - joints (order as it appears in the XML)
            - joint name
            - joint idx
            - xi_i at rest pose (theta=0)
            - Pi_i exp(xi_i*theta_i) up to the current node, and up to the current joint
        """
        pass

    def _construct_collision_pair_data(self):
        # * construct the mapping from collision pairs (and geom ids) to distance and collision requests and results
        collision_pair_to_distance_req = {}
        collision_pair_to_distance_res = {}
        collision_pair_to_collision_req = {}
        collision_pair_to_collision_res = {}
        col_pair_num = 0
        for i in range(len(self.collision_pairs)):
            link1 = self.collision_pairs[i][0]
            link2 = self.collision_pairs[i][1]
            col_pair_num += len(self.robot_link_name_to_fcl_objs[link1]) * len(self.robot_link_name_to_fcl_objs[link2])
            for geom1_i in range(len(self.robot_link_name_to_fcl_objs[link1])):
                for geom2_i in range(len(self.robot_link_name_to_fcl_objs[link2])):
                    collision_pair_to_distance_req[(i, geom1_i, geom2_i)] = hppfcl.DistanceRequest()
                    collision_pair_to_distance_res[(i, geom1_i, geom2_i)] = hppfcl.DistanceResult()
                    collision_pair_to_collision_req[(i, geom1_i, geom2_i)] = hppfcl.CollisionRequest()
                    collision_pair_to_collision_res[(i, geom1_i, geom2_i)] = hppfcl.CollisionResult()
                    # collision_pair_to_distance_req[(link1, link2, geom1_i, geom2_i)].enable_signed_distance = True
                    # collision_pair_to_distance_req[(link1, link2, geom1_i, geom2_i)].enable_nearest_points = True
                    # collision_pair_to_collision_req[(link1, link2, geom1_i, geom2_i)].enable_contact = True
                    # collision_pair_to_collision_req[(link1, link2, geom1_i, geom2_i)].num_max_contacts = 1
        self.collision_pair_to_distance_req = collision_pair_to_distance_req
        self.collision_pair_to_distance_res = collision_pair_to_distance_res
        self.collision_pair_to_collision_req = collision_pair_to_collision_req
        self.collision_pair_to_collision_res = collision_pair_to_collision_res
        self.col_pair_num = col_pair_num
        # NOTE: the nearest_point is in the world frame
        # ref: https://github.com/humanoid-path-planner/hpp-fcl/blob/412cf1000470aec00166eaa239e881a2cfa46995/include/hpp/fcl/internal/traversal_node_bvhs.h#L517

    def disable_collision_pairs(self, disabled_collision_pairs: set):
        self.disabled_collision_pairs = disabled_collision_pairs
        self.collision_pairs = []
        for i in range(len(self.robot_link_names)):
            for j in range(i+1,len(self.robot_link_names)):
                link1 = self.robot_link_names[i]
                link2 = self.robot_link_names[j]
                # if the two bodies are parent-child, skip
                if self.mj_model.body(link1).parentid == self.mj_model.body(link2).id:
                    continue
                if self.mj_model.body(link2).parentid == self.mj_model.body(link1).id:
                    continue
                if (self.mj_model.body(link1).id, self.mj_model.body(link2).id) in disabled_collision_pairs:
                    continue
                if (self.mj_model.body(link2).id, self.mj_model.body(link1).id) in disabled_collision_pairs:
                    continue
                self.collision_pairs.append((link1, link2))

        self._construct_collision_pair_data()

    def _update_transform(self):
        for link in self.robot_link_names:
            bid = self.mj_model.body(link).id
            link_pose = tf.quaternion_matrix([self.mj_data.xquat[bid,0], 
                                              self.mj_data.xquat[bid,1], 
                                              self.mj_data.xquat[bid,2], 
                                              self.mj_data.xquat[bid,3]])
            link_pose[0,3] = self.mj_data.xpos[bid,0]
            link_pose[1,3] = self.mj_data.xpos[bid,1]
            link_pose[2,3] = self.mj_data.xpos[bid,2]
            self.robot_link_name_to_transform[link] = link_pose

    def _update_fcl_transform(self, from_transform=True):
        """
        update the transforms of the robot links given the current joint values set in self.mj_data
        """
        # update the transforms of each link
        for link, fcl_objs in self.robot_link_name_to_fcl_objs.items():
            if from_transform:
                link_pose = self.robot_link_name_to_transform[link]
            else:
                bid = self.mj_model.body(link).id
                link_pose = tf.quaternion_matrix([self.mj_data.xquat[bid,0], 
                                                self.mj_data.xquat[bid,1], 
                                                self.mj_data.xquat[bid,2], 
                                                self.mj_data.xquat[bid,3]])
                link_pose[0,3] = self.mj_data.xpos[bid,0]
                link_pose[1,3] = self.mj_data.xpos[bid,1]
                link_pose[2,3] = self.mj_data.xpos[bid,2]
            for obj_i in range(len(fcl_objs)):
                geom = self.robot_link_name_to_geoms[link][obj_i]
                obj_pose = link_pose@geom['pose']
                fcl_objs[obj_i].setTransform(hppfcl.Transform3f(obj_pose[:3,:3], obj_pose[:3,3]))

    def _update_joint_values(self):
        """
        get the total joint value list extracted from self.mj_data
        """
        for i in range(len(self.robot_joint_names)):
            self.robot_joint_values[i] = self.mj_data.qpos[self.mj_model.jnt_qposadr[self.mj_model.joint(self.robot_joint_names[i]).id]]

    def _update_selected_joint_values(self):
        """
        get the selected joint values extracted from self.mj_data
        """
        for i in range(len(self.selected_joint_names)):
            self.selected_joint_values[i] = self.mj_data.qpos[self.mj_model.jnt_qposadr[self.mj_model.joint(self.selected_joint_names[i]).id]]

    def visualize(self, show=True) -> list:
        """
        visualize the robot at the current joint values
        """
        o3d_objs = []
        for link, o3d_geoms in self.robot_link_name_to_open3d_geoms.items():
            # obtain the pose of the link
            link_pose = self.robot_link_name_to_transform[link]

            for geom_i in range(len(o3d_geoms)):
                geom = self.robot_link_name_to_geoms[link][geom_i]
                prev_transform = self.robot_link_name_to_open3d_geoms_transform[link][geom_i]
                if geom['type'] == 'mesh':
                    mesh = o3d_geoms[geom_i]
                    pose = link_pose@geom['pose']@np.linalg.inv(prev_transform)
                    mesh.transform(pose)
                    o3d_objs.append(mesh)
                    self.robot_link_name_to_open3d_geoms_transform[link][geom_i] = link_pose@geom['pose']
                elif geom['type'] == 'box':
                    box = o3d_geoms[geom_i]
                    o3d_box_pose = np.eye(4)
                    o3d_box_pose[0,3] = o3d_box_pose[0,3] - geom['size'][0]
                    o3d_box_pose[1,3] = o3d_box_pose[1,3] - geom['size'][1]
                    o3d_box_pose[2,3] = o3d_box_pose[2,3] - geom['size'][2]
                    pose = link_pose@geom['pose']@o3d_box_pose@np.linalg.inv(prev_transform)
                    box.transform(pose)
                    o3d_objs.append(box)
                    self.robot_link_name_to_open3d_geoms_transform[link][geom_i] = link_pose@geom['pose']@o3d_box_pose
                    # TODO: the open3d box is not aligned with the mujoco box. Need to check the pose transformation (mujoco is center, but open3d is corner)
        if show:
            o3d.visualization.draw_geometries(o3d_objs)
        return copy.deepcopy(o3d_objs)

    def set_joint_values(self, joint_values: Union[dict, list, npt.NDArray[np.float32], npt.NDArray[np.float64]]):
        if isinstance(joint_values, dict):
            for joint_name, value in joint_values.items():
                self.mj_data.qpos[self.mj_model.jnt_qposadr[self.mj_model.joint(joint_name).id]] = value
        elif isinstance(joint_values, list) or isinstance(joint_values, np.ndarray):
            for i in range(len(joint_values)):
                self.mj_data.qpos[self.mj_model.jnt_qposadr[self.mj_model.joint(self.robot_joint_names[i]).id]] = joint_values[i]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._update_transform()
        self._update_fcl_transform()
        self._update_joint_values()
        self._update_selected_joint_values()

    def set_selected_joint_values(self, joint_values: Union[dict, list, npt.NDArray[np.float32], npt.NDArray[np.float64]]):
        if isinstance(joint_values, dict):
            # TODO: check that the joint names should be within self.selected_joint_names
            for joint_name, value in joint_values.items():
                self.mj_data.qpos[self.mj_model.jnt_qposadr[self.mj_model.joint(joint_name).id]] = value
        elif isinstance(joint_values, list) or isinstance(joint_values, np.ndarray):
            for i in range(len(joint_values)):
                self.mj_data.qpos[self.mj_model.jnt_qposadr[self.mj_model.joint(self.selected_joint_names[i]).id]] = joint_values[i]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._update_transform()
        self._update_fcl_transform()
        self._update_joint_values()
        self._update_selected_joint_values()

    def get_link_pose(self, link_name: str) -> np.ndarray:
        """
        get the pose of the link
        """
        return self.robot_link_name_to_transform[link_name]

    def get_link_spatial_jacobian(self, link_name: str) -> np.ndarray:
        """
        shape: 6xnv
        J = [[dg/dtheta_i g^{-1}]^V]
        this is the textbook implementation of the spatial Jacobian.
        ref: a mathematical introduction to robotic manipulation, page 115
        given mujoco spatial Jacobian, assume that g^{-1} = [R', p'; 0, 1]
        then there is:
        J[:,i] = [dg/dtheta_i g^{-1}]^V
        hat(J[:,i]) = [dR/dtheta_i, dp/dtheta_i; 0,0] g^{-1}
                    = [dR/dtheta_i*R', dR/dtheta_i*p' + dp/dtheta_i; 0,0]
        using J_mj, there is:
        hat(J[:,i]) = [hat(J_mj[3:,i]),hat(J_mj[3:,i])*R*p'+J_mj[:3,i];0,0]
        """
        bid = self.mj_model.body(link_name).id
        jacp_mj = np.zeros((3, self.mj_model.nv))
        jacr_mj = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacBody(self.mj_model, self.mj_data, jacp_mj, jacr_mj, bid)
        jacp_mj = jacp_mj[:,self.selected_joint_dofids]
        jacr_mj = jacr_mj[:,self.selected_joint_dofids]
        # obtain the textbook spatial Jacobian
        jacr = np.array(jacr_mj)
        pose = self.get_link_pose(link_name)
        pose_inv = np.linalg.inv(pose)
        jacp = so3.hat(jacr.T)@pose[:3,:3]@pose_inv[:3,3] + jacp_mj.T  # nvx3
        jacp = jacp.T
        return np.vstack((jacp, jacr))

    def get_link_spatial_jacobian_mj(self, link_name: str) -> np.ndarray:
        """
        shape: 6xnv
        NOTE: the Mujoco spatial Jacobian seems to be different from the textbook a math intro to robotic manipulations.
        currently checking with the textbook.
        The mujoco Jacobian hat(J_mj[:,i]) = [J_mj_r[:,:,i],J_mj_p[:,i];0,0] is as follows:
        J_mj_p = dp/dq  (the same as analytical Jacobian translation part)
        J_mj_r[:,:,i] = dR/dq_i R^{-1} (the same as the textbook spatial Jacobian rotation part)
        (above subject to transpose since J_mj is of shape 6xnv)
        Since [v,w]_hat = [w_hat, v; 0, 0], under mujoco spatial Jacobian, J_mj[:,:3] = J_mj_p
        """
        bid = self.mj_model.body(link_name).id
        jacp = np.zeros((3, self.mj_model.nv))
        jacr = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacBody(self.mj_model, self.mj_data, jacp, jacr, bid)
        jacp = jacp[:,self.selected_joint_dofids]
        jacr = jacr[:,self.selected_joint_dofids]
        return np.vstack((jacp, jacr))
    
    def get_link_analytical_jacobian(self, link_name: str) -> np.ndarray:
        """
        get the analytical Jacobian of the link of shape 4x4xnv, in the form dg/dq
        using mujoco's spatial jacobian, there is J_mj_p = dp/dq, J_mj_r = dR/dq R^{-1}
        hence the analytical Jacobian is:
        J_analytical[:3,:3,i] = J_mj_r[:,:,i] * R
        J_analytical[:3,3,i] = J_mj_p[:,i]
        NOTE: this is because Mujoco does not calculate the jacobian according to the textbook
        """
        bid = self.mj_model.body(link_name).id
        jacp_mj = np.zeros((3, self.mj_model.nv))
        jacr_mj = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacBody(self.mj_model, self.mj_data, jacp_mj, jacr_mj, bid)
        jacp_mj = jacp_mj[:,self.selected_joint_dofids]
        jacr_mj = jacr_mj[:,self.selected_joint_dofids]
        jacr = jacr_mj
        pose = self.get_link_pose(link_name)
        jac = np.zeros((4,4,len(self.selected_joint_dofids)))
        jacr = so3.hat(jacr.T)@pose[:3,:3]  # nvx3x3
        jacr = np.transpose(jacr, (1,2,0))  # 3x3xnv
        jac[:3,:3,:] = jacr
        jac[:3,3,:] = jacp_mj
        return jac

    def get_point_on_link_spatial_jacobian(self, link_name: str, point: np.ndarray) -> np.ndarray:
        """
        get the spatial Jacobian of a point on the link
        the point is in the local frame in the link
        computation:
        d (g(q)*p) / dq_i = d(g(q)) / dq_i * p
        TODO: related to the spatial Jacobian computation.
        """
        # compute the analytical jacobian
        jac = self.get_link_analytical_jacobian(link_name)  # 4x4xnv
        # get the point in the link frame
        jac = np.transpose(jac, (2,0,1))  # nvx4x4
        jac = np.tensordot(jac[:,:3,:3], point, axes=1) + jac[:,:3,3] # nvx3
        jac = jac.T
        return jac

    def compute_distance_total(self, dist_margin: float = 0.001, full: bool = False) -> list:
        """
        compute the distance between all the collision pairs.
        NOTE: this is much slower than the compute_collision_total
        """
        # TODO: add early stopping

        distance_results = []  # store the results of the collision, link1, link2, geom1_i, geom2_i, collision result
        for i in range(len(self.collision_pairs)):
            link_1, link_2 = self.collision_pairs[i]
            for obj1_i in range(len(self.robot_link_name_to_fcl_objs[link_1])):
                for obj2_i in range(len(self.robot_link_name_to_fcl_objs[link_2])):
                    dis_result = hppfcl.DistanceResult()
                    self.collision_pair_to_distance_req[(i, obj1_i, obj2_i)].enable_nearest_points = True
                    distance = hppfcl.distance(self.robot_link_name_to_fcl_objs[link_1][obj1_i],
                                               self.robot_link_name_to_fcl_objs[link_2][obj2_i],
                                               self.collision_pair_to_distance_req[(i, obj1_i, obj2_i)],
                                               dis_result)
                                            #    self.collision_pair_to_distance_res[(i, obj1_i, obj2_i)])
                    if distance < dist_margin:
                        distance_result = dis_result
                        # distance_result = self.collision_pair_to_distance_res[(i, obj1_i, obj2_i)]
                        # distance_result = copy.deepcopy(distance_result)
                        distance_results.append((link_1, link_2, obj1_i, obj2_i, distance_result))
                    else:
                        if full:
                            distance_result = dis_result
                            # distance_result = self.collision_pair_to_distance_res[(i, obj1_i, obj2_i)]
                            # distance_result = copy.deepcopy(distance_result)
                            distance_results.append((link_1, link_2, obj1_i, obj2_i, distance_result))
        return distance_results


    def compute_collision_total(self, dist_upper_bound: float = 0.001, security_margin: float = 0.001, full: bool = False) -> list:
        """
        compute the collision between all the collision pairs.
        """
        # TODO: add early stopping

        collision_results = []  # store the results of the collision, link1, link2, geom1_i, geom2_i, collision result
        for i in range(len(self.collision_pairs)):
            link_1, link_2 = self.collision_pairs[i]
            for obj1_i in range(len(self.robot_link_name_to_fcl_objs[link_1])):
                for obj2_i in range(len(self.robot_link_name_to_fcl_objs[link_2])):
                    # set the safety margin
                    self.collision_pair_to_collision_req[(i, obj1_i, obj2_i)].distance_upper_bound = dist_upper_bound
                    self.collision_pair_to_collision_req[(i, obj1_i, obj2_i)].security_margin = security_margin
                    col_result = hppfcl.CollisionResult()
                    collision = hppfcl.collide(self.robot_link_name_to_fcl_objs[link_1][obj1_i],
                                               self.robot_link_name_to_fcl_objs[link_2][obj2_i],
                                               self.collision_pair_to_collision_req[(i, obj1_i, obj2_i)],
                                               col_result)
                                            #    self.collision_pair_to_collision_res[(i, obj1_i, obj2_i)])
                    if collision:
                        collision_result = col_result#self.collision_pair_to_collision_res[(i, obj1_i, obj2_i)]
                        # collision_result = copy.deepcopy(collision_result)
                        collision_results.append((link_1, link_2, obj1_i, obj2_i, collision_result))
                    else:
                        if full:
                            collision_result = col_result#self.collision_pair_to_collision_res[(i, obj1_i, obj2_i)]
                            # collision_result = copy.deepcopy(collision_result)
                            collision_results.append((link_1, link_2, obj1_i, obj2_i, collision_result))
        return collision_results

    def compute_collision_min_dist_total(self, dist_upper_bound: float = 0.001, security_margin = 0.001, full: bool = False) -> list:
        """
        for objects not colliding (distance > 0), return the nearest points
        for objects colliding, return the collision results
        # for object distance smaller than upper bound, compute the minimum distance
        """
        distance_results = []  # store the results of the collision, link1, link2, geom1_i, geom2_i, collision result
        for i in range(len(self.collision_pairs)):
            link_1, link_2 = self.collision_pairs[i]
            for obj1_i in range(len(self.robot_link_name_to_fcl_objs[link_1])):
                for obj2_i in range(len(self.robot_link_name_to_fcl_objs[link_2])):
                    # set the safety margin
                    self.collision_pair_to_collision_req[(i, obj1_i, obj2_i)].distance_upper_bound = dist_upper_bound
                    self.collision_pair_to_collision_req[(i, obj1_i, obj2_i)].security_margin = dist_upper_bound
                    # we want the collision to return pairs that are below the distance upper bound,
                    # so we need to set the security margin as the upper bound, otherwise it will not be
                    # treated as in collision
                    # NOTE: the collision checker may not detect nearest points, but only return any pairs that are
                    # below the security margin
                    col_result = hppfcl.CollisionResult()
                    collision = hppfcl.collide(self.robot_link_name_to_fcl_objs[link_1][obj1_i],
                                               self.robot_link_name_to_fcl_objs[link_2][obj2_i],
                                               self.collision_pair_to_collision_req[(i, obj1_i, obj2_i)],
                                               col_result)
                                            #    self.collision_pair_to_collision_res[(i, obj1_i, obj2_i)])
                    if collision:
                        # * compute the min distance
                        self.collision_pair_to_distance_req[(i, obj1_i, obj2_i)].enable_nearest_points = True
                        dis_result = hppfcl.DistanceResult()
                        distance = hppfcl.distance(self.robot_link_name_to_fcl_objs[link_1][obj1_i],
                                               self.robot_link_name_to_fcl_objs[link_2][obj2_i],
                                               self.collision_pair_to_distance_req[(i, obj1_i, obj2_i)],
                                               dis_result)
                        p1 = dis_result.getNearestPoint1()
                        p2 = dis_result.getNearestPoint2()
                        normal = p2 - p1
                        normal = normal / np.linalg.norm(normal)
                        ret_result = {'p1': p1, 'p2': p2, 'normal': normal, 'distance': distance}
                        # collision_result = copy.deepcopy(collision_result)
                        # if distance < security margin, then it means in collision. In this case, we use the collision
                        # checker to get the distance results, since the nearest points do not provide enough information.
                        # the distance result does not return the center point. So we need to do it ourselves.
                        if distance < security_margin:
                            self.collision_pair_to_collision_req[(i, obj1_i, obj2_i)].distance_upper_bound = dist_upper_bound
                            self.collision_pair_to_collision_req[(i, obj1_i, obj2_i)].security_margin = security_margin
                            col_result = hppfcl.CollisionResult()
                            collision = hppfcl.collide(self.robot_link_name_to_fcl_objs[link_1][obj1_i],
                                                    self.robot_link_name_to_fcl_objs[link_2][obj2_i],
                                                    self.collision_pair_to_collision_req[(i, obj1_i, obj2_i)],
                                                    col_result)
                            contact = col_result.getContact(0)
                            normal = contact.normal
                            normal = normal / np.linalg.norm(normal)
                            ret_result['normal'] = normal
                            pos = contact.pos
                            p1 = pos - normal * 0.5 * (-contact.penetration_depth)
                            p2 = pos + normal * 0.5 * (-contact.penetration_depth)                            
                            ret_result['p1'] = p1
                            ret_result['p2'] = p2
                            ret_result['distance'] = -contact.penetration_depth
                        distance_results.append((link_1, link_2, obj1_i, obj2_i, ret_result))
                    else:
                        if full:
                            dis_result = hppfcl.DistanceResult()
                            # dis_result.min_distance = dist_upper_bound + 1e-1
                            ret_result = {'distance': dist_upper_bound + 1e-1}
                            # collision_result = copy.deepcopy(collision_result)
                            distance_results.append((link_1, link_2, obj1_i, obj2_i, ret_result))
        return distance_results



class MotomanRobot(Robot):
    def __init__(self, default_joint_value_dict: dict = None, selected_joint_names: list = None):
        xml_path = "../../xmls/motoman.xml"
        excluded_pairs = []
        exclude_links = ["arm_right_link_6_b", "arm_right_link_7_t", "base", "base_mount", "right_driver", "right_coupler", "right_spring_link", "right_follower", "right_pad"]
        exclude_links += ["left_driver", "left_coupler", "left_spring_link", "left_follower", "left_pad"]
        for i in range(len(exclude_links)):
            for j in range(i+1,len(exclude_links)):
                # a = mj_model.body(exclude_links[i]).id
                # b = mj_model.body(exclude_links[j]).id
                excluded_pairs.append((exclude_links[i],exclude_links[j]))

        Robot.__init__(self, xml_path, default_joint_value_dict, selected_joint_names, excluded_pairs)

