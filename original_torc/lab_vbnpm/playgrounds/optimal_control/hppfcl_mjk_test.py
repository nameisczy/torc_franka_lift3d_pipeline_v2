# obtain the robot MJCF model
import mujoco
import numpy as np
import transformations as tf
import open3d as o3d

xml_path = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/motoman_ws/src/lab_vbnpm/xmls/motoman.xml"
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)

# print out all the bodies
robot_links = []
for i in range(mj_model.nbody):
    # print(mj_model.body(i).name)
    if mj_model.body(i).name == "world":
        continue
    robot_links.append(mj_model.body(i).name)

# create the dictionary from robot_links to the list of collision meshes of geoms in the link
# for robot links, the collision meshes start from "c_"
# for pad box, they start from "c_", and are of box shape
robot_link_to_geoms = {}
for link in robot_links:
    print('handling link: ', link)
    robot_link_to_geoms[link] = []
    body_idx = mj_model.body(link).id
    print('geomadr: ', mj_model.body_geomadr[body_idx])
    print('geomnum: ', mj_model.body_geomnum[body_idx])
    for geom_idx in range(mj_model.body_geomadr[body_idx], mj_model.body_geomadr[body_idx]+mj_model.body_geomnum[body_idx]):
        if mj_model.geom(geom_idx).name.startswith("c_"):
            print('handling geom: ', mj_model.geom(geom_idx).name)
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
                mesh_pose[3,0] = mj_model.mesh_pos[mesh_idx,0]
                mesh_pose[3,1] = mj_model.mesh_pos[mesh_idx,1]
                mesh_pose[3,2] = mj_model.mesh_pos[mesh_idx,2]


                mesh_scale = [mj_model.mesh_scale[mesh_idx,0],
                              mj_model.mesh_scale[mesh_idx,1],
                              mj_model.mesh_scale[mesh_idx,2]]
                mesh_scale = np.array(mesh_scale)
                # print('mesh_scale: ')
                # print(mesh_scale)
                # print('mesh_pose: ')
                # print(mesh_pose)
                mesh_vertices = mesh_vertices * mesh_scale
                mesh_vertices = mesh_pose[:3,:3].dot(mesh_vertices.T).T + mesh_pose[:3,3]

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
            pose[3,0] = mj_model.geom_pos[geom_idx,0]
            pose[3,1] = mj_model.geom_pos[geom_idx,1]
            pose[3,2] = mj_model.geom_pos[geom_idx,2]
            geom['pose'] = pose
            robot_link_to_geoms[link].append(geom)

            # TODO: visualize the meshes or geoms, potentially in Open3D

# visualize the geoms at configuration
mujoco.mj_forward(mj_model, mj_data)
o3d_objs = []
for link, geoms in robot_link_to_geoms.items():
    for geom in geoms:
        if geom['type'] == 'mesh':
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(geom['vertices'])
            mesh.triangles = o3d.utility.Vector3iVector(geom['faces'])
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([1, 0, 0])
            mesh.transform(geom['pose'])
            o3d_objs.append(mesh)
        elif geom['type'] == 'box':
            box = o3d.geometry.TriangleMesh.create_box(width=geom['size'][0]*2, height=geom['size'][1]*2, depth=geom['size'][2]*2)
            box.compute_vertex_normals()
            box.paint_uniform_color([1, 0, 0])
            box.transform(geom['pose'])
            o3d_objs.append(box)


o3d.visualization.draw_geometries(o3d_objs)
