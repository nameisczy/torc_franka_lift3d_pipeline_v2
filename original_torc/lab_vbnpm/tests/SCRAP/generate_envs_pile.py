"""
generate the pile of objects in the workspace
"""

import mujoco
import numpy as np
import sys
import os
import transformations as tf
from mjcf_robot_scene_integrator import *

# def sample_obj_poses(mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
def sample_obj_poses(filename):
    mj_model = mujoco.MjModel.from_xml_path(filename)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    n_objs = mj_model.nbody
    obj_poses = []
    # min: pos - half_size
    workspace_x_min = 1.03#1.1 + 0.0 - 0.25 + 0.04 + 0.05
    workspace_y_min = -0.15# 0.0 + 0.0 - 0.58 + 0.19 + 0.1
    workspace_x_max = 1.17#1.1 + 0.0 + 0.25 - 0.18 - 0.05
    workspace_y_max = 0.15#0.0 + 0.0 + 0.58 - 0.19 - 0.1
    workspace_z_min = 1.1

    geom_ids = []
    obj_poses = []
    i = 0
    # for i in range(n_objs):
    while i < n_objs:
        # if the object is workspace, ignore
        if mj_model.body(i).name == 'workspace':
            i += 1
            continue
        if mj_model.body(i).name == 'world':
            i += 1
            continue
        print("body name: ", mj_model.body(i).name)
        geom_id = mj_model.body_geomadr[i]
        jnt_id = mj_model.body_jntadr[i]
        jnt_qposadr = mj_model.jnt_qposadr[jnt_id]
        # obtain the joint id of the object
        obj_position = np.zeros((3))
        obj_position[:2] = np.random.uniform(low=[workspace_x_min, workspace_y_min],
                                         high=[workspace_x_max, workspace_y_max])
        obj_position[2] = workspace_z_min
        obj_orientation = tf.random_rotation_matrix()
        obj_quat = tf.random_quaternion()
        obj_pose = np.array(obj_orientation)
        obj_pose[:3, 3] = obj_position
        # * set the object pose in mujoco data
        mj_data.qpos[jnt_qposadr+0] = obj_position[0]
        mj_data.qpos[jnt_qposadr+1] = obj_position[1]
        mj_data.qpos[jnt_qposadr+2] = obj_position[2]
        mj_data.qpos[jnt_qposadr+3] = obj_quat[0]
        mj_data.qpos[jnt_qposadr+4] = obj_quat[1]
        mj_data.qpos[jnt_qposadr+5] = obj_quat[2]
        mj_data.qpos[jnt_qposadr+6] = obj_quat[3]
        viewer = mujoco_viewer.MujocoViewer(mj_model, mj_data)
        print('viwer alive? ', viewer.is_alive)
        while True:
            mujoco.mj_step(mj_model, mj_data)
            if viewer.is_alive:
                viewer.render()
            else:
                break
        viewer.close()
        # * after verification, append to geom_ids
        geom_ids.append(geom_id)
        obj_poses.append(obj_pose)
        i += 1

    # check the geom_ids
    print("geom_ids: ", geom_ids)
    for geom_id in geom_ids:
        print("geom name: ", mj_model.geom(geom_id).name)

    # * store the body poses into model poses
    for i in range(mj_model.nbody):
        if mj_model.body(i).name == 'workspace':
            continue
        mj_model.body(i).pos = mj_data.body(i).xpos
        mj_model.body(i).quat = mj_data.body(i).xquat
    viewer = mujoco_viewer.MujocoViewer(mj_model, mj_data)
    while True:
        if viewer.is_alive:
            viewer.render()
        else:
            break
    viewer.close()
    return mj_model

def generate_envs():
    # * load the xml file
    # * define the workspace to sample in
    # * randomly sample each object's position and orientation (orientation: around z axis)
    # avoid collision with each other
    # * stablize the scene

    ## get args ##
    robot_xml = sys.argv[1].strip() if len(sys.argv) > 1 else None
    if robot_xml is None:
        print('Please specify robot xml file.', file=sys.stderr)
        sys.exit(-1)

    scene_desc = sys.argv[2].strip() if len(sys.argv) > 2 else None
    if scene_desc is None:
        print('Please specify json, pkl, or xml file.', file=sys.stderr)
        sys.exit(-1)

    out_filename = sys.argv[3].strip() if len(sys.argv) > 3 else None
    if out_filename is None:
        print(
            'Please specify output filename for generated scene.',
            file=sys.stderr
        )
        sys.exit(-1)

    gui = sys.argv[4][0] in ('t', 'T', 'y', 'Y') if len(sys.argv) > 4 else False

    robot_dir = os.path.abspath('/'.join(robot_xml.split('/')[:-1]))
    scene_dir_rel = '/'.join(scene_desc.split('/')[:-1])
    scene_dir = os.path.abspath(scene_dir_rel)
    out_dir = os.path.abspath('/'.join(out_filename.split('/')[:-1]))
    tmp_name = out_dir + '/tmp.mjcf'

    ## load model ##
    if scene_desc.split('.')[-1] == 'pkl':
        with open(scene_desc, 'rb') as f:
            data = pickle.load(f)
            scene_f = data[0]
            obj_poses = data[1]
            obj_pcds = data[2]
            obj_shapes = data[3]
            obj_sizes = data[4]
            # scene_f, obj_poses, obj_pcds, obj_shapes, obj_sizes, target_pose, target_pcd, target_obj_shape, target_obj_size = data
        world_model = load_problem(scene_f, obj_poses, obj_shapes, obj_sizes)
    elif scene_desc.split('.')[-1] == 'json':
        world_model = load_problem(scene_desc, [], [], [])
    elif scene_desc.split('.')[-1] == 'xml':
        root = ET.parse(scene_desc).getroot()
        for child in root:
            for include in child.findall('include'):
                child.remove(include)
        updateAssetDirs(root, scene_dir, out_dir)
        world_model = ET.tostring(root).decode()
    else:
        # TODO generate random scene
        pass

    ## fix scene xml string ###
    if type(world_model) is not str:
        scene_xml_str = world_model.to_xml_string()
        # scene_xml_str = re.sub('-[a-f0-9]+\.', '.', scene_xml_str)
        scene_xml_str = re.sub(
            '\/\/|<default class="\/".+|class="\/"', '', scene_xml_str
        )
        scene_xml_str = re.sub(
            '<default class="main".+|class="main"', '', scene_xml_str
        )
    else:
        scene_xml_str = world_model

    ## copy robot.xml to out_filename directory ##
    scene_root = ET.fromstring(scene_xml_str)
    robot_root = ET.parse(robot_xml).getroot()
    comp_scene = scene_root.find('compiler')
    if comp_scene is not None:
        mesh_dir = comp_scene.get(
            'meshdir', comp_scene.get('assetdir', scene_dir_rel)
        )
        text_dir = comp_scene.get(
            'texturedir', comp_scene.get('assetdir', scene_dir_rel)
        )
    else:
        mesh_dir = scene_dir_rel
        text_dir = scene_dir_rel
    print(mesh_dir)
    print(text_dir)
    print(robot_dir)
    print(out_dir)
    updateAssets(robot_root, mesh_dir, text_dir, robot_dir, out_dir)
    robot_fname = robot_root.get(
        'model', f'temp_robot_{scene_desc.split("/")[-1]}'
    )
    robot_fname = out_dir + '/' + robot_fname + '.xml'
    with open(robot_fname, 'wb') as f:
        f.write(ET.tostring(robot_root))

    ## write unstabilized model to xml ##
    with open(tmp_name, 'w') as f:
        f.write(scene_xml_str)

    ## randomly generate object poses for the temp xml file ##
    model = sample_obj_poses(tmp_name)
    mujoco.mj_saveLastXML(tmp_name, model)

    ## stabilize model ##
    model = stabilize(tmp_name, gui)

    ## save file ##
    mujoco.mj_saveLastXML(tmp_name, model)

    ## include robot.xml in temp xml file ##
    rel_path = os.path.relpath(robot_fname, out_dir)
    include = ET.fromstring(f'<include file="{rel_path}"/>')
    scene_root = ET.parse(tmp_name).getroot()
    scene_root.insert(0, include)
    scene_xml_str = ET.tostring(scene_root).decode()
    with open(tmp_name, 'wb') as f:
        f.write(ET.tostring(scene_root))

    ## combine robot and temp xml into out_filename xml ##
    model = mujoco.MjModel.from_xml_path(tmp_name)
    mujoco.mj_saveLastXML(out_filename, model)

    ## clean up files ##
    os.remove(tmp_name)
    os.remove(robot_fname)


if __name__ == "__main__":
    generate_envs()