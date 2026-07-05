import motoman_config
import sys
sys.path.insert(0,'/usr/local/lib/python3.11/site-packages')


import pinocchio as pin
import hppfcl
from pinocchio.visualize import MeshcatVisualizer
import time
import numpy as np

"""
load robot model and point cloud representation of the scene.
"""
def test_case1():
    motoman = motoman_config.MotomanSDA10F()
    print('urdf_path: ', motoman.robot_urdf)
    print('packge_dirs: ', motoman.package_dirs)


    model, collision_model, visual_model = pin.buildModelsFromUrdf(motoman.robot_urdf, package_dirs=motoman.package_dirs)

    # Start a new MeshCat server and client.
    # Note: the server can also be started separately using the "meshcat-server" command in a terminal:
    # this enables the server to remain active after the current script ends.
    #
    # Option open=True pens the visualizer.
    # Note: the visualizer can also be opened seperately by visiting the provided URL.    

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)

    # Load the robot in the viewer.
    viz.loadViewerModel()

    # Display a robot configuration.
    q0 = pin.neutral(model)
    print('q0: ')
    print(q0)
    time.sleep(1.0)
    viz.display(q0)
    viz.displayVisuals(True)

"""
load the MJCF file
"""
def test_case2(): # this is only supported by 3.1.0
    model, collision_model, visual_model = pin.buildModelsFromMJCF('/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/motoman_ws/src/lab_vbnpm/xmls/shelf.xml')
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    time.sleep(1.0)
    # Load the robot in the viewer.
    viz.loadViewerModel()

    # Display a robot configuration.
    q0 = pin.neutral(model)
    print('q0: ')
    print(q0)
    time.sleep(1.0)
    viz.display(q0)
    viz.displayVisuals(True)


"""
collision checking with point cloud representation of the scene.
"""
def test_case3():
    motoman = motoman_config.MotomanSDA10F()
    print('urdf_path: ', motoman.robot_urdf)
    print('packge_dirs: ', motoman.package_dirs)

    model, collision_model, visual_model = pin.buildModelsFromUrdf(motoman.robot_urdf, package_dirs=motoman.package_dirs)

    # *** add collisions of point cloud in FCL and then to pinocchio ***
    """
    shelf_bottom:
    - position: [0.85, 0, 0.5]
    - size: 0.175, 0.5, 0.5
    shelf_top:
    - position: [0.85, 0, 1.42]
    - size: 0.175, 0.5, 0.025
    shelf_padding_left:
    - position: [0.85, -0.475, 1.2]
    - size: 0.175, 0.025, 0.2
    shelf_padding_right:
    - position: [0.85, 0.475, 1.2]
    - size: 0.175, 0.025, 0.2
    shelf_padding_back:
    - position: [1.0, 0, 1.2]
    - size: 0.025, 0.5, 0.2
    """
    pcd_total = []
    # shelf-bottom
    num_points = 1000
    position = np.array([0.85, 0, 0.5])
    half_size = np.array([0.175, 0.5, 0.5])
    pcd = np.random.uniform(low=position-half_size, high=position+half_size, size=(num_points, 3))
    pcd_total.append(pcd)
    # shelf-top
    num_points = 1000
    position = np.array([0.85, 0, 1.42])
    half_size = np.array([0.175, 0.5, 0.025])
    pcd = np.random.uniform(low=position-half_size, high=position+half_size, size=(num_points, 3))
    pcd_total.append(pcd)
    # shelf-padding-left
    num_points = 1000
    position = np.array([0.85, -0.475, 1.2])
    half_size = np.array([0.175, 0.025, 0.2])
    pcd = np.random.uniform(low=position-half_size, high=position+half_size, size=(num_points, 3))
    pcd_total.append(pcd)
    # shelf-padding-right
    num_points = 1000
    position = np.array([0.85, 0.475, 1.2])
    half_size = np.array([0.175, 0.025, 0.2])
    pcd = np.random.uniform(low=position-half_size, high=position+half_size, size=(num_points, 3))
    pcd_total.append(pcd)
    # shelf-padding-back
    num_points = 1000
    position = np.array([1.0, 0, 1.2])
    half_size = np.array([0.025, 0.5, 0.2])
    pcd = np.random.uniform(low=position-half_size, high=position+half_size, size=(num_points, 3))
    pcd_total.append(pcd)
    pcd_total = np.concatenate(pcd_total, axis=0)


    # add pcd to fcl
    fcl_octree = hppfcl.makeOctree(pcd_total, 0.01)
    # add fcl_pcd to pinocchio
    octree_obj = pin.GeometryObject('octree', 0, fcl_octree, pin.SE3.Identity())
    octree_obj.meshColor[0] = 1.0
    collision_model.addGeometryObject(octree_obj)
    # visual_model.addGeometryObject(octree_obj)

    # add point cloud for visualization
    point_cloud = hppfcl.BVHModelOBBRSS()
    point_cloud.beginModel(0, len(pcd_total))
    point_cloud.addVertices(pcd_total)
    bvh_obj = pin.GeometryObject('bvh', 0, point_cloud, pin.SE3.Identity())
    bvh_obj.meshColor[0] = 1.0
    visual_model.addGeometryObject(bvh_obj)
 
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)

    # Load the robot in the viewer.
    viz.loadViewerModel()

    # Display a robot configuration.
    q0 = pin.neutral(model)
    print('q0: ')
    print(q0)
    time.sleep(1.0)
    viz.display(q0)
    # viz.displayVisuals(True)

    # Add collisition pairs
    collision_model.addAllCollisionPairs()
    print("num collision pairs - initial:", len(collision_model.collisionPairs))

    srdf_path = '/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/motoman_ws/src/motoman/motoman_sda10f_moveit_config/config/motoman_sda10f.srdf'
    pin.removeCollisionPairs(model, collision_model, srdf_path)
    print("num collision pairs - after:", len(collision_model.collisionPairs))

    is_collision = False
    data = model.createData()
    collision_data = collision_model.createData()
    while not is_collision:
        q = pin.randomConfiguration(model)

        is_collision = pin.computeCollisions(model, data, collision_model, collision_data, q, True)

    print("Found a configuration in collision:",q)
    viz.display(q)

    for i in range(len(collision_data.collisionResults)):
        if collision_data.collisionResults[i].isCollision():
            print(i)
            break
            
    idx = i
    print(collision_model.geometryObjects[collision_model.collisionPairs[idx].first].name)
    print(collision_model.geometryObjects[collision_model.collisionPairs[idx].second].name)
    print(collision_data.collisionResults[idx].isCollision())
    print(collision_data.collisionResults[idx].numContacts())
    print(collision_data.collisionResults[idx].__dir__())
    contacts = collision_data.collisionResults[idx].getContacts()
    print(contacts[0])
    # print(contacts[0].__dir__())
    print(contacts[0].o1)
    print(contacts[0].o2)
    # print(contacts[0].o1.__dir__())
    # print(contacts[0].b1)
    # print(contacts[0].b2)
    # print(contacts[0].normal)
    # print(contacts[0].pos)
    # print(contacts[0].penetration_depth)
    # distance = collision_data.distanceResults[idx]
    # print(distance)
    # print(distance.__dir__())
    # print(distance.min_distance)
    # print(distance.normal)
    # print(distance.getNearestPoint1())
    # print(distance.getNearestPoint2())



if __name__ == '__main__':
    # test_case1()
    # test_case2()
    test_case3()