"""
test the implementation of planning scene
"""

from robot import MotomanRobot
from planning_scene import PlanningScene
import numpy as np

def test1():
    # test the robot functions
    # sample joint angles until the robot is in collision

    torso_b1 = ["torso_joint_b1"]
    left = [
        "arm_left_joint_1_s",
        "arm_left_joint_2_l",
        "arm_left_joint_3_e",
        "arm_left_joint_4_u",
        "arm_left_joint_5_r",
        "arm_left_joint_6_b",
        "arm_left_joint_7_t",
    ]
    right = [
        "arm_right_joint_1_s",
        "arm_right_joint_2_l",
        "arm_right_joint_3_e",
        "arm_right_joint_4_u",
        "arm_right_joint_5_r",
        "arm_right_joint_6_b",
        "arm_right_joint_7_t",
    ]
    robot_joint_names = torso_b1 + left + right

    robot = MotomanRobot(selected_joint_names=robot_joint_names)

    while True:
        robot_joint_angles = np.random.uniform(robot.selected_joint_limits[:,0], robot.selected_joint_limits[:,1])
        # set the joint angles in mj_data
        robot.set_selected_joint_values(robot_joint_angles)

        collision_results = robot.compute_collision_total()
        if len(collision_results) > 0:
            # print the collisions
            for collision in collision_results:
                link1, link2, obj1_i, obj2_i, collision_result = collision
                print('colllision between ', link1, "and ", link2)
                print(collision_result.__dir__())
                # checking the contact
                print('there are ', collision_result.numContacts(), ' contacts')
                for i in range(collision_result.numContacts()):
                    contact = collision_result.getContact(i)
                    print('contact interface: ')
                    print(contact.__dir__())
            break

    # visualize the robot
    robot.visualize()

def test2():
    """
    test the robot functions. test for the selected joint values
    """
    # test the robot functions
    # sample joint angles until the robot is in collision

    torso_b1 = ["torso_joint_b1"]
    left = [
        "arm_left_joint_1_s",
        "arm_left_joint_2_l",
        "arm_left_joint_3_e",
        "arm_left_joint_4_u",
        "arm_left_joint_5_r",
        "arm_left_joint_6_b",
        "arm_left_joint_7_t",
    ]
    right = [
        "arm_right_joint_1_s",
        "arm_right_joint_2_l",
        "arm_right_joint_3_e",
        "arm_right_joint_4_u",
        "arm_right_joint_5_r",
        "arm_right_joint_6_b",
        "arm_right_joint_7_t",
    ]
    robot_joint_names = right

    robot = MotomanRobot(selected_joint_names=robot_joint_names)

    while True:
        robot_joint_angles = np.random.uniform(robot.selected_joint_limits[:,0], robot.selected_joint_limits[:,1])
        # set the joint angles in mj_data
        robot.set_selected_joint_values(robot_joint_angles)

        collision_results = robot.compute_collision_total()
        if len(collision_results) > 0:
            # print the collisions
            for collision in collision_results:
                link1, link2, obj1_i, obj2_i, collision_result = collision
                print('colllision between ', link1, "and ", link2)
            break

    # visualize the robot
    robot.visualize()

def test3():
    # test the robot with the scene
    # add environment collisions
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


    torso_b1 = ["torso_joint_b1"]
    left = [
        "arm_left_joint_1_s",
        "arm_left_joint_2_l",
        "arm_left_joint_3_e",
        "arm_left_joint_4_u",
        "arm_left_joint_5_r",
        "arm_left_joint_6_b",
        "arm_left_joint_7_t",
    ]
    right = [
        "arm_right_joint_1_s",
        "arm_right_joint_2_l",
        "arm_right_joint_3_e",
        "arm_right_joint_4_u",
        "arm_right_joint_5_r",
        "arm_right_joint_6_b",
        "arm_right_joint_7_t",
    ]
    robot_joint_names = right
    robot = MotomanRobot(selected_joint_names=robot_joint_names)
    # scene = PlanningScene(robot, scene_pcd=pcd_total)
    scene = PlanningScene(robot, scene_pcd=None)
    scene.update_scene_pcd(pcd_total)

    while True:
        robot_joint_angles = np.random.uniform(robot.selected_joint_limits[:,0], robot.selected_joint_limits[:,1])
        # set the joint angles in mj_data
        robot.set_selected_joint_values(robot_joint_angles)

        collision_results = scene.compute_collision_total()
        scene.visualize()
        if len(collision_results) > 0:
            # print the collisions
            for collision in collision_results:
                link1, link2, obj1_i, obj2_i, collision_result = collision
                print('colllision between ', link1, "and ", link2)
            # break
        else:
            print('no collision found.')

    # visualize the robot
    # scene.visualize()


if __name__ == "__main__":
    # test1()
    # test2()
    test3()