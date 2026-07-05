import rospy
import actionlib
from robotiq_2f_gripper_msgs.msg import CommandRobotiqGripperAction, CommandRobotiqGripperFeedback, CommandRobotiqGripperGoal


rospy.init_node("gripper_control")
gas_done = False
robotiq_client = actionlib.SimpleActionClient(
    '/command_robotiq_action', CommandRobotiqGripperAction
)
rospy.sleep(2.0)

def robotiq_gripper_control(control):
    # action clients
    # robotiq_client = actionlib.SimpleActionClient(
    #     '/command_robotiq_action', CommandRobotiqGripperAction
    # )

    client = robotiq_client
    client.wait_for_server()
    action_goal = CommandRobotiqGripperGoal()
    action_goal.position = control
    action_goal.force = 25

    gas_done = False

    def feedback_cb(feedback):
        # rospy.loginfo('Receiving Gripper Feedback...')
        pass

    def done_cb(state, result):
        rospy.loginfo(
            f"""Gripper Action Server is Done.
            State: {state}, Result: {result.fault_status}"""
        )
        gas_done = True

    client.send_goal(action_goal, feedback_cb=feedback_cb, done_cb=done_cb)
    print('Gripper Goal is Sent!')

    # rate = rospy.Rate(30)
    # while not gas_done:
    #     rate.sleep()

    result = client.get_result()
    print('Done!', result)

def ee_control(control):
    robotiq_gripper_control(control)

def ee_close():
    resp = ee_control(0.0)

def ee_open():
    resp = ee_control(0.085)

ee_close()
input('Open?')
ee_open()