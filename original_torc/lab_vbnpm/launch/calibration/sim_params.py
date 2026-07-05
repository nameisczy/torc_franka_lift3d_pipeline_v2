import rospy

pose, size = (
    [0.55, 0.65, 1.05],
    [0.50, 1.30, 0.52],
)
rospy.set_param('/workspace/pose', pose)
rospy.set_param('/workspace/size', size)
rospy.set_param('/robot/vel_ang_lim', 60)
rospy.set_param('/robot/acc_ang_lim', 850)

rospy.set_param(
    '/my_eob_calib_eye_on_base/robot_effector_frame', 'torso_link_b1'
)
rospy.set_param('/my_eob_calib_eye_on_base/transformation', False)
rospy.set_param(
    '/my_eob_calib_eye_on_hand/robot_effector_frame', 'arm_right_link_7_t'
)
rospy.set_param('/my_eob_calib_eye_on_hand/transformation', False)
