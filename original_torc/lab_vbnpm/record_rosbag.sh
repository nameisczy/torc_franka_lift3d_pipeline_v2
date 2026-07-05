#!/usr/bin/bash
BAG_FILE_PATH="$1"
# /camera/aligned_depth_to_color/camera_info \
# /camera/aligned_depth_to_color/image_raw \
# /camera/color/image_raw \
# /camera0/aligned_depth_to_color/image_raw \
# /camera0/color/camera_info \
# /camera0/color/image_raw \
# /camera1/aligned_depth_to_color/image_raw \
# /camera1/color/camera_info \
# /camera1/color/image_raw \
# /ground_truth/camera0/seg_image \
# /ground_truth/camera1/seg_image \
# /ground_truth/object_ids_to_names \
# /ground_truth/selected_object_id \
# /ground_truth/selected_object_name \
# /tf_camera/camera0 \
# /tf_camera/camera1 \
rosbag record -O $BAG_FILE_PATH __name:=rosbag_record \
    /clock \
    /plot_grasps \
    /tf \
    /tf_static \
    /visualization_marker \
    /camera0/color/image_raw
    # /joint_states \
    # /sda10f/sda10f_b1_controller/joint_states \
    # /sda10f/sda10f_r1_controller/joint_states \
    # /sda10f/sda10f_r2_controller/joint_states \
    # /ground_truth/object_poses \
    # /rosout \
    # /rosout_agg \
    # /ray_publish \
    # /ray_publish_2 \
    # /ray_publish_2_array \
    # /joint_command \
    # /debug/full_pcd \
    # /debug/realtime_pcd \
    # /debug/surface_pcd \
    # /debug/target_points \
    # /command_robotiq_action/cancel \
    # /command_robotiq_action/feedback \
    # /command_robotiq_action/goal \
    # /command_robotiq_action/result \
    # /command_robotiq_action/status \
    # /joint_states_all \
    # /joint_trajectory_action/cancel \
    # /joint_trajectory_action/feedback \
    # /joint_trajectory_action/goal \
    # /joint_trajectory_action/result \
    # /joint_trajectory_action/status \
    # /move_group/display_planned_path \
