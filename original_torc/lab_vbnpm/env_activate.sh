eval "$(micromamba shell hook --shell bash)"
micromamba activate ros_env
source /home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/motoman_ws/devel/setup.bash
source connect_ros_network.sh 100.89.246.79 tailscale0
