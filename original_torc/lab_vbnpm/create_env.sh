#!/usr/bin/env bash
# sudo apt install libeigen3-dev liborocos-kdl-dev libkdl-parser-dev liburdfdom-dev libnlopt-dev
# RUNNING_SHELL="$(ps -p $$ -o ucmd=)"

eval "$(micromamba shell hook --shell bash)"

micromamba create -n ros_env -c conda-forge -c robostack-staging \
	catkin_tools ros-noetic-desktop ros-noetic-ros-numpy \
	ros-noetic-sensor-filters ros-noetic-robot-body-filter \
	ros-noetic-moveit-commander ros-noetic-moveit-fake-controller-manager \
	ros-noetic-moveit-simple-controller-manager ros-noetic-moveit-planners \
	ros-noetic-moveit-ros-visualization ros-noetic-moveit-setup-assistant \
	ros-noetic-controller-manager ros-noetic-ur-msgs ros-noetic-ros-controllers \
	uv cuda=12.6 cudnn cusparselt nccl nlopt eigen pink "cmake<4"

micromamba activate ros_env

echo "ln -s $CONDA_PREFIX/include/eigen3/Eigen $CONDA_PREFIX/include"
ln -s $CONDA_PREFIX/include/eigen3/Eigen $CONDA_PREFIX/include

uv pip install --system -r requirements.txt --torch-backend cu126
uv pip install -U --no-deps dm_control
uv pip uninstall \
	pin \
	eigenpy \
	cmeel-octomap \
	cmeel-urdfdom \
	nvidia-cublas-cu12 \
	nvidia-cuda-cupti-cu12 \
	nvidia-cuda-nvrtc-cu12 \
	nvidia-cuda-runtime-cu12 \
	nvidia-cudnn-cu12 \
	nvidia-cufft-cu12 \
	nvidia-cufile-cu12 \
	nvidia-curand-cu12 \
	nvidia-cusolver-cu12 \
	nvidia-cusparse-cu12 \
	nvidia-cusparselt-cu12 \
	nvidia-nccl-cu12 \
	nvidia-nvjitlink-cu12 \
	nvidia-nvtx-cu12 \
	protobuf

echo "rm $CONDA_PREFIX/lib/python3.1"
rm $CONDA_PREFIX/lib/python3.1

# crazy fix but it works to fix moveit viz
cd $CONDA_PREFIX/lib
for x in $(ls libopencv*.411)
do
	echo "$CONDA_PREFIX/lib: ln -s ${x::-4} ${x::-1}0"
	ln -s ${x::-4} ${x::-1}0
done

echo 'Navigate to curobo directory and install with the following command:'
echo '**Note! Substitute "8.9" with your desired CUDA architecture if different!**'
echo '**Find your architecture here: https://developer.nvidia.com/cuda-gpus **'
echo
echo 'TORCH_CUDA_ARCH_LIST="8.9" uv pip install -e . --no-build-isolation'
