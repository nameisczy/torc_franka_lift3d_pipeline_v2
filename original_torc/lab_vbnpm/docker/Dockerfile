FROM rwthika/ros-cuda:noetic-desktop-full

# Add user
RUN useradd user && \
	echo "user ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/user && \
	chmod 0440 /etc/sudoers.d/user && \
	mkdir -p /home/user && \
	chown user:user /home/user && \
	chsh -s /bin/bash user

RUN echo 'root:root' | chpasswd
RUN echo 'user:user' | chpasswd

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Update ROS key + install dependencies
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt update && apt remove --purge "libcublas*" "cuda*" -y --allow-change-held-packages && \
	rm -rf /usr/local/cuda-11.4 && \
	apt install git wget neovim tree swig libeigen3-dev liborocos-kdl-dev libkdl-parser-dev liburdfdom-dev libnlopt-dev libgraphviz-dev libnlopt-cxx-dev python3 python3-pip ros-noetic-ros-numpy cuda-toolkit-12-8 ros-noetic-industrial-msgs ros-noetic-moveit tmux wmctrl bind9-host libgl1-mesa-glx libgl1-mesa-dri -y && \
	rm -rf /var/lib/apt/lists/*

# Update CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v4.0.4/cmake-4.0.4-linux-x86_64.sh -O /tmp/cmake-install.sh && \
	chmod +x /tmp/cmake-install.sh && \
	mkdir -p /opt/cmake && \
	/tmp/cmake-install.sh --skip-license --prefix=/opt/cmake
ENV PATH=/opt/cmake/bin:$PATH

RUN pip3 install uv

# Install PyTorch
RUN uv pip install --system torch torchvision setuptools==70.3.0

# Install Most requirements
COPY ./src/lab_vbnpm/requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt

# Install Curobo
ENV TORCH_CUDA_ARCH_LIST=8.6
RUN git clone https://github.com/gh_user/curobo_custom_cost /home/user/curobo_custom_cost && \
 	cd /home/user/curobo_custom_cost && \
 	uv pip install --system -e . --no-build-isolation

USER user
COPY --chown=user . /home/user/lab_ws

RUN git config --global --add safe.directory /home/user/lab_ws/src/lab_vbnpm

WORKDIR /home/user/lab_ws
SHELL ["/usr/bin/bash", "-ic"]

RUN source /opt/ros/noetic/setup.bash && \
	rosdep update && \
	rosdep install --from-paths src --ignore-src -r -y && \
	catkin clean && \
	cd src && \
	./ignore_pkgs_for_real.sh && \
	cd .. && \
	catkin build -DCMAKE_POLICY_VERSION_MINIMUM=3.5

WORKDIR /home/user/lab_ws/src/lab_vbnpm

RUN echo 'export PATH="$HOME/.local/bin:/usr/local/cuda:/opt/cmake/bin :$PATH"' >> ~/.bashrc && \
	echo 'source ~/lab_ws/devel/setup.bash' >> ~/.bashrc

CMD ["bash"]
