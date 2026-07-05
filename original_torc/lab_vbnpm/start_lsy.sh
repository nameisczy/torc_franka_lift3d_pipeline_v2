#!/usr/bin/env bash
gnome-terminal --tab -- bash -ic "source connect_ros_network.sh eno1 moto.cs.lab.edu; roslaunch --wait cgn_ros container.launch"
gnome-terminal --tab -- bash -ic "source connect_ros_network.sh eno1 moto.cs.lab.edu; roslaunch --wait lab_vbnpm perception.launch client_only:=true"
source connect_ros_network.sh eno1 moto.cs.lab.edu

