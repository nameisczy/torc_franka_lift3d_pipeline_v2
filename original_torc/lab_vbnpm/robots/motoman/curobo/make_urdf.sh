#!/usr/bin/env bash
xacro sda10f.xacro > motoman.urdf
# mkdir -p meshes
# for x in $(cat motoman_mujoco.urdf | grep mesh | sed 's#        <mesh filename="package:/#../../../motoman#' | sed 's#"/>##' | tail -n+2)
# do
# 	echo $x meshes/$(echo $x | sed 's#../../../motoman/motoman_sda10f_support/meshes/sda10f/collision/#c_#' | sed 's#../../../motoman/motoman_sda10f_support/meshes/sda10f/visual/#v_#')
# 	cp $x meshes/$(echo $x | sed 's#../../../motoman/motoman_sda10f_support/meshes/sda10f/collision/#c_#' | sed 's#../../../motoman/motoman_sda10f_support/meshes/sda10f/visual/#v_#')
# done
sed -i 's#package://motoman_sda10f_support/##' motoman.urdf
sed -i 's#package://robotiq_2f_85_gripper_visualization/meshes/#meshes/robotiq/#' motoman.urdf
