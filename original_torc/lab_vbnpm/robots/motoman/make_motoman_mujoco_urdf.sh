#!/usr/bin/env bash
xacro motoman_mujoco.xacro > motoman_mujoco.urdf
mkdir -p meshes
for x in $(cat motoman_mujoco.urdf | grep mesh | sed 's#        <mesh filename="package:/#../../../motoman#' | sed 's#"/>##' | tail -n+2)
do
	echo $x meshes/$(echo $x | sed 's#../../../motoman/motoman_sda10f_support/meshes/sda10f/collision/#c_#' | sed 's#../../../motoman/motoman_sda10f_support/meshes/sda10f/visual/#v_#')
	cp $x meshes/$(echo $x | sed 's#../../../motoman/motoman_sda10f_support/meshes/sda10f/collision/#c_#' | sed 's#../../../motoman/motoman_sda10f_support/meshes/sda10f/visual/#v_#')
done
sed -i 's#package://motoman_sda10f_support/meshes/sda10f/collision/#./meshes/c_#' motoman_mujoco.urdf
sed -i 's#package://motoman_sda10f_support/meshes/sda10f/visual/#./meshes/v_#' motoman_mujoco.urdf
echo "
Make sure to comment out lines 43 as well as 52-59 in motoman_mujoco.urdf
"
