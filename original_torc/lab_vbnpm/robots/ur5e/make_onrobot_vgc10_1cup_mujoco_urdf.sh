#!/usr/bin/env bash
xacro onrobot_vgc10_1cup_mujoco.xacro > onrobot_vgc10_1cup_mujoco.urdf
mkdir -p assets
for x in $(cat onrobot_vgc10_1cup_mujoco.urdf | grep mesh | sed 's#        <mesh filename="package:/#../../../onrobot#' | sed 's#"/>##' | tail -n+2)
do
	echo $x assets/$(echo $x | sed 's#../../../onrobot/onrobot_vg_description/meshes/vgc10/collision/#c_#' | sed 's#../../../onrobot/onrobot_vg_description/meshes/vgc10/visual/#v_#')
	cp $x assets/$(echo $x | sed 's#../../../onrobot/onrobot_vg_description/meshes/vgc10/collision/#c_#' | sed 's#../../../onrobot/onrobot_vg_description/meshes/vgc10/visual/#v_#')
done
sed -i 's#package://onrobot_vg_description/meshes/vgc10/collision/#assets/c_#' onrobot_vgc10_1cup_mujoco.urdf
sed -i 's#package://onrobot_vg_description/meshes/vgc10/visual/#assets/v_#' onrobot_vgc10_1cup_mujoco.urdf
