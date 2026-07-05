FILE="$1"
sed -i 's#<camera name="cam_torso" pos="0 0 0" fovy="64"/>##' "$FILE"
sed -i 's#<light pos="-0.5 0.5 3" dir="0 0 -1" directional="true" castshadow="false" diffuse="1 1 1"/>#<light pos="-0.5 0.5 3" dir="0 0 -1" directional="true" castshadow="false" diffuse="1 1 1"/>\n    <camera name="cam_torso" pos="0.1 0 1.38" fovy="64" quat="0.52 0.48 -0.48 -0.52"/>#' "$FILE"
