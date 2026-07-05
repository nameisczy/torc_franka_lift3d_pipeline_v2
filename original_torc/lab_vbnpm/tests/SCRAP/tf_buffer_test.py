import sys
from fusion import fusion
fusion.TSDFVolume([[0,10],[0,10],[0,10]], voxel_size=1)
sys.exit()

from motion_planner.bio_ik_planner import BioIkPlanner
from rospkg import RosPack
import rospy
import tf2_ros

rospy.init_node('asf')
rp = RosPack()
urdf = rp.get_path('motoman_sda10f_moveit_config')
urdf += '/config/gazebo_motoman_sda10f.urdf'

# planner = BioIkPlanner(
#     urdf,
#     ['motoman_left_ee', 'motoman_right_ee'],
#     ['arm_right', 'arm_left'],
#     is_sim=True,
# )


tf_buffer = tf2_ros.Buffer(rospy.Duration(60))
tf_listen = tf2_ros.TransformListener(tf_buffer)
rospy.sleep(1)

for i in range(10):
    t0 = rospy.Time.now().to_sec()
    camera2world = tf_buffer.lookup_transform(
        'world',
        'd435_color_optical_frame',
        rospy.Time(),
        rospy.Duration(0.1),
    )
    t1 = rospy.Time.now().to_sec()
    print('Time taken to lookup transform: ', t1 - t0)
    print(camera2world.header.stamp.to_sec())
    print(t1)
