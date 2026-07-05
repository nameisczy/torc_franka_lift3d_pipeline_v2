from pprint import pp

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from utils.visual_utils import decode_seg_img_rgb
from lab_vbnpm.msg import ObjectIdsToNames

rospy.init_node('print_object_ids')

bridge = CvBridge()
seg_img = rospy.wait_for_message("/ground_truth/camera0/seg_image", Image)
seg_raw = bridge.imgmsg_to_cv2(seg_img, 'rgb8')
seg = decode_seg_img_rgb(seg_raw)
ids = set(seg.flatten())
print(ids)
print()

obj_ids_to_names = rospy.wait_for_message("/ground_truth/object_ids_to_names", ObjectIdsToNames)
print('obj_ids_to_names: ')
id2name = dict(zip(obj_ids_to_names.obj_ids,obj_ids_to_names.names))
pp(id2name)
print()

obj_names = [id2name[i] for i in ids if i in id2name]
pp(obj_names)
