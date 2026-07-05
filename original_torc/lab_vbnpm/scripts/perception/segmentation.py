#!/usr/bin/env python
"""
This code provides functions that deal with fake perceptions in simulations.
"""
import rospy
import cv_bridge
import numpy as np
import transformations as tf
from sensor_msgs.msg import Image

from utils.visual_utils import encode_seg_img_rgb


class DeticSegmentation():

    def __init__(self, subscribe=None):
        self.bridge = cv_bridge.CvBridge()
        if subscribe is not None:
            self.encoded_seg_pub = rospy.Publisher(
                '/segmentation/image_encoded', Image, queue_size=5
            )
            rospy.Subscriber(subscribe, Image, self.publish_encoded_seg_img)

    def segment_img(self, rgb_img, depth_img):
        seg_img_msg = rospy.wait_for_message(
            '/docker/detic_segmentor/segmentation', Image
        )
        seg_img_raw = self.bridge.imgmsg_to_cv2(seg_img_msg, '32SC1')
        seg_img = np.subtract(seg_img_raw, 1)
        seg_img[seg_img == -1] = -2
        #TODO set robot arm segmentation to -1?
        return seg_img

    def publish_encoded_seg_img(self, seg_img_msg):
        seg_img_raw = self.bridge.imgmsg_to_cv2(seg_img_msg, '32SC1')
        seg_img = np.subtract(seg_img_raw, 1)
        seg_img[seg_img == -1] = -2

        print('indicies:', set(seg_img.flatten()))

        seg_img_encoded = encode_seg_img_rgb(seg_img)
        msg = self.bridge.cv2_to_imgmsg(seg_img_encoded, 'rgb8')
        msg.header = seg_img_msg.header
        print(msg.header)
        print()
        msg.header.stamp = rospy.Time.now()
        print(msg.header)
        self.encoded_seg_pub.publish(msg)
        print()
        print()


if __name__ == '__main__':
    rospy.init_node('encode_segmentation_image')
    rospy.sleep(1.0)
    seg_img_encoder = DeticSegmentation('/docker/detic_segmentor/segmentation')
    rospy.spin()
