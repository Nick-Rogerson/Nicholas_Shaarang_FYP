#!/usr/bin/env python3

#Imports
from cv_bridge import CvBridge, CvBridgeError
import cv2
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image, CompressedImage

mask_topic = "/fcn_object_segmentation/output"

new_topic = "/Sidewalk_mask/image_raw"

br = CvBridge()

class image_masking:

    def __init__(self):
        self.mask = None
        self.pub = self.pub = rospy.Publisher(new_topic, Image, queue_size=1)
        print("remap set up")

    def callback(self, mask_msg):
        cv_mask = br.imgmsg_to_cv2(mask_msg, desired_encoding="passthrough")
        cv_mask_arr = np.array(np.array(cv_mask,np.float32)*255,np.uint8)
        cv_mask_arr = np.where(cv_mask_arr < 150,0, cv_mask_arr)
        #self.mask = cv2_img = br.imgmsg_to_cv2(mask_msg, "mono8")
        if cv_mask_arr is not None:
            #self.mask = cv2.imdecode(cv_mask_arr,cv2.IMREAD_GRAYSCALE)
            overlay_msg = br.cv2_to_imgmsg(cv_mask_arr, encoding="mono8")
            overlay_msg.header = mask_msg.header
            self.pub.publish(overlay_msg)
            print("Image published")

if __name__ == '__main__':
    rospy.init_node('Sidewalk_mask')
    ic = image_masking()

    # Set up your subscriber and define its callback
    rospy.Subscriber(mask_topic, Image, ic.callback)
    # Spin until ctrl + c
    rospy.spin()
