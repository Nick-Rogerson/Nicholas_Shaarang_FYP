#!/usr/bin/env python3

#Imports
from cv_bridge import CvBridge, CvBridgeError
import cv2
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image, CompressedImage
from matplotlib import pyplot as plt

image_topic = "/camera/color/image_raw/compressed"

mask_topic = "/fcn_object_segmentation/output"

br = CvBridge()

class image_masking:

    def __init__(self):
        self.img = None
        self.mask = None
        self.flags = [False, False]
        self.pub = rospy.Publisher('/mask_overlay', CompressedImage, queue_size=1)
        print(self.img is not None)
        print(self.mask is not None)
        self.fig = plt.figure()

        rospy.Subscriber(image_topic, CompressedImage, self.image_callback)
        rospy.Subscriber(mask_topic, Image, self.mask_callback)

    def image_callback(self, img_msg):
        print(type(img_msg))
        np_arr_img = np.fromstring(img_msg.data, np.uint8)
        #print(np_arr_img)
        #print(type(np_arr_img))
        print(np_arr_img.shape)
        self.img = cv2.imdecode(np_arr_img,cv2.IMREAD_COLOR)
        self.img = cv2.resize(self.img,(640, 480))
        print(self.img.shape)
        print(type(self.img))
        self.img_header = img_msg.header
        self.flags[0] = True
        print("image found")
        if all(self.flags):
            self.publisher()




    def mask_callback(self, mask_msg):
        print(type(mask_msg))
        cv_mask = br.imgmsg_to_cv2(mask_msg, desired_encoding="passthrough")
        cv_mask_arr = np.array(np.array(cv_mask,np.float32)*255,np.uint8)
        #print(cv_mask_arr.shape)
        #cv_mask_arr = cv_mask_arr.flatten()
        #print(cv_mask_arr)
        print(cv_mask_arr.shape)
        self.mask = cv_mask_arr
        #self.mask = cv2.cvtColor(self.mask,cv2.COLOR_GRAY2BGR)
        print(type(self.mask))
        self.flags[1] = True
        print("mask found")
        if all(self.flags):
            self.publisher()

    def publisher(self):
        print("publisher")
        if (all([self.img is not None, self.mask is not None])):
            #cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
            #plt.imshow(self.img, interpolation='none')
            #plt.imshow(self.mask, cmap='hot', interpolation='none', alpha=0.3)
            #self.fig.canvas.draw()
            #overlay = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            #overlay = cv2.resize(overlay,(640, 480))
            #overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            #overlay = cv2.addWeighted(self.img,0.7,self.mask,0.3,0)
            overlay = self.img
            print(self.img)
            self.mask = np.where(self.mask >125,0,1)
            print(overlay[:,:,0].shape)

            overlayb = overlay[:,:,0]
            overlayg = overlay[:,:,1]
            overlayr = overlay[:,:,2]
            print("stuff")
            print(overlayb.shape)
            print(self.mask.shape)
            overlayb[self.mask == 1] = 0
            overlayg[self.mask == 1] = 0
            overlayr[self.mask == 1] = 255

            overlay = np.array([overlayb, overlayg, overlayr])
            overlay = np.moveaxis(overlay,0,-1)
            print(overlay.shape)
            print(type(overlay))
            overlay_msg = br.cv2_to_imgmsg(overlay)
            print(overlay)
            print(type(overlay_msg))
            overlay_msg.header = self.img_header
            self.pub.publish(overlay_msg)
            self.flags = [False, False]
            print("Image published")

    def loop(self):
        rospy.logwarn("Starting Loop...")
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('Masking')
    ic = image_masking()


    ic.loop()
