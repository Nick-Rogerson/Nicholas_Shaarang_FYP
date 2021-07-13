#! /usr/bin/python

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2


# Instantiate CvBridge
bridge = CvBridge()

class image_feature:

    def __init__(self):
        self.folder = 'Dataset/'

        self.image_count = input('starting image number: ')

    def image_callback(self, msg):
        #print("Received an image!")

        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        except e:
            print(e)
        else:
            cv2.imshow('Image window',cv2_img)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                img_fname = self.folder + str(self.image_count)+".png"
		        # Save your OpenCV2 image as a jpeg
                cv2.imwrite(img_fname, cv2_img)
                print('Saved image no. '+str(self.image_count))
                self.image_count = self.image_count + 1
            rospy.sleep(1)


if __name__ == '__main__':
    rospy.init_node('image_listener')
    ic = image_feature()
    # Define your image topic
    image_topic = "/camera/color/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, ic.image_callback)
    # Spin until ctrl + c
    rospy.spin()
