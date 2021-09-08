#!/usr/bin/env python3
#Imports
import cv_bridge
import cv2
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image
import copy
import rospy

import torch
import segmentation_models_pytorch as smp
# augmenation library
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2


image_topic = "/camera/color/image_raw/compressed"






def get_transform(phase,mean,std):
    list_trans=[]
    if phase=='train':
        list_trans.extend([HorizontalFlip(p=0.5)])
    list_trans.extend([Normalize(mean=mean,std=std, p=1), ToTensorV2()])  #normalizing the data & then converting to tensors
    list_trans=Compose(list_trans)
    return list_trans


class FCNObjectSegmentation:
    def __init__(self):
        self.gpu = 0  # -1 is cpu mode
        self.pub = rospy.Publisher('/fcn_object_segmentation/output', Image, queue_size=1)
        self.pub_proba = rospy.Publisher('fcn_object_segmentation/output/proba_image', Image, queue_size=1)
        #rospy.init_node('fcn_object_segmentation')
        #self.sub_img = rospy.Subscriber(image_topic, Image, self._cb, queue_size=1, buff_size=2**24)
        self.device = torch.device("cuda")
        self.model = smp.Unet("resnet18", encoder_weights=None, classes=1, activation=None)
        self.model.to(self.device)
        self.model.eval()
        self.state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(self.state["state_dict"])
        self.proba_threshold = 0.0

        # imagenet mean/std will be used as the resnet backbone is trained on imagenet stats
        mean, std=(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)
        self.mean = mean
        phase='val'
        self.trasnform=get_transform(phase,mean,std)


    def cb(self,img_msg):
        br = cv_bridge.CvBridge()
        img = br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        #cv2.imshow('Image window', img)
        label, proba_img = self.segment(img)
        #cv2.imshow('Mask window',label)
        label_msg = br.cv2_to_imgmsg(label.astype(np.float32))
        label_msg.header = img_msg.header
        self.pub.publish(label_msg)
        proba_msg = br.cv2_to_imgmsg(proba_img.astype(np.float32))
        proba_msg.header = img_msg.header
        self.pub_proba.publish(proba_msg)

    def segment(self, bgr):
        augmentation=self.trasnform(image=bgr)
        x=augmentation['image']
        #blob = (bgr - self.mean).transpose((2, 0, 1))
        #x_data = np.array([blob], dtype=np.float32)
        #x = torch.from_numpy(x_data).to(self.device)
        x = x.to(self.device)
        #if self.gpu >= 0:
        #    x = x.cuda(self.gpu)
        with torch.no_grad():
            x = x.unsqueeze(0)
            score = self.model(x)
        proba = torch.sigmoid(score)
        #max_proba, label = torch.max(proba, 1)
        # uncertain because the probability is low
        label = copy.deepcopy(proba)
        #label[max_proba < self.proba_threshold] = 0
        # gpu -> cpu
        proba = proba.permute(0, 2, 3, 1).data.cpu().numpy()[0]
        label = np.squeeze(label.data.cpu().numpy())
        #label = torch.sigmoid(score)
        #label = np.squeeze(label.detach().cpu().numpy())
        #proba = bgr
        return label, proba

if __name__ == '__main__':
    rospy.init_node('segmentation')
    ckpt_path=rospy.get_param('~model_file')
    ic = FCNObjectSegmentation()
    image_topic = "/camera/color/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, ic.cb)
    print('Pipeline set up')
    rospy.spin()
