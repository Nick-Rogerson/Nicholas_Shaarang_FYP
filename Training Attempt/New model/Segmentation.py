#Imports
import cv_bridge
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image

import torch
import segmentation_models_pytorch as smp
# augmenation library
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor


image_topic = "/camera/color/image_raw"






def get_transform(self,phase,mean,std):
    list_trans=[]
    if phase=='train':
        list_trans.extend([HorizontalFlip(p=0.5)])
    list_trans.extend([Normalize(mean=mean,std=std, p=1), ToTensor()])  #normalizing the data & then converting to tensors
    list_trans=Compose(list_trans)
    return list_trans


class FCNObjectSegmentation:
    def __init__(self):
        self.gpu = 0  # -1 is cpu mode
        self.pub = rospy.Publisher('/fcn_object_segmentation/output', Image, queue_size=1)
        self.pub_proba = rospy.Publisher('fcn_object_segmentation/output/proba_image', Image, queue_size=1)
        rospy.init_node('fcn_object_segmentation')
        #self.sub_img = rospy.Subscriber(image_topic, Image, self._cb, queue_size=1, buff_size=2**24)
        self.device = torch.device("cuda")
        self.model = smp.Unet("resnet18", encoder_weights=None, classes=1, activation=None)
        self.model.to(self.device)
        self.model.eval()
        self.state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(self.state["state_dict"])
        # imagenet mean/std will be used as the resnet backbone is trained on imagenet stats
        mean, std=(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)


    def cb(self,img_msg):
        br = cv_bridge.CvBridge()
        img = br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        cv2.imshow('Image window', img)
        label, proba_img = self.segment(img)
        cv2.imshow('Mask window',label)
        label_msg = br.cv2_to_imgmsg(label.astype(np.int32), '32SC1')
        label_msg.header = img_msg.header
        self.pub.publish(label_msg)
        proba_msg = br.cv2_to_imgmsg(proba_img.astype(np.float32))
        proba_msg.header = img_msg.header
        self.pub_proba.publish(proba_msg)

    def segment(self, bgr):
        transform=get_transform
        augmentation=transform(image=bgr)
        x_data=augmentation['image']
        if self.gpu >= 0:
            x_data = x_data.cuda(self.gpu)
        x = torch.autograd.Variable(x_data, volatile=True)
        score = self.model(x)
        proba = torch.nn.functional.softmax(score)
        max_proba, label = torch.max(proba, 1)
        # uncertain because the probability is low
        label[max_proba < self.proba_threshold] = self.bg_label
        # gpu -> cpu
        proba = proba.permute(0, 2, 3, 1).data.cpu().numpy()[0]
        label = label.data.cpu().numpy().squeeze((0, 1))
        return label, proba

if __name__ == '__main__':
    ckpt_path='/home/nrogerson/Documents/Nicholas_Shaarang_FYP/Training Attempt/New model/model_office.pth'
    ic = FCNObjectSegmentation()
    image_topic = "/camera/color/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, ic.cb)
    print('Pipeline set up')
    rospy.spin()
