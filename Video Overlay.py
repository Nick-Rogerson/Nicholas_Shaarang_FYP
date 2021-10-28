#!/usr/bin/env python3

#Imports
import cv2
import numpy as np
import sys

img_fold = "/home/nrogerson/Desktop/Camera_video.avi"

msk_fold = "/home/nrogerson/Desktop/Mask_video.avi"

out_fold = "/home/nrogerson/Desktop/Overlayed_video.avi"

fcc = cv2.VideoWriter_fourcc('M','J','P','G')

capimg = cv2.VideoCapture(img_fold)

capmsk = cv2.VideoCapture(msk_fold)

imgret = False
mskret = False
img_string = "n"
mask_string = "n"

img_start_time= int(input("please enter the starting time for Camera video: "))
mask_start_time = int(input("please enter the starting time for mask video: "))
img_frame_start = img_start_time*15
mask_frame_start = mask_start_time*15

frame_number = 0

out = cv2.VideoWriter(out_fold,fcc,15,(640,480))

if (capimg.isOpened() == False):
    print("Error opening camera video")

if (capmsk.isOpened() == False):
    print("Error opening mask video")

while(capimg.isOpened() and capmsk.isOpened()):
    sys.stdout.write("\rFrame number: %i, Image started = %c, Mask start = %c" % (frame_number, img_string, mask_string))
    sys.stdout.flush()

    if (frame_number >= img_frame_start):
        imgret, imgframe = capimg.read()
        img_string = "y"
    if (frame_number >= mask_frame_start):
        mskret, mskframe = capmsk.read()
        mask_string = "y"

    if (imgret == True and mskret == True):
        dst = cv2.addWeighted(imgframe, 0.7,mskframe,0.3,0)




        out.write(dst)
    else:
        if (frame_number >= img_frame_start and frame_number >= mask_frame_start):
            break
    frame_number = frame_number + 1
print("\nOverlay Complete")

out.release()
capimg.release()
capmsk.release()
