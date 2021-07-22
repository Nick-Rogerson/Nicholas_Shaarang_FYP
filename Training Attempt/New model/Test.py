## imports
# visualization library
import cv2
from matplotlib import pyplot as plt
# data storing library
import numpy as np
import pandas as pd
# torch libraries
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
# architecture and data split library
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
# augmenation library
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2
# others
import os
import pdb
import time
import warnings
import random
from tqdm import tqdm_notebook as tqdm
import concurrent.futures
# warning print supression
warnings.filterwarnings("ignore")

# *****************to reproduce same results fixing the seed and hash*******************
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


## dataframe
df=pd.read_csv('/home/nrogerson/Documents/Nicholas_Shaarang_FYP/Training Attempt/New model/data.csv')

# location of original and mask image
img_fol='/home/nrogerson/Documents/Nicholas_Shaarang_FYP/data/all/JPEGImages'
mask_fol='/home/nrogerson/Documents/Nicholas_Shaarang_FYP/data/all/SegmentationClassPNG'
ckpt_path='/home/nrogerson/Documents/Nicholas_Shaarang_FYP/Training Attempt/New model/model_office.pth'
base_path='/home/nrogerson/Documents/Nicholas_Shaarang_FYP/Training Attempt/New model/'

# imagenet mean/std will be used as the resnet backbone is trained on imagenet stats
mean, std=(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)

## transforms
# during traning/val phase make a list of transforms to be used.
# input-->"phase",mean,std
# output-->list
def get_transform(phase,mean,std):
    list_trans=[]
    if phase=='train':
        list_trans.extend([HorizontalFlip(p=0.5)])
    list_trans.extend([Normalize(mean=mean,std=std, p=1), ToTensorV2()])  #normalizing the data & then converting to tensors
    list_trans=Compose(list_trans)
    return list_trans

## getting datasets
'''when dataloader request for samples using index it fetches input image and target mask,
apply transformation and returns it'''
class CarDataset(Dataset):
    def __init__(self,df,img_fol,mask_fol,mean,std,phase):
        self.fname=df['img'].values.tolist()
        self.img_fol=img_fol
        self.mask_fol=mask_fol
        self.mean=mean
        self.std=std
        self.phase=phase
        self.trasnform=get_transform(phase,mean,std)
    def __getitem__(self, idx):
        name=self.fname[idx]
        img_name_path=os.path.join(self.img_fol,name)
        mask_name_path=img_name_path.split('.')[0].replace('JPEGImages','SegmentationClassPNG')+'.png'
        img=cv2.imread(img_name_path)
        mask=cv2.imread(mask_name_path,cv2.IMREAD_GRAYSCALE)
        augmentation=self.trasnform(image=img, mask=mask)
        img_aug=augmentation['image']                           #[3,128,128] type:Tensor
        mask_aug=augmentation['mask']                           #[1,128,128] type:Tensor
        return img_aug, mask_aug

    def __len__(self):
        return len(self.fname)

## return datasets
'''divide data into train and val and return the dataloader depending upon train or val phase.'''
def CarDataloader(df,img_fol,mask_fol,mean,std,phase,batch_size,num_workers):
    df_train,df_valid=train_test_split(df, test_size=0.2, random_state=69)
    df = df_train if phase=='train' else df_valid
    for_loader=CarDataset(df, img_fol, mask_fol, mean, std, phase)
    dataloader=DataLoader(for_loader, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return df, dataloader

## mask overlay
def overlay(mask_img, img, plt):
    plt.imshow(img, interpolation='none')
    plt.imshow(mask_img, cmap='jet', interpolation='none', alpha=0.3)

## Inference
df, test_dataloader=CarDataloader(df,img_fol,mask_fol,mean,std,'val',1,4)
filenames = df['img'].values.tolist()

device = torch.device("cuda")
model = smp.Unet("resnet18", encoder_weights=None, classes=1, activation=None)
model.to(device)
model.eval()
state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])

# start prediction
predictions = []
inf_time_arr = []
# fig = plt.figure(figsize=(14,14))
# fig2 = plt.figure(figsize=(14,14))
# fig.suptitle('predicted_mask')
# fig2.suptitle('original_mask')
# for i, batch in enumerate(test_dataloader):
#     if i < 30:
#         x = fig.add_subplot(12,5,i+1)
#         y = fig2.add_subplot(12,5,i+1)
#         images,mask_target = batch
#         start_time = time.time()
#         batch_preds = torch.sigmoid(model(images.to(device)))
#         batch_preds = batch_preds.detach().cpu().numpy()
#         inf_time = time.time() - start_time
#         print("Image no. " + str(i) + " took " + str(inf_time) + "s to infer")
#         x.imshow(np.squeeze(batch_preds),cmap='gray')
#         y.imshow(np.squeeze(mask_target),cmap='gray')
#     else:
#         break
# fig.set_figheight(15)
# fig.set_figwidth(15)
# fig2.set_figheight(15)
# fig2.set_figwidth(15)
# plt.show()
for i, batch in enumerate(test_dataloader):
    if i < 30:
        fig, ax=plt.subplots(2,2,figsize=(15,15))
        fig.suptitle('Footpath Semantic Segmentation Validation',fontsize=22)
        images,mask_target = batch
        start_time = time.time()
        batch_preds = torch.sigmoid(model(images.to(device)))
        batch_preds = batch_preds.detach().cpu().numpy()
        inf_time = time.time() - start_time
        inf_time_arr.append(inf_time)
        image = cv2.imread(os.path.join(img_fol,filenames[i]))
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.array(image)
        plt.figtext(0.4,0.9,"Image " + str(filenames[i]) + " took " + str(round(inf_time,4)) + "s to infer", fontsize=18)
        ax[0,0].imshow(np.squeeze(mask_target),cmap='gray')
        ax[0,0].set_title('Original Mask')
        ax[0,1].imshow(np.squeeze(batch_preds),cmap='gray')
        ax[0,1].set_title('Predicted Mask')
        ax[1,0].imshow(image)
        ax[1,0].set_title('Original Image')
        overlay(np.squeeze(batch_preds),image,ax[1,1])
        ax[1,1].set_title('Prediction overlayed on Original Image')
        fig.savefig(base_path + "Test_out/" + str(i) + ".png")
    else:
        break
print("average inference time: " + str(round(sum(inf_time_arr)/len(inf_time_arr),4)) + "s")
