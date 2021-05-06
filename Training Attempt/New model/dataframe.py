df=pd.read_csv('/home/arun/Shashank/carvana/train_masks.csv')

# location of original and mask image
img_fol='/media/shashank/New Volume/carvana/train-128'
mask_fol='/media/shashank/New Volume/carvana/train_masks-128'

# imagenet mean/std will be used as the resnet backbone is trained on imagenet stats
mean, std=(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)
