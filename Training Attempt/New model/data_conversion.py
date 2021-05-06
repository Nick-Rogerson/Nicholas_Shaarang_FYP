PATH = Path('/media/shashank/New Volume/carvana')

# using fastai below lines convert the gif image to pil image.
(PATH/'train_masks_png').mkdir(exist_ok=True)
def convert_img(fn):
    fn = fn.name
    PIL.Image.open(PATH/'train_masks'/fn).save(PATH/'train_masks_png'/f'{fn[:-4]}.png') #opening and saving image
files = list((PATH/'train_masks').iterdir())
with concurrent.futures.ThreadPoolExecutor(8) as e: e.map(convert_img, files)  #uses multi thread for fast conversion

# we convert the high resolution image mask to 128*128 for starting for the masks.
(PATH/'train_masks-128').mkdir(exist_ok=True)
def resize_mask(fn):
    PIL.Image.open(fn).resize((128,128)).save((fn.parent.parent)/'train_masks-128'/fn.name)

files = list((PATH/'train_masks_png').iterdir())
with concurrent.futures.ThreadPoolExecutor(8) as e: e.map(resize_mask, files)

# # # we convert the high resolution input image to 128*128
(PATH/'train-128').mkdir(exist_ok=True)
def resize_img(fn):
    PIL.Image.open(fn).resize((128,128)).save((fn.parent.parent)/'train-128'/fn.name)

files = list((PATH/'train').iterdir())
with concurrent.futures.ThreadPoolExecutor(8) as e: e.map(resize_img, files)
