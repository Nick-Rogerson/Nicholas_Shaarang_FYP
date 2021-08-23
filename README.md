# Nicholas_Shaarang_FYP

# Labelme script
run this script in the git folder in terminal:
labelme Dataset --labels labels.txt --nodata --validatelabel exact --config '{shift_auto_shape_color: -2}' --output Labelled_Dataset --autosave

keyboard shortcuts:
d - next image
a - prev image
w - create a ploygon
del - delete selected box
ctrl + z - undo

generate dataset
./labelme2voc.py Dataset data_dataset_voc --labels labels.txt
