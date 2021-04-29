import os
import shutil
from random import randint

if __name__ == '__main__':

    dir = 'data/'
    a = 'JPEGImages'
    b = 'SegmentationClass'
    c = 'SegmentationClassPNG'
    d = 'SegmentationClassVisualization'
    j = '.jpg'
    p = '.png'
    n = '.npy'

    all_dir = os.path.join(dir, 'all')
    trn_dir = os.path.join(dir, 'train')
    val_dir = os.path.join(dir, 'val')
    tst_dir = os.path.join(dir, 'test')


    f_trn_a = []
    f_trn_b = []
    f_trn_c = []
    f_trn_d = []
    f_val_a = []
    f_val_b = []
    f_val_c = []
    f_val_d = []
    f_tst_a = []
    f_tst_b = []
    f_tst_c = []
    f_tst_d = []


    f_all_a = sorted(os.listdir(os.path.join(all_dir, a)))
    f_all_b = sorted(os.listdir(os.path.join(all_dir, b)))
    f_all_c = sorted(os.listdir(os.path.join(all_dir, c)))
    f_all_d = sorted(os.listdir(os.path.join(all_dir, d)))
    print(f_all_a)

    size = len(f_all_a)
    trn_size = int(round(size * 0.6))
    val_size = int(round(size * 0.2))
    tst_size = int(round(size * 0.2))

    for i in range(0, trn_size):
        index = randint(0,size-i-3)
        print(str(index) + ', '+ str(i) + '. ' +str(len(f_all_a)))
        f_trn_a.append(f_all_a.pop(index))
        f_trn_b.append(f_all_b.pop(index))
        f_trn_c.append(f_all_c.pop(index))
        f_trn_d.append(f_all_d.pop(index))

    for i in range(0,val_size):
        index = randint(0,size-trn_size-i-3)
        print(str(index) + ', '+ str(i) + '. ' +str(len(f_all_a)))
        f_val_a.append(f_all_a.pop(index))
        f_val_b.append(f_all_b.pop(index))
        f_val_c.append(f_all_c.pop(index))
        f_val_d.append(f_all_d.pop(index))

    for i in range(0,tst_size):
        index = randint(0,size-trn_size-val_size-i-1)
        print(str(index) + ', '+ str(i) + '. ' +str(len(f_all_a)))
        f_tst_a.append(f_all_a.pop(index))
        f_tst_b.append(f_all_b.pop(index))
        f_tst_c.append(f_all_c.pop(index))
        f_tst_d.append(f_all_d.pop(index))

    for f1_a in f_trn_a:
        shutil.copy(os.path.join(all_dir, a, f1_a),os.path.join(trn_dir, a))
    for f1_b in f_trn_b:
        shutil.copy(os.path.join(all_dir, b, f1_b),os.path.join(trn_dir, b))
    for f1_c in f_trn_c:
        shutil.copy(os.path.join(all_dir, c, f1_c),os.path.join(trn_dir, c))
    for f1_d in f_trn_d:
        shutil.copy(os.path.join(all_dir, d, f1_d),os.path.join(trn_dir, d))

    for f2_a in f_val_a:
        shutil.copy(os.path.join(all_dir, a, f2_a),os.path.join(val_dir, a))
    for f2_b in f_val_b:
        shutil.copy(os.path.join(all_dir, b, f2_b),os.path.join(val_dir, b))
    for f2_c in f_val_c:
        shutil.copy(os.path.join(all_dir, c, f2_c),os.path.join(val_dir, c))
    for f2_d in f_val_d:
        shutil.copy(os.path.join(all_dir, d, f2_d),os.path.join(val_dir, d))

    for f3_a in f_tst_a:
        shutil.copy(os.path.join(all_dir, a, f3_a),os.path.join(tst_dir, a))
    for f3_b in f_tst_b:
        shutil.copy(os.path.join(all_dir, b, f3_b),os.path.join(tst_dir, b))
    for f3_c in f_tst_c:
        shutil.copy(os.path.join(all_dir, c, f3_c),os.path.join(tst_dir, c))
    for f3_d in f_tst_d:
        shutil.copy(os.path.join(all_dir, d, f3_d),os.path.join(tst_dir, d))
