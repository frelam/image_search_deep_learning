import numpy as np
import math
import sys
file='/home/scw4750/frelam_20161027/get_txt/txt/path1.txt'
f2=open(file)
image_files=None
for line in f2:
    if image_files is None:
        a = line.strip()
        image_files = [a]
    else:
        a = line.strip()
        image_files.append(a)
image_array=np.array(image_files)
f = '/home/scw4750/frelam_20161027/get_txt/train_zhuanli.npy'
np.save(f,image_files)
