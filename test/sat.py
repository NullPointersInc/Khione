import numpy as np
import h5py
import scipy.io as sio
import cv2
from tqdm import trange
tags = ['barren_land', 'trees', 'grassland', 'none']

f = sio.loadmat('deepsat-sat4/sat-4-full.mat')
print(f.keys())


for i in trange(1, f['train_x'].shape[-1] + 1):
    cv2.imwrite(
        'deepsat-sat4/train/{}/{}.jpg'.format(tags[np.argmax(f['train_y'][:, i-1])], i), f['train_x'][:, :, :3, i-1])

for i in trange(1, f['test_x'].shape[-1] + 1):
    cv2.imwrite(
        'deepsat-sat4/val/{}/{}.jpg'.format(tags[np.argmax(f['train_y'][:, i-1])], i), f['test_x'][:, :, :3, i-1])
