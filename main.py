import numpy as np
import cv2
from HOG import HOG
import os
import glob
from sklearn import svm

hog = HOG()
curr_path = os.getcwd()

pos_dir_path = curr_path+'/data/INRIAPerson/train_64x128_H96/pos'

pos_img_path = glob.glob(pos_dir_path+'/*.png')

img = cv2.imread(pos_img_path[46], 2)

h, w = img.shape
img2 = img[:int(h/2)][:]

print(img2.shape)

vec = hog.compute(img2)

print(vec)
print(len(vec))