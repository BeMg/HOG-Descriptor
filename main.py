import numpy as np
import cv2
import os
import glob
from HOG import HOG
from utils import draw

path = os.getcwd()

test_dir_path = path+'/data/INRIAPerson/Train/pos'
test_img_path = glob.glob(test_dir_path+'/*.png')

hog = HOG()

print(test_img_path[0])

img = cv2.imread(test_img_path[3], 2)

ori = cv2.imread(test_img_path[3])
print(img.shape)

print('Start Detect')
rects = hog.detect(img)

print(len(rects))

draw(ori, rects, (0,255,0) )

cv2.imshow('a', ori)
cv2.waitKey()