import numpy as np
import cv2
import os
import glob
from HOG import HOG

path = os.getcwd()

test_dir_path = path+'/data/INRIAPerson/Train/pos'
test_img_path = glob.glob(test_dir_path+'/*.png')

hog = HOG()

img = cv2.imread(test_img_path[0], 2)

print(len(hog.detect(img)))
