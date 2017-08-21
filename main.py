import numpy as np
import cv2
import os
import glob
from utils import draw

path = os.getcwd()

test_dir_path = path+'/data/INRIAPerson/train_64x128_H96/pos'
test_img_path = glob.glob(test_dir_path+'/*.png')


img = cv2.imread(test_img_path[55], 2)

ori = cv2.imread(test_img_path[55])
print(img.shape)

print('Start Detect')

hog = cv2.HOGDescriptor((32, 64), (16, 16), (8,8), (8,8), 9)
svm = cv2.ml.SVM_load('./svm_model/upperbody.dat')

vec = svm.getSupportVectors()
vec = vec.flatten()
vec = [[i] for i in vec]
vec = np.array(vec)

hog.setSVMDetector(vec)

rects = hog.detectMultiScale(img)

rect = [i for i in rects if len(i) == 4]

print(rect)
draw(ori, rect, (0, 255, 0))

cv2.imshow('a', ori)
cv2.waitKey()