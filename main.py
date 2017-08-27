import numpy as np
import cv2
import os
import glob
from utils import draw2

W = 40
H = 64

path = os.getcwd()

# test_dir_path = path+'/data/INRIAPerson/train_64x128_H96/pos'
test_dir_path = path+'/data/INRIAPerson/Train/pos'
test_img_path = glob.glob(test_dir_path+'/*.png')

hog = cv2.HOGDescriptor((W, H), (16, 16), (8,8), (8,8), 9)
# hog = cv2.HOGDescriptor()
svm = cv2.ml.SVM_load('./svm_model/upperbody{}x{}.dat'.format(H,W))

supvec = svm.getSupportVectors()

rho, alpha, svidx = svm.getDecisionFunction(0)
resultmat = -np.dot(alpha[0], supvec)
resultmat = np.append(resultmat, rho)
hog.setSVMDetector(resultmat)
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for i in range(20):

    img = cv2.imread(test_img_path[i], 2)

    ori = cv2.imread(test_img_path[i])
    print(img.shape)

    rects = hog.detectMultiScale(img)
    
    print(rects)

    rect = [i for i in rects[0] if len(i) == 4]

    print(rect)
    draw2(ori, rect, (0, 255, 0))

    cv2.imshow('a', ori)
    cv2.waitKey()