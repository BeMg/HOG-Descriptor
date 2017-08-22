import numpy as np
import cv2
import os
import glob
from utils import draw

path = os.getcwd()

test_dir_path = path+'/data/INRIAPerson/train_64x128_H96/pos'
test_img_path = glob.glob(test_dir_path+'/*.png')

hog = cv2.HOGDescriptor((64, 64), (16, 16), (8,8), (8,8), 9)
# hog = cv2.HOGDescriptor()
svm = cv2.ml.SVM_load('./svm_model/upperbody2.dat')

supvec = svm.getSupportVectors()

rho, alpha, svidx = svm.getDecisionFunction(0)
resultmat = -np.dot(alpha[0], supvec)
resultmat = np.append(resultmat, rho)
hog.setSVMDetector(resultmat)
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


for i in range(len(test_img_path)):

    img = cv2.imread(test_img_path[i], 2)

    ori = cv2.imread(test_img_path[i])
    print(img.shape)

    print('Start Detect')


    rects = hog.detectMultiScale(img)
    rects = rects[0]

    rect = [i for i in rects if len(i) == 4]
    
    # for i in range(len(rect)):
    #     rect[i][0] += 15
    #     rect[i][2] += 15


    print(rect)
    draw(ori, rect, (0, 255, 0))

    cv2.imshow('a', ori)
    cv2.waitKey()