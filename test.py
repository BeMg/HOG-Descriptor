import cv2
from utils import DetectMymethod, draw
import os
import glob
import pickle

W = 40
H = 64

path = os.getcwd()
test_dir_path = path+'/data/INRIAPerson/Train/pos'
test_img_path = glob.glob(test_dir_path+'/*.png')

for i in range(20):
    print(test_img_path[60+i])
    ori = cv2.imread(test_img_path[60+i])
    img = cv2.imread(test_img_path[60+i], 2)

    with open('./svm_model/upperbody64x40', 'rb') as fp:
        clf = pickle.load(fp)

    rects = DetectMymethod(img, H, W, clf)

    print(rects)

    draw(ori, rects, (0, 255, 0))

    cv2.imshow('a', ori)
    cv2.waitKey()