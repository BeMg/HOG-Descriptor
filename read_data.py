import numpy as np
import cv2
from HOG import HOG
import os
import glob
import pickle
# from sklearn import svm
# from sklearn.externals import joblib


hog = HOG()
curr_path = os.getcwd()

pos_dir_path = curr_path+'/data/INRIAPerson/train_64x128_H96/pos'
neg_dir_path = curr_path+'/data/INRIAPerson/train_64x128_H96/neg'

pos_img_path = glob.glob(pos_dir_path+'/*.png')
neg_img_path = glob.glob(neg_dir_path+'/*.png')
neg_img_path += glob.glob(neg_dir_path+'/*.jpg')

img = cv2.imread(pos_img_path[1], 2)
h, w = img.shape

train_data = []
train_label = []

for i in range(len(pos_img_path)):
    img = cv2.imread(pos_img_path[i], 2)
    img2 = img[20:int(h/2), int(w/6):int(w*(5/6))]
    vec = hog.compute(img2)
    train_data.append(vec)
    train_label.append(1)
    print("{}: {}".format(i, len(vec)))
    
for i in range(len(neg_img_path)):
    img = cv2.imread(neg_img_path[i], 2)
    img2 = cv2.resize(img, (60, 64))
    vec = hog.compute(img2)
    train_data.append(vec)
    train_label.append(0)
    print("{}: {}".format(i, len(vec)))

with open('Train_data', 'wb') as fp:
    pickle.dump((train_data, train_label), fp)
