import numpy as np
import cv2
import os
import glob
import pickle
from sklearn import svm
from sklearn.externals import joblib


hog = cv2.HOGDescriptor((40, 40), (16, 16), (8,8), (8,8), 9)
curr_path = os.getcwd()

pos_dir_path = curr_path+'/data/INRIAPerson/train_64x128_H96/pos'
neg_dir_path = curr_path+'/data/INRIAPerson/train_64x128_H96/neg'

pos_img_path = glob.glob(pos_dir_path+'/*.png')
neg_img_path = glob.glob(neg_dir_path+'/*.png')
neg_img_path += glob.glob(neg_dir_path+'/*.jpg')

# img = cv2.imread(pos_img_path[1], 2)
# train_data = []
# train_label = []

# for i in range(len(pos_img_path)):
#     img = cv2.imread(pos_img_path[i], 2)
#     img2 = img[20:84, 20:76]
#     vec = hog.compute(img2)
#     train_data.append(vec.flatten())
#     train_label.append(1)
#     print("{}: {}".format(i, len(vec)))

# with open('Train_data', 'wb') as fp:
#     pickle.dump((train_data, train_label), fp)

train_data = []
train_label = []

Test_data = []
Test_label = []

padding = 10

for i in range(len(neg_img_path)):
    img = cv2.imread(neg_img_path[i], 2)
    h2, w2 = img.shape
    h2, w2 = int((h2-40)/10), int((w2-40)/10)
    for j in range(h2):
        for k in range(w2):
            x = j*10
            y = k*10
            img2 = img[x:x+40, y:y+40]
            vec = hog.compute(img2)
            Test_data.append(vec.flatten())
            Test_label.append(0)
            print("{}.{}.{}: {}".format(i, j, k, len(vec)))

with open('Test_data', 'wb') as fp:
    pickle.dump((Test_data, Test_label), fp)
