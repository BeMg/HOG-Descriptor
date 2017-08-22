import numpy as np
import cv2
import os
import glob
import pickle

with open('Train_data', 'rb') as fp:
    Data, Label = pickle.load(fp)

with open('Test_data', 'rb') as fp:
    Data2, Label2 = pickle.load(fp)

Data += Data2
Label += Label2

shu = np.random.permutation(len(Data))

shu_data = []
shu_label = []

num = int(len(Data)/8)*7


for i in range(len(shu)):
    shu_data.append(Data[shu[i]])
    shu_label.append(Label[shu[i]])

shu_data = np.array(shu_data, np.float32)
shu_label = np.array(shu_label, np.int32)

Train_data, Test_data = shu_data[:num], shu_data[num:]
Train_label, Test_label = shu_label[:num], shu_label[num:]

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

svm.train(Train_data, cv2.ml.ROW_SAMPLE, Train_label)
svm.save('./svm_model/upperbody2.dat')

svm = cv2.ml.SVM_load('./svm_model/upperbody2.dat')

cnt = 0


for i in range(len(Test_data)):
    data = np.array([Test_data[i]], np.float32)
    pred = svm.predict(data)
    _, pred1 = pred
    if pred1[0][0] == Test_label[i]:
        cnt += 1
    else:
        print(Test_label[i])
