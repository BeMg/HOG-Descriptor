import numpy as np
import cv2
from HOG import HOG
import os
import glob
import pickle
from sklearn import svm
from sklearn.externals import joblib

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


Train_data, Test_data = shu_data[:num], shu_data[num:]
Train_label, Test_label = shu_label[:num], shu_label[num:]

clf = svm.SVC()

clf.fit(Train_data, Train_label)

cnt = 0
cnt_one = 0;

for i in range(len(Test_data)):
    pred = clf.predict([Test_data[i]])
    print("{} {}".format(Test_label[i],pred[0]))
    if Test_label[i] == pred[0]:
        if Test_label[i] == 1:
            cnt_one += 1
        cnt += 1

print(cnt/len(Test_data))

joblib.dump(clf, './svm_model/upperbody.pkl') 
