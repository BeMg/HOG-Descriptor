import cv2
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
import pickle

with open('pos_data', 'rb') as fp:
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

clf = svm.SVC()

clf.fit(Train_data, Train_label)

joblib.dump(clf, 'face40x40.pkl') 

cnt = 0

for i in range(len(Test_data)):
    pred = clf.predict([Test_data[i]])
    if pred[0] == Test_label[i]:
        cnt += 1
    else:
        print(Test_label[i])

print(cnt)