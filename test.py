import numpy as np
from sklearn import svm
from sklearn.externals import joblib
import pickle

print("Load Train_data")
with open('train_data', 'rb') as fp:
    train_data, train_label = pickle.load(fp)

print("Load SVM model")
clf = joblib.load('./svm_model/upperbody.pkl')

cnt = 0

print("Start Test")

print(len(train_data))
print(len(train_data[1]))

for i in range(len(train_data)):
    pred = clf.predict([train_data[i]])
    print("{}:{}, {}".format(i,pred[0], train_label[i]))
    if pred[0] == train_label[i]:
        cnt += 1

print(cnt)
    