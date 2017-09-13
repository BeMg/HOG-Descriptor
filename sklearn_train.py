from sklearn import svm
import pickle
import glob
import numpy as np

W = 40
H = 64

with open('POS_data', 'rb') as fp:
    Data, Label = pickle.load(fp)

with open('NEG_data', 'rb') as fp:
    Data2, Label2 = pickle.load(fp)

Data += Data2
Label += Label2

shu = np.random.permutation(len(Data))

shu_data = []
shu_label = []

num = int(len(Data)/8)*7
Test_num = num/7


for i in range(len(shu)):
    shu_data.append(Data[shu[i]])
    shu_label.append(Label[shu[i]])

shu_data = np.array(shu_data, np.float32)
shu_label = np.array(shu_label, np.int32)

Train_data, Test_data = shu_data[:num], shu_data[num:]
Train_label, Test_label = shu_label[:num], shu_label[num:]

print("Train Start")

clf = svm.SVC()
clf.fit(Train_data, Train_label)

with open('upperbody{}x{}'.format(H, W), 'wb') as fp:
    pickle.dump(clf, fp)

test_result = clf.score(Test_data, Test_label)

print(test_result)