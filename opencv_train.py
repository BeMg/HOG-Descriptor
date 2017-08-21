import cv2
import numpy as np

with open('Train_data', 'rb') as fp:
    Data, Label = pickle.load(fp)

with open('Test_data', 'rb') as fp:
    Data2, Label2 = pickle.load(fp)
