import cv2
from utils import draw, DetectMymethod2
import pickle

with open('./svm_model/upperbody64x40', 'rb') as fp:
    clf = pickle.load(fp)

W = 40
H = 64

cap = cv2.VideoCapture(0)

while True:
    flag, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    rects = DetectMymethod2(gray, H, W, 3.5, 20,  clf)

    draw(frame, rects, (0,255,0))

    cv2.imshow('a', frame)

    if cv2.waitKey(5) == 27:
        cap.release()
