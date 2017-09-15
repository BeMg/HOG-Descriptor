import cv2
import numpy as np
from sklearn import svm


def detect(img, cascade):

    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(10, 10),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def draw(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (y1, x1), (y2, x2), color, 2)
        # cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def draw2(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x1+x2, y1+y2), color, 2)


def face_cascade():
    cascade_xml = []

    cascade_xml.append('./haarcascades/haarcascade_frontalface_alt2.xml')

    cascades = []

    for i, xml in enumerate(cascade_xml):
        cascades.append(cv2.CascadeClassifier(xml))

    return cascades

def DetectFace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascades = face_cascade()
    rects = detect(img, cascades[0])
    return rects

def GetHighWeightRect(rects, weights, num=2):
    select = []
    rects = list(rects)
    weights = list(weights)

    print(rects)
    print(weights)

    num = min(len(weights), num)

    for i in range(num):
        index = weights.index(max(weights))
        weights.remove(max(weights))
        select.append(rects[index])

    return select

def slice_window(img, H, W, padding):
    h, w = img.shape
    heigh = int((h-H)/padding)
    weigh = int((w-W)/padding)
    result = []
    for i in range(heigh):
        for j in range(weigh):
            x = i*padding
            y = j*padding
            result.append((x, y, x+H, y+W))
    
    return result

def DetectMymethod(img, H, W, clf):
    
    rects = []
    hog = cv2.HOGDescriptor((W, H), (16, 16), (8,8), (8,8), 9)

    

    for i in range(3):
        
        new_H = int(H * (2+0.5*i))
        new_W = int(W * (2+0.5*i))
        
        sw = slice_window(img, new_H, new_W, 25)
        all_vec = []

        for (x1, y1, x2, y2) in sw:
            img2 = img[x1:x2, y1:y2]
            img2 = cv2.resize(img2, (W, H))
            vec = hog.compute(img2)
            vec = vec.flatten()
            all_vec.append(vec)

        pred = clf.predict(all_vec)

        for j, val in enumerate(pred):
            if val == 1:
                rects.append(sw[j])
    
    return rects


def computeDelta(a, b, eps = 0.2):
    (x11, y11, x12, y12) = a
    (x21, y21, x22, y22) = b
    return (min(x12-x11, x22-x21) + min(y12-y11, y22-y21))/2 * eps 

def Checkoverlap(a, b):
    delta = computeDelta(a, b)
    for i in range(4):
        if abs(a[i]-b[i]) > delta:
            return False
    return True

def CheckBinA(a, b):
    if b[0] > a[0] and b[1] > a[1] and b[2] < a[2] and b[3] < a[3]:
        return True
    return False


def RectsGroup(rects, eps = 0.2):
    
    mp = [0] * len(rects)
    
    result = []

    for i in range(len(rects)):
        if mp[i] == -1:
            continue

        for j in range(i+1, len(rects)):
            if CheckBinA(rects[i], rects[j]):
                mp[i] = 1
                mp[j] = -1
            elif CheckBinA(rects[j], rects[i]):
                mp[i] = -1
                mp[j] = 1
                break
            elif Checkoverlap(rects[i], rects[j]):
                mp[i] = 1
                mp[j] = -1
        
        if mp[i] == 1:
            result.append(rects[i])
    
    return result