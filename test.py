import cv2
from slice_window import slice_window
import os
import glob

W = 40
H = 64

path = os.getcwd()
test_dir_path = path+'/data/INRIAPerson/Train/pos'
test_img_path = glob.glob(test_dir_path+'/*.png')

print(test_img_path[10])
img = cv2.imread(test_img_path[10], 2)


for (x1, y1, x2, y2) in slice_window(img, H, W, 10):
    img2 = img[x1:x2, y1:y2]
    print((x1, y1, x2, y2))

