import numpy as np
import cv2
from HOG import HOG

hog = HOG()

img = cv2.imread('./data/1.jpg', 2)
img = cv2.resize(img, (64+2, 128+2))

grad = hog.ComputeGradient(img)

histogram = hog.WeightVote(grad)

print(histogram)
print(sum(histogram[0][0]))