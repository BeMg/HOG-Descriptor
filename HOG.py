import numpy as np
import cv2

class HOG:
    # 4 * 4 cell -> a block
    block_size = 4
    # 8 * 8 pixel -> a cell
    cell_size = 8
    
    # Get Gradient to computer the angle
    # Grad2Degree(grad:double) -> degree:double
    def Gra2Degree(self, grad):
        return np.arctan(grad) / math.pi * 180

    # ComputeGradient(img:2D-array)
    # Compute Greadient for
    # From a grayscale image to compute each cell gradient histogram.
    # Expecting to get (wight/cell_size) * (hight/cell_size) number gradient vector.
    def ComputeGradient(self, img):
        # The height and weight in image is reverse in numpy array.
        h,w = img.shape
        img = img.astype(float)
        grad = np.zeros(img.shape, dtype=(float,2))
        for i in range(h-2):
            for j in range(w-2):
                # The first element 
                grad[i][j] = (img[i][j]-img[i][j+2], img[i][j]-img[i+2][j])
        return grad
                
        