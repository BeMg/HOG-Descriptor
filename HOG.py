import numpy as np
import cv2
import math

class HOG:
    # 4 * 4 cell -> a block
    block_size = 4
    # 8 * 8 pixel -> a cell
    cell_size = 8
    nbin = 9

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
        
    # For each pixel, we compute the grad to angle, 
    # then collect angle into histogram for each cell. 
    def WeightVote(self, grad):
        h,w,_ = grad.shape
        h = int(h / self.cell_size)
        w = int(w / self.cell_size)
        cell_histogram = np.zeros((h, w, self.nbin))
        dest = 180 / self.nbin
        
        # For each cell
        for i in range(h):
            for j in range(w):
                # 0-180 degree divide to nbin
                histogram = np.zeros(self.nbin)
                # For each pixel in cell
                for k in range(self.cell_size):
                    for l in range(self.cell_size):
                        x = self.cell_size * i + k;
                        y = self.cell_size * j + l;
                        angle = np.arctan2(grad[x][y][1],grad[x][y][0]) / np.pi * 180
                        angle = np.abs(angle)
                        
                        index = int(angle / dest)
                        offset = angle % dest
                        
                        if index == self.nbin or ((index == self.nbin-1) and (offset >= 10)):
                            histogram[self.nbin-1] += 1
                        elif index == 0 and offset < 10:
                            histogram[0] += 1
                        
                        else:
                            if offset >= 10:
                                offset -= 10
                                histogram[index] += (dest-offset)/dest
                                histogram[index+1] += offset/dest
                            else:
                                offset += 10
                                histogram[index] += (dest-offset)/dest
                                histogram[index-1] += offset/dest
                cell_histogram[i][j] = histogram

        return cell_histogram
        
        
    
        