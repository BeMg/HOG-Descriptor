import cv2


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

