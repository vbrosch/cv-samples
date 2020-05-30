import sys
from scipy import ndimage
import numpy as np

import cv2 as cv

img = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
img_blurred = cv.blur(img, (9, 9))

d_depth = cv.CV_16S

grad_x = cv.Sobel(img, d_depth, 1, 0)
grad_x = cv.convertScaleAbs(grad_x)

grad_y = cv.Sobel(img, d_depth, 0, 1)
grad_y = cv.convertScaleAbs(grad_y)

sobeled = cv.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

direction = np.arctan2(grad_x, grad_y)
direction = np.degrees(direction)

bins = [30, 90, 135, 180, 225, 270, 315, 360]
dir_bins = np.digitize(direction, bins)

filtered_bins = dir_bins == bins.index(0)

cv.namedWindow('A', cv.WINDOW_NORMAL)
cv.imshow('A', sobeled)
cv.resizeWindow('A', 1000, 1000)
cv.waitKey(0)
cv.destroyAllWindows()
