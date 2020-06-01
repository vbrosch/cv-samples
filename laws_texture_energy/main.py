import math
import sys

from scipy.signal import convolve2d
import cv2 as cv
import numpy as np

l5 = np.array([1, 4, 6, 4, 1]).reshape((1, 5))
e5 = np.array([-1, -2, 0, 2, 1]).reshape((1, 5))
s5 = np.array([-1, 0, 2, 0, -1]).reshape((1, 5))
r5 = np.array([1, -4, 6, 4, 1]).reshape((1, 5))

e5_e5 = np.dot(np.transpose(e5), e5)
s5_s5 = np.dot(np.transpose(s5), s5)
r5_r5 = np.dot(np.transpose(r5), r5)


def _combine_texture(x5, y5):
    x5_y5 = np.dot(np.transpose(x5), y5)
    y5_x5 = np.dot(np.transpose(y5), x5)

    return ((x5_y5 + y5_x5) / 2).astype(int)


l5e5_e5l5 = _combine_texture(l5, e5)
l5s5_s5l5 = _combine_texture(l5, s5)
l5r5_r5l5 = _combine_texture(l5, r5)
e5s5_s5e5 = _combine_texture(e5, s5)
e5r5_r5e5 = _combine_texture(e5, r5)
s5r5_r5s5 = _combine_texture(s5, r5)

masks = [e5_e5, s5_s5, r5_r5, l5e5_e5l5, l5s5_s5l5, l5r5_r5l5, e5s5_s5e5, e5r5_r5e5, s5r5_r5s5]

gray_img = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
# gray_img = cv.blur(gray_img, (48, 48))

sum_mask = np.ones((15, 15))

img_masks = []

average_window = np.ones((15, 15)) * (1 / (15 * 15))

for i, mask in enumerate(masks):
    normalized_img = convolve2d(gray_img, average_window)
    mask_img = np.abs(convolve2d(normalized_img, mask, boundary='symm', mode='same'))
    sum_mask_img = convolve2d(mask_img, sum_mask, boundary='fill', mode='same')
    sum_mask_img *= (255.0 / sum_mask_img.max())
    sum_mask_img = np.around(sum_mask_img).astype(np.uint8)

    print('Mask {}: {}'.format(i + 1, np.sum(sum_mask_img)))
