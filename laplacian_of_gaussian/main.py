import math
import sys
from pprint import pprint
from typing import Callable

import numpy as np
import cv2 as cv
from scipy.signal import convolve2d


def _l_o_g_entry(x: float, y: float, sigma: float) -> float:
    a = - 1 / (math.pi * pow(sigma, 4))
    b = math.exp(-((pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2))))
    c = 1 - (pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2))

    res = a * b * c

    return res


def _d_o_f_entry(x: float, y: float, sigma: float) -> float:
    return 1 / (2 * math.pi * pow(sigma, 2)) * math.exp(-(pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2)))


def _build_filter(name: str, entry_function: Callable[[float, float, float], float], k: int,
                  sigma: float) -> np.ndarray:
    x_mat = np.zeros((k, k))
    highest_val = 255

    k_low, k_high = - math.floor(k / 2), math.ceil(k / 2)

    for y in range(k_low, k_high):
        for x in range(k_low, k_high):
            x_mat[y + abs(k_low), x + abs(k_low)] = entry_function(x, y, sigma)

    print('{} MAT for sigma = {} and k = {}:'.format(name, sigma, k))
    pprint(x_mat)

    norm = x_mat / x_mat[0, 0] / highest_val

    print('{} Normalized mat:'.format(name))
    pprint(norm)

    scaled = norm * highest_val

    print('{} Norm scaled with val = {}'.format(name, highest_val))
    pprint(scaled)

    rounded = np.round(scaled, decimals=0)

    rounded = rounded.astype(int)

    print('{} Round:'.format(name))
    pprint(rounded)

    return rounded


def main():
    k = 7
    sigma = 1.5

    rounded_l_o_g = _build_filter('LOG', _l_o_g_entry, k, sigma)
    rounded_d_o_g = _build_filter('DOG', _d_o_f_entry, k, 2 * sigma) - _build_filter('DOG_2', _d_o_f_entry, k, sigma)

    print('Final DOG')
    pprint(rounded_d_o_g)

    img = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
    img_log = convolve2d(img, rounded_l_o_g)
    img_log = np.clip(img_log, 0, 255).astype(np.uint8)

    img_dog = convolve2d(img, rounded_d_o_g)
    img_dog = np.clip(img_dog, 0, 255).astype(np.uint8)

    cv.namedWindow('LOG', cv.WINDOW_NORMAL)
    cv.imshow('LOG', img_log)
    cv.resizeWindow('LOG', 1000, 1000)

    cv.namedWindow('DOG', cv.WINDOW_NORMAL)
    cv.imshow('DOG', img_log)
    cv.resizeWindow('DOG', 1000, 1000)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
