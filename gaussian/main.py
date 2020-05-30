import math
import sys
from pprint import pprint
from typing import Callable

import cv2 as cv
import numpy as np


def _entry(x: float, y: float, sigma: float) -> float:
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
    # img = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)

    k = 5
    sigma = 1.2

    f = _build_filter('Gaussian', _entry, k, sigma)

    pass


if __name__ == '__main__':
    main()
