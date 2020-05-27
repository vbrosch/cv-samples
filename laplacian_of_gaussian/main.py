import math
from pprint import pprint

import numpy as np
from sklearn.preprocessing import normalize


def _l_o_g_entry(x: float, y: float, sigma: float) -> float:
    a = - 1 / (math.pi * pow(sigma, 4))
    b = math.exp(-((pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2))))
    c = 1 - (pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2))

    res = a * b * c

    return res


def main():
    k = 7
    sigma = 1.6
    x_mat = np.zeros((k, k))
    highest_val = 255

    k_low, k_high = - math.floor(k / 2), math.ceil(k / 2)

    for y in range(k_low, k_high):
        for x in range(k_low, k_high):
            x_mat[y + abs(k_low), x + abs(k_low)] = _l_o_g_entry(x, y, sigma)

    print('LOG MAT for sigma = {} and k = {}:'.format(sigma, k))
    pprint(x_mat)

    # norm = normalize(x_mat)
    norm = x_mat / x_mat[0, 0] / highest_val

    print('Normalized mat:')
    pprint(norm)

    scaled = norm * highest_val

    print('Norm scaled with val = {}'.format(highest_val))
    pprint(scaled)

    rounded = np.round(scaled, decimals=0)

    rounded = rounded.astype(int)

    print('Round:')
    pprint(rounded)


if __name__ == '__main__':
    main()
