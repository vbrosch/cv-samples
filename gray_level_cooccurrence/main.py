import math
from typing import Callable

import numpy as np


def _calculate_energy(value: float, idx_1: float, idx_2: float) -> float:
    """
    calculate the energy
    :return:
    """
    return pow(value, 2)


def _calculate_entropy(value: float, idx_1: float, idx_2: float) -> float:
    """
    calculate the entropy
    :return:
    """
    return value * math.log(value, 2) if value > 0 else 0


def _calculate_contrast(value: float, idx_1: float, idx_2: float) -> float:
    """
    calculate the entropy
    :return:
    """
    return pow((idx_1 - idx_2), 2) * value


def _calculate_homogeneity(value: float, idx_1: float, idx_2: float) -> float:
    """
    calculate the homogeneity
    :return:
    """
    return value / (1 + math.fabs(idx_1 - idx_2))


def _get_fn(func_str: str) -> Callable[[float, float, float], float]:
    """
    get the function
    :param func_str:
    :return:
    """
    if func_str == 'Energy':
        return _calculate_energy
    elif func_str == 'Entropy':
        return _calculate_entropy
    elif func_str == 'Contrast':
        return _calculate_contrast
    else:
        return _calculate_homogeneity


def calculate_performance_value(cooccurrence_matrix: np.ndarray, func_str: str) -> None:
    """
    calculate the specified value
    :param cooccurrence_matrix: the matrix
    :return: void
    """
    factor = np.sum(cooccurrence_matrix)
    normalized_cooccurence_matrix = (1 / factor) * cooccurrence_matrix

    # print('Factor is: {}'.format(1 / factor))

    fn = _get_fn(func_str)

    entries = []

    for (x, y), value in np.ndenumerate(normalized_cooccurence_matrix):
        entries.append(fn(value, x, y))

    value = sum(entries)

    print('{}: {:.4f}'.format(func_str, value))


matrix = np.array([[2, 1, 1, 2], [3, 0, 3, 1], [2, 3, 0, 1], [0, 2, 2, 1]])

print(matrix)

calculate_performance_value(matrix, 'Energy')
calculate_performance_value(matrix, 'Entropy')
calculate_performance_value(matrix, 'Contrast')
calculate_performance_value(matrix, 'Homogeneity')
