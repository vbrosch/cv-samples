import argparse
import os
from typing import List, Union

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def _get_grayscale_image(img: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def _absolute_histogram(img: np.ndarray) -> List[int]:
    height, width = img.shape

    hist = [0] * 256

    for v in np.nditer(img):
        hist[v] += 1

    assert sum(hist) == height * width
    return hist


def _relative_histogram(absolute_hist: List[int]) -> List[float]:
    factor = sum(absolute_hist)
    rel_hist = [1 / factor * bin for bin in absolute_hist]

    assert abs(sum(rel_hist) - 1) < 1e-6

    return rel_hist


def _cumulative_histogram(hist: List[Union[int, float]]) -> List[Union[int, float]]:
    assert len(hist) == 256
    cumulative_hist = [sum(hist[:v]) for v in range(0, 256)]

    return cumulative_hist


def _calculate_histogram_mapping(abs_hist: List[float], m: int, n: int) -> List[int]:
    c = []
    for i in range(256):
        a = 255 / (m * n)
        b = sum(abs_hist[:i + 1])
        c.append(int(a * b))

    return c


def _equalize_histogram(img: np.ndarray, mapping: List[int]) -> np.ndarray:
    img_output = img.copy()

    for (x, y), v in np.ndenumerate(img):
        img_output[x, y] = mapping[v]

    return img_output


def _parse_arguments() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='The input image file.')

    return vars(parser.parse_args())


image_path = _parse_arguments()['file']

if not os.path.exists(image_path):
    raise FileNotFoundError(image_path)

image = cv.imread(image_path)
image_gray = _get_grayscale_image(image)
width, height, _ = image.shape
abs_histogram = _absolute_histogram(image_gray)
rel_histogram = _relative_histogram(abs_histogram)
cum_abs_histogram = _cumulative_histogram(abs_histogram)
cum_rel_histogram = _cumulative_histogram(rel_histogram)

plt.title('Absolute histogram of {}'.format(image_path))
plt.bar(range(len(abs_histogram)), abs_histogram)

plt.show()

plt.title('Relative histogram of {}'.format(image_path))
plt.bar(range(len(rel_histogram)), rel_histogram)
plt.show()

plt.title('Cumulative relative histogram of {}'.format(image_path))
plt.bar(range(len(cum_rel_histogram)), cum_rel_histogram)
plt.show()

mapping_hist = _calculate_histogram_mapping(abs_histogram, width, height)
img_eq = _equalize_histogram(image_gray, mapping_hist)

plt.title('Source {} gray'.format(image_path))
plt.imshow(image_gray, cmap='gray')
plt.show()

plt.title('Equalized {}'.format(image_path))
plt.imshow(img_eq, cmap='gray')
plt.show()

abs_eq_histogram = _absolute_histogram(img_eq)
rel_eq_histogram = _relative_histogram(abs_eq_histogram)
cum_eq_rel_histogram = _cumulative_histogram(rel_eq_histogram)

plt.title('Eq. Cumulative relative histogram of {}'.format(image_path))
plt.bar(range(len(cum_eq_rel_histogram)), cum_eq_rel_histogram)
plt.show()
