import math
import sys
from typing import Tuple

import cv2 as cv
import numpy as np

from harris_corners.main import to_grayscale, blur, get_image, show_image_and_wait_for_key

CANNY_THRES_LOW = 30
CANNY_THRES_HIGH = 230

DELTA_RHO = 0.01
VERTICAL_BINS = 10000


def _transform_point_to_pl_space(point: np.ndarray, rho: float) -> float:
    x, y = tuple(point.squeeze())

    return x * math.cos(rho) + y * math.sin(rho)


def _initialize_accumulator() -> np.ndarray:
    rho_steps = math.floor(2 * math.pi / DELTA_RHO)
    return np.zeros((rho_steps, VERTICAL_BINS))


def _fill_accumulator(point: np.ndarray, accumulator: np.ndarray):
    rho_max, l_max = accumulator.shape

    for rho in range(rho_max):
        l = _transform_point_to_pl_space(point, rho * DELTA_RHO)
        l = l + 1

        # TODO


def main():
    img = get_image()
    img_grayscale = to_grayscale(img)
    img_grayscale = blur(img_grayscale)

    edges = cv.Canny(img_grayscale, CANNY_THRES_LOW, CANNY_THRES_HIGH)
    show_image_and_wait_for_key(edges, 'Edges')

    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    img_contours = cv.drawContours(img, contours, -1, (255, 0, 0))

    show_image_and_wait_for_key(img_contours, 'Contours')

    contour_points = [c for contour in contours for c in contour]
    accumulator = _initialize_accumulator()

    for point in contour_points:
        _fill_accumulator(point, accumulator)


if __name__ == '__main__':
    main()
