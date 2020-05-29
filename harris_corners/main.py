import math
import sys
from typing import Tuple

from scipy.signal import convolve2d, find_peaks
from scipy.signal.windows import gaussian

import numpy as np
import cv2 as cv

ALPHA = 0.1
THRESHOLD = 120


def _show_image_and_wait_for_key(img: np.ndarray, title: str) -> None:
    cv.imshow('{} – Please press a key to continue'.format(title), img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def _blur(img: np.ndarray) -> np.ndarray:
    return cv.blur(img, (6, 6))


def _get_window_function() -> np.ndarray:
    sigma = 5
    return gaussian(25, sigma).reshape(5, 5)


def _get_derivatives(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_depth = cv.CV_16S

    dx_x = cv.Sobel(img, d_depth, 1, 0)
    dx_x = cv.convertScaleAbs(dx_x)

    dx_y = cv.Sobel(img, d_depth, 0, 1)
    dx_y = cv.convertScaleAbs(dx_y)

    dx_xy = np.clip(dx_x * dx_y, a_min=0, a_max=255)
    return dx_x, dx_y, dx_xy


def _get_second_moment_matrix_for_derivative(derivative: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    get second moment matrix for a specific derivative
    :param derivative: the derivative
    :param w: the window
    :return: numpy array
    """
    convolution = convolve2d(derivative, w)
    norm_convolution = convolution / convolution.max()
    return (norm_convolution * 255).astype(np.uint8)


def _get_second_moment_matrix(dx_x: np.ndarray, dx_y: np.ndarray, dx_xy: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    build the second matrix for each pixel
    :param dx_x: the derivative x (squared)
    :param dx_y: the derivative y (squared)
    :param dx_xy: the derivative x * y
    :param w: the window
    :return: the second matrix
    """
    second_moment_matrix_x = _get_second_moment_matrix_for_derivative(dx_x, w)
    second_moment_matrix_y = _get_second_moment_matrix_for_derivative(dx_y, w)
    second_moment_matrix_xy = _get_second_moment_matrix_for_derivative(dx_xy, w)

    _show_image_and_wait_for_key(second_moment_matrix_x, 'second_moment_matrix_x')
    _show_image_and_wait_for_key(second_moment_matrix_y, 'second_moment_matrix_y')
    _show_image_and_wait_for_key(second_moment_matrix_xy, 'second_moment_matrix_xy')

    img_shape = dx_x.shape

    second_moment_matrix = np.zeros(second_moment_matrix_x.shape + (2, 2))

    for y in range(img_shape[0]):
        for x in range(img_shape[1]):
            second_moment_matrix[y, x] = np.array(
                [[second_moment_matrix_x[y, x], second_moment_matrix_xy[y, x]],
                 [second_moment_matrix_xy[y, x], second_moment_matrix_y[y, x]]])

    return second_moment_matrix


def _get_response_matrix(second_moment_matrix: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """
    get the response matrix
    :param second_moment_matrix: the second matrix
    :return: the response matrix
    """
    response_matrix = np.zeros(img_shape)

    for y in range(img_shape[0]):
        for x in range(img_shape[1]):
            det = np.linalg.det(second_moment_matrix[y, x])
            trace = np.trace(second_moment_matrix[y, x])
            response_matrix[y, x] = det - ALPHA * trace * trace

    return response_matrix


def _apply_threshold(response_matrix: np.ndarray, threshold: int) -> np.ndarray:
    """
    apply a threshold
    :param response_matrix: the response matrix
    :param threshold: the threshold
    :return: the thresholded response matrix
    """
    ret = response_matrix.copy()
    ret[response_matrix < threshold] = 0
    return ret


def _eliminate_non_maxima_points(response_matrix: np.ndarray) -> np.ndarray:
    """
    eliminate all points that are non local maxima
    :param response_matrix: the response matrix
    :return:
    """
    response_signal = response_matrix.flatten()
    peak_ids, _ = find_peaks(response_signal, distance=25)

    height, width = response_matrix.shape
    response_matrix_new = np.zeros(response_matrix.shape)

    for peak in peak_ids:
        y = math.floor(peak / width)
        x = peak - y * width

        response_matrix_new[y, x] = min(max(response_matrix[y, x], 0), 255)

    return response_matrix_new


def main():
    img_p = sys.argv[1]

    if img_p is None or img_p == '':
        raise Exception('Img argument must be given.')

    img = cv.imread(img_p)

    if img is None:
        raise Exception('Image could not be loaded.')

    img_grayscale = _to_grayscale(img)
    img_grayscale = _blur(img_grayscale)

    _show_image_and_wait_for_key(img_grayscale, 'Grayscale')

    dx_x, dx_y, dx_xy = _get_derivatives(img_grayscale)

    dx_x_squared = dx_x * dx_x
    dx_y_squared = dx_y * dx_y

    _show_image_and_wait_for_key(dx_x, 'dx_x')
    _show_image_and_wait_for_key(dx_x_squared, 'dx_x_squared')
    _show_image_and_wait_for_key(dx_y, 'dx_y')
    _show_image_and_wait_for_key(dx_y_squared, 'dx_y')
    _show_image_and_wait_for_key(dx_xy, 'dx_xy')

    window_fn = _get_window_function()
    second_moment_matrix = _get_second_moment_matrix(dx_x_squared, dx_y_squared, dx_xy, window_fn)
    response_matrix = _get_response_matrix(second_moment_matrix, img_grayscale.shape)

    _show_image_and_wait_for_key(response_matrix, 'Response Matrix')

    response_matrix = _apply_threshold(response_matrix, THRESHOLD)

    _show_image_and_wait_for_key(response_matrix, 'Response Matrix (Thresholded)')

    response_matrix = _eliminate_non_maxima_points(response_matrix)
    _show_image_and_wait_for_key(response_matrix, 'Response Matrix (Eliminated Non Maximum Points)')


if __name__ == '__main__':
    main()
