from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

VARIABLE = 3


def generate_points(n: int = 10) -> np.ndarray:
    points: np.ndarray = 2.5 * np.random.randn(n, 2) + 3
    # points = np.array([[10, 5], [29, 15], [50, 24]])
    return points.astype(int)


def _get_points(arg: np.ndarray, r: range) -> List[float]:
    m, b = tuple(arg.squeeze())
    return [m * x + b for x in r]


def _get_points_orthogonal(a: float, b: float, d: float, r: range):
    return [(d - a * x) / b for x in r]


def _get_point_total_lq(arg: np.ndarray):
    pass


def _build_input_matrix(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x_out = np.ones((x.shape[0], 2))
    x_out[:, 0] = x[:, 0]

    return x_out


def _perform_svd(matrix_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, vh = np.linalg.svd(matrix_in, full_matrices=False)
    s_diag = np.diag(s)

    matrix = np.dot(np.dot(u, s_diag), vh)

    if not np.allclose(matrix_in, matrix):
        raise Exception('SVD sanity check failed.')

    return u, s_diag, vh


def _fit_with_least_squares_and_pseudo_inverse(points: np.ndarray) -> np.ndarray:
    x = _build_input_matrix(points)

    y = points[:, 1]

    x_t = np.transpose(x)

    pseudo_inverse = np.linalg.inv(np.dot(x_t, x))
    x_t_y = np.dot(x_t, y)

    return np.dot(pseudo_inverse, x_t_y)


def _fit_with_least_squares_and_svd(points: np.ndarray) -> np.ndarray:
    x = _build_input_matrix(points)
    y = points[:, 1]

    u, s, vh = _perform_svd(x)

    s_diag_inv = np.linalg.inv(s)

    y_hat = np.dot(np.transpose(u), y)

    u_hat = np.dot(s_diag_inv, y_hat)

    p = np.dot(vh, u_hat)

    return p


def _fit_with_total_least_squares_and_svd(points: np.ndarray) -> Tuple[float, float, float]:
    x = points.astype(float)
    number_of_points, _ = points.shape

    mean_x, mean_y = np.sum(x[:, 0]) / number_of_points, np.sum(x[:, 1]) / number_of_points

    x[:, 0] = x[:, 0] - mean_x
    x[:, 1] = x[:, 1] - mean_y

    x_t = np.transpose(x)
    x_t_x = np.dot(x_t, x)

    u, s, vh = np.linalg.svd(x_t_x)

    a, b = tuple(vh[-1])

    d = a * mean_x + b * mean_y

    return a, b, d


def main():
    points = generate_points()
    l_s_p = _fit_with_least_squares_and_pseudo_inverse(points)
    l_s_s = _fit_with_least_squares_and_svd(points)
    t_l_s_a, t_l_s_b, t_l_s_d = _fit_with_total_least_squares_and_svd(points)

    x_max = 10
    x_range = range(-3, x_max + 3)

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(points[:, 0], points[:, 1], 'bo')
    axs[0, 0].plot(x_range, _get_points(l_s_p, x_range))
    axs[0, 0].axis(xmin=0, xmax=10, ymin=0, ymax=10)
    axs[0, 0].set_title('Least Squares – Pseudo inverse')

    axs[0, 1].plot(points[:, 0], points[:, 1], 'bo')
    axs[0, 1].plot(x_range, _get_points(l_s_s, x_range))
    axs[0, 1].axis(xmin=0, xmax=10, ymin=0, ymax=10)
    axs[0, 1].set_title('Least Squares – SVD')

    axs[1, 0].plot(points[:, 0], points[:, 1], 'bo')
    axs[1, 0].plot(x_range, _get_points_orthogonal(t_l_s_a, t_l_s_b, t_l_s_d, x_range))
    axs[1, 0].axis(xmin=0, xmax=10, ymin=0, ymax=10)
    axs[1, 0].set_title('Total Least Squares – SVD')

    plt.show()


if __name__ == '__main__':
    main()
