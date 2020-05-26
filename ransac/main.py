import math
import sys
from pprint import pprint
from typing import Tuple, List

import matplotlib.pyplot as plt

import numpy as np

X_DELTA = 0.5
X_MIN = 0
X_MAX = 25
Y_MIN = 0
Y_MAX = 25

INLIER_COUNT = 100
OUTLIER_COUNT = 20

TERMINATION_THRESHOLD = 1 - (OUTLIER_COUNT / INLIER_COUNT) - 0.2
DISTANCE_THRESHOLD = 1.5

SAMPLE_SIZE = 2

ACCURACY = 0.95


def _random_sample(size: int, a: int, b: int) -> np.ndarray:
    return (b - a) * np.random.sample((size,)) + a


def _get_line_parameters() -> np.ndarray:
    return _random_sample(2, 0, 2)


def _get_inlier_points() -> np.ndarray:
    u = _get_line_parameters()

    #
    print('Creating inliers by sampling from line:')
    pprint(u)

    x = np.ones((INLIER_COUNT, 2))
    x[:, 0] = np.arange(X_MIN, X_MAX, (X_MAX / INLIER_COUNT))

    print('X-Coords:')
    pprint(x)

    y = np.dot(x, u)
    offsets = _random_sample(INLIER_COUNT, -3, 3)
    x[:, 1] = y + offsets

    print('Points:')
    pprint(x)

    return x


def _get_outlier_points() -> np.ndarray:
    outliers = np.zeros((OUTLIER_COUNT, 2))
    outliers[:, 0] = _random_sample(OUTLIER_COUNT, X_MIN, X_MAX)
    outliers[:, 1] = _random_sample(OUTLIER_COUNT, Y_MIN, Y_MAX)

    print('Generated outliers:')
    pprint(outliers)

    return outliers


def _sample_points() -> Tuple[np.ndarray, np.ndarray]:
    inliers = _get_inlier_points()
    outliers = _get_outlier_points()

    return inliers, outliers


def _determine_inliers(params: np.ndarray, points: np.ndarray) -> List[int]:
    x = np.ones((points.shape[0], 2))
    x[:, 0] = points[:, 0]

    y = np.dot(x, params)
    distances = np.abs(y - points[:, 1])

    return list(np.where(distances < DISTANCE_THRESHOLD)[0])


def _perform_sample(points: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    # sample random points
    point_indexes = np.random.randint(points.shape[0], size=SAMPLE_SIZE)
    p = points[point_indexes, :]

    # build linear equation [x, 1] * [m, b] = [y]
    x = np.ones((SAMPLE_SIZE, 2))
    x[:, 0] = p[:, 0]

    b = p[:, 1]

    # solve for [m, b]
    params = np.linalg.solve(x, b)

    inliers = _determine_inliers(params, points)

    return params, inliers


def _perform_ransac(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = sys.maxsize
    i = 0
    e = 1.0

    best_consensus_set = []
    best_params = np.zeros((2,))
    number_of_points = points.shape[0]

    while n > i:
        params, consensus_set = _perform_sample(points)

        if len(best_consensus_set) < len(consensus_set):
            print('New consensus set with {} inliers (old had {})'.format(len(consensus_set), len(best_consensus_set)))
            best_consensus_set = consensus_set
            best_params = params

        amount_of_inliers = len(best_consensus_set) / number_of_points
        if amount_of_inliers > TERMINATION_THRESHOLD:
            print('Inlier amount ({:.2f}%) is above threshold! Terminating.'.format(amount_of_inliers * 100))
            break

        i += 1

        current_e = 1 - amount_of_inliers

        print('Our estimated outlier share is {:.2f}%'.format(current_e * 100))

        if current_e < e:
            e = current_e
            print('Outlier share is below estimated value. Updating number of iterations.')
            n = math.log2(1 - ACCURACY) / math.log2(1 - math.pow((1 - e), SAMPLE_SIZE))
            print('RANSAC will run for approximately {} iterations. Current iteration: {}'.format(n, i))

    consensus_points = points[best_consensus_set, :]

    return best_params, consensus_points


def _get_points_on_line(params: np.ndarray) -> np.ndarray:
    x = np.ones((5, 2))
    x[:, 0] = np.arange(X_MIN, X_MAX, (X_MAX / 5))

    points = x.copy()
    y = np.dot(x, params)

    points[:, 1] = y

    return points


def main():
    inliers, outliers = _sample_points()

    points = np.append(inliers, outliers, axis=0)
    print('Total points array:')
    pprint(points)
    pprint(points.shape)

    params, ransac_inliers = _perform_ransac(points)
    line_points = _get_points_on_line(params)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_title('Points')
    ax1.plot(inliers[:, 0], inliers[:, 1], 'bo')
    ax1.plot(outliers[:, 0], outliers[:, 1], 'ro')

    ax2.set_title('Ransac Result')
    ax2.plot(ransac_inliers[:, 0], ransac_inliers[:, 1], 'bo')
    ax2.plot(line_points[:, 0], line_points[:, 1])

    plt.show()


if __name__ == '__main__':
    main()
