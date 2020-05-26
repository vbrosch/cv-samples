from pprint import pprint

import numpy as np

np.set_printoptions(suppress=True)

point_correspondences = [
    ((0, 0, 2), (0, 0)),
    ((1, 0, 2), (2 / 3, 0)),
    ((0, 1, 2), (0, 2 / 3)),
    ((1, 1, 2), (2 / 3, 2 / 3)),
    ((1, 1, 5), (0.4, 0.4)),
    ((2, 2, 5), (0.8, 0.8))
]

points = [(-3, 1, 5), (1, 3, 5), (3, -1, 2), (-1, -3, 2)]


def _build_matrix() -> np.ndarray:
    p = []

    for (x_o, y_o, z_o), (u_1, v_1) in point_correspondences:
        p.append([x_o, y_o, z_o, 1, 0, 0, 0, 0, -u_1 * x_o, -u_1 * y_o, -u_1 * z_o, -u_1])
        p.append([0, 0, 0, 0, x_o, y_o, z_o, 1, -v_1 * x_o, -v_1 * y_o, -v_1 * z_o, -v_1])

    return np.array(p)


def main():
    x = _build_matrix()

    print('Point Matrix:')
    pprint(x)

    u, s, vh = np.linalg.svd(x)

    camera_matrix = vh[-1].reshape(3, 4)
    normalised_matrix = camera_matrix / camera_matrix[2, 3]
    print('Camera Matrix:')
    pprint(camera_matrix)

    print('Normalised Camera Matrix:')
    pprint(normalised_matrix)

    for x, y, z in points:
        point = np.dot(normalised_matrix, np.array([[x], [y], [z], [1]])).squeeze()
        x_o, y_o, z_o = tuple(point)
        x_i, y_i = x_o / z_o, y_o / z_o

        print('Point: ({:.4f}, {:.4f}, {:.4f}) => ({:.4f}, {:.4f})'.format(x_o, y_o, z_o, x_i, y_i))


if __name__ == '__main__':
    main()
