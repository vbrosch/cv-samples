import math

import numpy as np


def _multi_class_svm_loss(prediction_vector: np.ndarray, true_class: int) -> float:
    v = np.delete(prediction_vector, true_class - 1)
    loss_vector = (v - prediction_vector[true_class - 1] + 1).clip(min=0)
    return np.sum(loss_vector)


def _softmax(prediction_vector: np.ndarray) -> float:
    s_e = np.exp(prediction_vector)
    s_j = np.sum(s_e)
    return s_e / s_j * 100


m = np.array([[0.5, 0.9, 0.1, -1.2], [0.1, 0.6, 0.8, 0.1], [-2, 0.2, 0, 0.8]])
t = np.array([[1.1], [-4.4], [0.2]])

print(m)
print(t)

# INPUT
x = np.array([[0, 1, 3, 20]]).transpose()
t_c = 3
# /INPUT

print(x)

y = m.dot(x) + t

print(y)

c = np.argmax(y) + 1

print('y is in class: {}'.format(c))
print('Multi class svm loss: {:.5f}'.format(_multi_class_svm_loss(y, t_c)))
print('Softmax: {}'.format(_softmax(y)))
