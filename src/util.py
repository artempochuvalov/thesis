import numpy as np
from numpy.typing import NDArray

def normalize_values(data: NDArray[float]) -> NDArray[float]:
    min = np.min(data)
    max = np.max(data)

    return (data - min) / (max - min)

def ideal_point_method(
    X: NDArray[float],
    Y: NDArray[float],
    Z: NDArray[float],
    ideal_point: (float, float, float)
) -> NDArray[float]:
    points_num = len(X)
    distances = np.empty(points_num, dtype=float)

    for i in range(points_num):
        distances[i] = np.sqrt((X[i] - ideal_point[0]) ** 2 + (Y[i] - ideal_point[1]) ** 2 + (Z[i] - ideal_point[2]) ** 2)

    return distances

def linear_convolution(
    X: NDArray[float],
    Y: NDArray[float],
    Z: NDArray[float],
    weights: (float, float, float)
) -> NDArray[float]:
    return weights[0] * X + weights[1] * Y + weights[2] * Z

def little_law(lambda_val: float, requests_in_system, loss_probability: float):
    return requests_in_system / (lambda_val * (1 - loss_probability))
