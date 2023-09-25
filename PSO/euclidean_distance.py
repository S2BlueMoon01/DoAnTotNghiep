import numpy as np


def euclidean_distance(point1, point2):
    """
    Tính khoảng cách Euclidean giữa hai điểm.

    Parameters:
    point1 (list): Tọa độ của điểm thứ nhất.
    point2 (list): Tọa độ của điểm thứ hai.

    Returns:
    float: Khoảng cách Euclidean giữa hai điểm.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))
