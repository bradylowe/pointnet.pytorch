import numpy as np
from typing import Union, List


def rotation_matrix_2d(degrees: float) -> np.array:
    """
    takes in a floating point as degrees and outputs a 2-dimensional rotaion matrix
    of the form [[cos(degrees), -sin(degrees)],
                 [sin(degrees),  cos(degrees)]]

    """
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    return np.array(((c, -s), (s, c)))


def rotation_matrix_3d(degrees: float, axis='z') -> np.array:
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    if isinstance(axis, str):
        axis = {'x': 0, 'y': 1, 'z': 2}[axis]

    if axis == 0:
        return np.array(((1, 0, 0), (0, c, -s), (0, s, c)))
    elif axis == 1:
        return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))
    elif axis == 2:
        return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))


def rotate(points: Union[list, np.array], degrees: float, center=None, axis='z') -> np.array:
    """
    Rotate a point or set of points about a given center point and axis.
    :param points: Set of 2D or 3D points to rotate
    :param degrees: Amount to rotate points
    :param center: Center of rotation (must be same dimension as points)
    :param axis: Direction of rotation (can only be about the x, y, or z axis
    :return: Rotated set of points
    """
    if len(points):
        if hasattr(points[0], '__len__'):
            single_point = False
            points = np.asarray(points)
        else:
            single_point = True
            points = np.asarray([points])

        if len(points[0]) == 2:
            rot = rotation_matrix_2d(degrees)
        elif len(points[0]) == 3:
            rot = rotation_matrix_3d(degrees, axis)
        else:
            raise AttributeError('Geometry.rotate():  Points must be 2D or 3D')

        if center is None:
            points = np.dot(rot, points.T).T
        elif len(center) < points.shape[1]:
            center = {'x': np.insert(center, 0, 0), 'y': np.insert(center, 1, 0), 'z': np.insert(center, 2, 0)}[axis]
            points = np.dot(rot, (points - center).T).T + center
        else:
            center = center[:points.shape[1]]
            points = np.dot(rot, (points - center).T).T + center

        if single_point:
            return points[0]
        else:
            return points

    else:
        raise AttributeError('Geometry.rotate():  Points must be 2D or 3D')
