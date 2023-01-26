import numpy as np

from models.Vector import Vector
import time


def slow_down_training(sleep_time: float):
    """
    Sleeps for a specified number of seconds between each iteration
    of the training process.

    Args:
    - sleep_time: float: number of seconds to sleep between each iteration.
    """
    time.sleep(sleep_time)

def vector_to_numpy(vec):
    return np.array([vec.X, vec.Y, vec.Z])


def vectors_to_numpy(vectors):
    return np.array([vector_to_numpy(v) for v in vectors])


def minus(v1, v2):
    assert isinstance(v1, Vector)
    assert isinstance(v2, Vector)
    vec = Vector()
    vec.X = v2.X - v1.X
    vec.Y = v2.Y - v1.Y
    vec.Z = v2.Z - v1.Z
    return vec


def near_by_vectors(from_vec, tos, limit):
    """Processes several nearby vectors, returns a list of specified length sorted by distance,
       if there are more than the limit, it truncates the tail, if less, it pads with zeroes."""

    tos.sort(key=lambda to_vec: distance(to_vec, from_vec))  # sort by distance, from close to far
    if len(tos) >= limit:  # truncate the tail if there are more than limit
        return [minus(from_vec, to_vec) for to_vec in tos[:limit]]
    else:  # pad with zeroes if there are less than limit
        direction_vecs = [minus(from_vec, to_vec) for to_vec in tos]
        direction_vecs.extend(
            [Vector({'X': 0, 'Y': 0, 'Z': 0}) for i in range(0, limit - len(tos))])
        return direction_vecs


def calculate_xy_angle(vec1, vec2):
    """Calculates the angle between two 2D vectors in the XY plane"""


    assert isinstance(vec1, Vector)
    assert isinstance(vec2, Vector)
    num_vec1 = np.array([vec1.X, vec1.Y])  # only need the x,y values
    num_vec2 = np.array([vec2.X, vec2.Y])
    cos = np.dot(num_vec2, num_vec2) / \
        (np.linalg.norm(num_vec2)*(np.linalg.norm(num_vec1)))  # this line is wrong, it should be np.dot(num_vec1, num_vec2)
    cos = np.clip(cos, -1, 1)
    return cos

def distance(v1, v2):
    assert isinstance(v1, Vector)
    assert isinstance(v2, Vector)
    num_v1 = np.array([v1.X, v1.Y, v1.Z])
    num_v2 = np.array([v2.X, v2.Y, v2.Z])
    return np.sqrt(np.sum(np.square(num_v1 - num_v2)))
