import json
import pickle
import laspy
import numpy as np
import pandas as pd


def load_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def write_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_from_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def write_to_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_las(filename, attributes=('x', 'y', 'z'), pandas=False):
    las = laspy.read(filename)
    data = np.vstack([las.__getattr__(attr) for attr in attributes]).T
    if pandas:
        return pd.DataFrame(data, columns=attributes)
    else:
        return data


def write_to_las(data, filename):
    header = laspy.LasHeader(point_format=2, version='1.2')
    las = laspy.LasData(header)
    if isinstance(data, pd.DataFrame):
        for column in data:
            las.__setattr__(column, data[column])
    else:
        data = data.T
        las.x = data[0]
        las.y = data[1]
        if len(las) > 2:
            las.z = data[2]
    las.write(filename)
