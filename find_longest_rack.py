import os
from utils.rack import Rack
from utils.data import load_from_json


def largest_length(data_path):
    max_length = 0.0
    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.json'):
                json_file = os.path.join(subdir, file)
                max_length = max(max_length, Rack(load_from_json(json_file).length))

    return max_length
