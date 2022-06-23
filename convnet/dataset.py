from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import os
from utils.data import load_from_json, load_from_pkl


class LasDatasetSlices(data.Dataset):

    def __init__(self, paths):
        self.paths = paths
        self.output_names = ['min_x', 'min_y', 'max_x', 'max_y']

        self.pkl_files, self.json_files = [], []
        for path in paths:
            json_dir = os.path.join(path, 'json')
            pkl_dir = os.path.join(path, 'pkl')
            self.pkl_files.extend([os.path.join(pkl_dir, f) for f in os.listdir(pkl_dir)])
            self.json_files.extend([os.path.join(json_dir, f) for f in os.listdir(json_dir)])

        self.resolution = self.get_resolution(self.pkl_files[0])
        self.n_slices = self.get_n_slices(self.pkl_files[0])

    @staticmethod
    def get_resolution(pkl_file):
        return load_from_pkl(pkl_file).shape[1]

    @staticmethod
    def get_n_slices(pkl_file):
        return load_from_pkl(pkl_file).shape[0]

    def __getitem__(self, index):

        slices = load_from_pkl(self.pkl_files[index]).astype(np.float32)
        target = load_from_json(self.json_files[index])['scaled_to_image']
        target = np.array(target, dtype=np.float32).flatten()
        return torch.from_numpy(slices), torch.from_numpy(target)

    def __len__(self):
        return len(self.pkl_files)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Dataset path")
    opt = parser.parse_args()

    dataset = LasDatasetSlices(root=opt.dataset)
    for data, target in dataset:
        print()
        print('Data min:', data.min(axis=0))
        print('Data max:', data.max(axis=0))
