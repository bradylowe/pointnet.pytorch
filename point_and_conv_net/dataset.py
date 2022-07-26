from __future__ import print_function
import torch.utils.data as data
import os

from pointnet.dataset import LasDataset
from convnet.dataset import LasDatasetSlices


class PointsAndSlices(data.Dataset):

    def __init__(self, paths, split='train'):
        self.paths = paths
        self.split = split
        self.output_names = ['min_x', 'min_y', 'max_x', 'max_y']

        self.las_files, self.pkl_files, self.json_files = [], [], []
        for path in paths:
            json_dir = os.path.join(path, split, 'json')
            las_dir = os.path.join(path, split, 'las')
            pkl_dir = os.path.join(path, split, 'pkl')
            self.las_files.extend([os.path.join(las_dir, f) for f in os.listdir(las_dir)])
            self.pkl_files.extend([os.path.join(pkl_dir, f) for f in os.listdir(pkl_dir)])
            self.json_files.extend([os.path.join(json_dir, f) for f in os.listdir(json_dir)])

        self.las_dataset = LasDataset(las_files=self.las_files, json_files=self.json_files)
        self.pkl_dataset = LasDatasetSlices(pkl_files=self.pkl_files, json_files=self.json_files, split=split)

    def __getitem__(self, index):
        points, _ = self.las_dataset[index]
        slices, target = self.pkl_dataset[index]
        return (points, slices), target

    def __len__(self):
        return len(self.pkl_dataset)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Dataset path")
    opt = parser.parse_args()

    dataset = LasDatasetSlices(paths=[opt.dataset])
    for data, target in dataset:
        print()
        print('Data min:', data.min(axis=0))
        print('Data max:', data.max(axis=0))
