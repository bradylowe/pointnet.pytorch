from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import os
from utils.data import load_from_json, load_from_pkl


class LasDatasetSlices(data.Dataset):

    def __init__(self, root):
        self.root = root
        self.output_names = ['min_x', 'min_y', 'max_x', 'max_y']

        self.json_dir, self.pkl_dir = os.path.join(root, 'json'), os.path.join(root, 'pkl')
        self.pkl_files = [os.path.join(self.pkl_dir, f) for f in os.listdir(self.pkl_dir)]
        self.json_files = [os.path.join(self.json_dir, f) for f in os.listdir(self.json_dir)]

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
