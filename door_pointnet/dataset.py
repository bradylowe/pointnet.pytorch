from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import pandas as pd
import os
import laspy
from utils.geometry import rotate


class LasDataset(data.Dataset):

    def __init__(self,
                 root=None,
                 npoints=2500,
                 split='train',
                 data_augmentation=False,
                 normalize=False,
                 point_attribs=('x', 'y', 'z')):

        if root is not None:
            self.root = root
            self.csv_file = os.path.join(root, 'doors.csv')
            self.csv_data = pd.read_csv(self.csv_file)
        else:
            print('Error: Could not create LasDataset object')
            return

        self.npoints = npoints
        self.point_attribs = point_attribs
        self.point_dim = len(self.point_attribs)
        self.split = split
        self.data_augmentation = data_augmentation
        self.normalize = normalize
        self.output_names = ['x1', 'y1', 'x2', 'y2']

    def __getitem__(self, index):

        # Read the data into an array
        row = self.csv_data.iloc[index]

        las = laspy.read(os.path.join(self.root, row['las']))
        pts = np.vstack([getattr(las, attr) for attr in self.point_attribs]).T
        target = np.array((row['x1'], row['y1'], row['x2'], row['y2']))

        # Randomly subsample
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        if self.data_augmentation:
            rot = np.random.random() * 360.0
            point_set = rotate(point_set, rot)
            target[0:2] = rotate(target[0:2], rot)
            target[2:4] = rotate(target[2:4], rot)

        # Shift to (0, 0, 0)
        min_point = point_set.min(axis=0)
        point_set = (point_set - min_point) / row['box_size']
        target[0:2] = (target[0:2] - min_point[:2]) / row['box_size']
        target[2:4] = (target[2:4] - min_point[:2]) / row['box_size']

        return torch.from_numpy(point_set).float(), torch.from_numpy(target).float()

    def __len__(self):
        return len(self.csv_data)
