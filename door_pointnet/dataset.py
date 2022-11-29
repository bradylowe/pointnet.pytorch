from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import pandas as pd
import os
import laspy


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

        # Randomly subsample
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        # Shift to (0, 0, 0)
        min_point = point_set.min(axis=0)
        point_set = (point_set - min_point) / row['box_size']

        if len(min_point) == 3:
            mx, my, _ = min_point
        else:
            mx, my = min_point
        answer = np.array((row['x1'] - mx, row['y1'] - my, row['x2'] - mx, row['y2'] - my)) / row['box_size']
        return torch.from_numpy(point_set).float(), torch.from_numpy(answer).float()

    def __len__(self):
        return len(self.csv_data)
