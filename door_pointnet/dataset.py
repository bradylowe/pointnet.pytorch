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
        self.output_dim = len(self.output_names)

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


class PointPillarsDataset(data.Dataset):

    def __init__(self,
                 root=None,
                 n_pillars=20,
                 grid_size=0.1,
                 points_per_pillar=2500,
                 split='train',
                 data_augmentation=False,
                 point_attribs=('x', 'y', 'z')):

        if root is not None:
            self.root = root
            self.csv_file = os.path.join(root, 'doors.csv')
            self.csv_data = pd.read_csv(self.csv_file)
        else:
            print('Error: Could not create LasDataset object')
            return

        self.grid_size = grid_size
        self.n_pillars = n_pillars
        self.points_per_pillar = points_per_pillar
        self.point_attribs = point_attribs
        self.point_dim = len(self.point_attribs)
        self.split = split
        self.data_augmentation = data_augmentation
        self.output_names = ['x1', 'y1', 'x2', 'y2']

    def __getitem__(self, index):

        # Retrieve the data
        row = self.csv_data.iloc[index]
        las = laspy.read(os.path.join(self.root, row['las']))
        pts = np.vstack([getattr(las, attr) for attr in self.point_attribs]).T
        target = np.array((row['x1'], row['y1'], row['x2'], row['y2']))

        if self.data_augmentation:
            rot = np.random.random() * 360.0
            pts[:, :2] = rotate(pts[:, :2], rot)
            target[0:2] = rotate(target[0:2], rot)
            target[2:4] = rotate(target[2:4], rot)

        # Shift to (0, 0, 0)
        min_point = pts[:, :2].min(axis=0)
        pts[:, :2] = (pts[:, :2] - min_point)
        target[0:2] = (target[0:2] - min_point[:2])
        target[2:4] = (target[2:4] - min_point[:2])

        # Count and sort the points (2d histogram)
        dims = pts[:, :2].max(axis=0)
        bins = (dims / self.grid_size).astype(int) + 1
        h, x_edges, y_edges = np.histogram2d(*pts[:, :2].T, bins=bins)

        # Find the fullest pillars, build index map
        sorted_h = np.argsort(h)[:-self.n_pillars]
        x_ind, y_ind = np.unravel_index(sorted_h, h.shape)

        # Build the pillars
        point_dim = pts.shape[1]
        output_dim = point_dim + 5
        pillars = np.zeros((self.n_pillars, self.points_per_pillar, output_dim))
        for pillar_idx, (xi, yi) in enumerate(zip(x_ind, y_ind)):

            # Find the points in this pillar
            x_min, y_min = x_edges[xi], y_edges[yi]
            keep = (x_min < pts[:, 0]) & (pts[:, 0] < x_min + self.grid_size)
            keep &= (y_min < pts[:, 1]) & (pts[:, 1] < y_min + self.grid_size)
            these_pts = pts[keep]

            if len(these_pts) > self.points_per_pillar:
                these_pts = these_pts[:self.points_per_pillar]

            mean = these_pts[:, :3].mean(axis=0)
            distances = np.abs(these_pts[:, :3] - mean)

            center = np.array(x_min, y_min) + self.grid_size / 2.0
            offsets = these_pts[:, :2] - center

            n_points = len(these_pts)
            pillars[pillar_idx, :n_points, :point_dim] = these_pts
            pillars[pillar_idx, :n_points, point_dim:point_dim+3] = distances
            pillars[pillar_idx, :n_points, point_dim+3:point_dim+5] = offsets

        return torch.from_numpy(pillars).float(), torch.from_numpy(target).float(), \
               torch.from_numpy(x_ind), torch.from_numpy(y_ind)

    def __len__(self):
        return len(self.csv_data)
