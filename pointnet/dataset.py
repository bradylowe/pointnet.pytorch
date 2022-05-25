from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import json
import os
import laspy


def flatten_rack(rack):
    return rack[0][0], rack[0][1], rack[1][0], rack[1][1]


class LasDataset(data.Dataset):

    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True,
                 normalize=False):

        self.root = root
        self.npoints = npoints
        self.split = split
        self.data_augmentation = data_augmentation
        self.normalize = normalize
        self.output_names = ['min_x', 'min_y', 'max_x', 'max_y']

        self.json_dir, self.las_dir = os.path.join(root, 'json'), os.path.join(root, 'las')
        self.las_files = [os.path.join(self.las_dir, f) for f in os.listdir(self.las_dir)]
        self.json_files = [os.path.join(self.json_dir, f) for f in os.listdir(self.json_dir)]

    def __getitem__(self, index):

        # Read the data into an array
        las = laspy.read(self.las_files[index])
        pts = np.vstack([las.x, las.y, las.z]).T

        # Randomly subsample the point cloud
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        with open(self.json_files[index], 'r') as f:
            data = json.load(f)
            fine_annot = np.asarray(data['rack'], dtype=np.float32)
            rough_annot = np.asarray(data['rough_rack'], dtype=np.float32)
            buffered_annot = np.asarray(data['buffered_annot'], dtype=np.float32)

        # Center the points and scale them to a box of size 1x1x1
        center = (buffered_annot[0] + buffered_annot[1]) / 2.0
        scale = np.max(buffered_annot[1] - buffered_annot[0])
        point_set[:2] = (point_set[:2] - center) / scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        rough_annot = torch.from_numpy(flatten_rack(rough_annot))
        fine_annot = torch.from_numpy(flatten_rack(fine_annot))
        return (point_set, rough_annot), fine_annot

    def __len__(self):
        return len(self.las_files)
