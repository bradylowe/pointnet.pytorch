from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

from pointnet.model import PointNetCls
from convnet.model import SimpleConv2d


class PointnetPlusConv(nn.Module):
    def __init__(self, k, point_dim, image_resolution, n_slices=1):
        super().__init__()
        self.pointnet = PointNetCls(k=k, feature_transform=False, point_dim=point_dim, return_features=True)
        self.convnet = SimpleConv2d(image_resolution=image_resolution, n_slices=n_slices, return_features=True)

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 4)

    def forward(self, x):
        x = torch.cat([self.pointnet(x[0]), self.convnet(x[1])], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


if __name__ == '__main__':
    pass
