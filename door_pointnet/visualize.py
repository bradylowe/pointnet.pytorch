import argparse
import os
import torch.utils.data
from door_pointnet.dataset import LasDataset
from door_pointnet.model import PointNetCls

import matplotlib.pyplot as plt
import numpy as np


def plot_results(points, pred, correct, filename=None):
    plt.scatter(*points, np.ones(len(points[0])), marker='.')
    x1, y1, x2, y2 = pred
    plt.scatter((x1, x2), (y1, y2), (40, 40), c='red')
    x1, y1, x2, y2 = correct
    plt.scatter((x1, x2), (y1, y2), (60, 60), c='darkgreen')

    if filename:
        plt.savefig(filename, dpi=1000)
        plt.clf()
    else:
        plt.show()


os.system('color')
blue = lambda x: '\033[94m' + x + '\033[0m'


parser = argparse.ArgumentParser()
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--model', type=str, required=True, help='model path')
parser.add_argument('--dataset', type=str, required=True, help="Dataset path")
parser.add_argument('--n_items', type=int, default=0, help="Maximum number of items to process (default all)")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--output', type=str, help="Path to put output images")
opt = parser.parse_args()


# Load data
point_attributes = ['x', 'y', 'z']
dataset = LasDataset(
    root=opt.dataset,
    npoints=opt.num_points,
    point_attribs=point_attributes)
print('Number of items:', len(dataset))
output_dim = len(dataset.output_names)
print('Output names', dataset.output_names)

# Check for output dir (create if does not exist)
if opt.output and not os.path.exists(opt.output):
    os.makedirs(opt.output)

# Load model
classifier = PointNetCls(k=output_dim, feature_transform=opt.feature_transform, point_dim=dataset.point_dim)
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()


for idx in range(opt.n_items or len(dataset)):
    las_file = dataset.csv_data.iloc[idx]['las']
    points, target = dataset[idx]
    points = points.T[None, :]
    pred, _, _ = classifier.forward(points)
    points, pred, target = points.detach().numpy(), pred.detach().numpy(), target.detach().numpy()
    if opt.output:
        plot_results(points[0], pred[0], target, os.path.join(opt.output, las_file.replace('.las', '.png')))
    else:
        plot_results(points[0], pred[0], target)

