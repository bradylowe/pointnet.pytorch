import os
from typing import Tuple
import numpy as np
import pandas as pd
from utils.Voxelize import VoxelGrid
from utils.data import load_from_json, load_from_las, write_to_pkl
from utils.rack import Rack
from utils.scan import Scan
from convnet.dataset import RACK_SCALE


def build_dataset_from_scan(scan: Scan, output_dir: str, split: float = 0.1, multiplier: int = 1,
                            min_points: int = 100000, resolution: int = 512, n_slices: int = 1,
                            min_z: float = 0.5, max_z: float = 8.0, count_start: int = 0):

    # Set up output directories
    pkl_dir_train, json_dir_train = os.path.join(output_dir, 'train', 'pkl'), os.path.join(output_dir, 'train', 'json')
    pkl_dir_test, json_dir_test = os.path.join(output_dir, 'test', 'pkl'), os.path.join(output_dir, 'test', 'json')
    if not os.path.exists(pkl_dir_train):
        os.makedirs(pkl_dir_train)
    if not os.path.exists(json_dir_train):
        os.makedirs(json_dir_train)
    if not os.path.exists(pkl_dir_test):
        os.makedirs(pkl_dir_test)
    if not os.path.exists(json_dir_test):
        os.makedirs(json_dir_test)

    # Loop over racks
    count = count_start
    for rack in scan.racks:

        # Randomly shuffle rack data into train and test directories
        if split and np.random.random() > split:
            pkl_dir = pkl_dir_train
            json_dir = json_dir_train
        else:
            pkl_dir = pkl_dir_test
            json_dir = json_dir_test

        # Loop over each rack multiple times (data augmentation)
        for _ in range(multiplier):
            rack.augment()

            # Subsample rack from LAS file
            mask = Rack.points_in_rack(scan.pc.points, rack.buffered)
            if mask.sum() >= min_points:

                # Write pkl file (bitmaps)
                data_file = os.path.join(pkl_dir, f'rack_{count}.pkl')
                slices = build_bitmap(scan.pc.points.loc[mask, ['x', 'y', 'z']], resolution, n_slices, min_z, max_z)
                write_to_pkl(slices, data_file)

                # Write JSON data
                json_file = os.path.join(json_dir, f'rack_{count}.json')
                rack.save(json_file)

                count += 1


def load_data(json_file: str, min_z: float, max_z: float) -> Tuple[pd.DataFrame, dict]:
    """Returns the point cloud data (only between min_z and max_z) as well as the JSON data"""
    json_data = load_from_json(json_file)
    points = load_from_las(json_file.replace('json', 'las'), attributes=('x', 'y', 'z'), pandas=True)
    keep_above, keep_below = points.z >= min_z, points.z <= max_z
    keep = np.logical_and(keep_above, keep_below)
    return points.loc[keep], json_data


def build_bitmap(points: pd.DataFrame, resolution: int, n_slices: int,
                 min_z: float, max_z: float) -> np.ndarray:
    """
    Turns a point cloud object (with json_data) into one or more 2D bitmap images.
    :param points: Input point cloud as a pandas DataFrame object with columns x, y, and z
    :param resolution: Number of pixels in the horizontal and vertical dimensions
    :param n_slices: Number of horizontal slices to cut the point cloud into
    :param min_z: Only count data points that are above this height
    :param max_z: Only count data points that are below this height
    """

    # Move the point cloud to (x, y, z) = (0, 0, 0)
    points.loc[:, ['x', 'y']] = points[['x', 'y']] - points[['x', 'y']].min(axis=0)
    points.loc[:, 'z'] = points['z'] - points['z'].min(axis=0)

    # Build a voxel grid using the point cloud
    scale_z = (max_z - min_z) / n_slices
    vg = VoxelGrid(points[['x', 'y', 'z']].values, mesh_size=(RACK_SCALE, RACK_SCALE, scale_z))

    # Split the point cloud into a stack of 2D slices (bitmaps)
    slices = np.zeros(shape=(n_slices, resolution, resolution), dtype=np.uint8)
    for x in range(resolution):
        for y in range(resolution):
            for z in range(n_slices):
                slices[z][x][y] = vg.counts((x, y, z))

    return slices


def scale_rack_to_image(json_data: dict, scale_xy: float) -> list:
    """Apply the same scaling to a rack annotation as to the point cloud"""
    return ((np.array(json_data['fine']) - np.array(json_data['buffered'])) / scale_xy).tolist()


def test(json_path, min_z, max_z):

    for json_file in os.listdir(json_path):
        print('Working on', json_file)
        json_file = os.path.join(json_path, json_file)

        points, json_data = load_data(json_file, min_z, max_z)
        print('Data min/max:', points.min(axis=0), points.max(axis=0))
        print('Buffered annotation:', json_data['buffered'])
        print('Scaled annotation:', scale_rack_to_image(json_data, RACK_SCALE))
        points.loc[:, ['x', 'y']] = (points[['x', 'y']] - np.array(json_data['buffered'][0]))
        points.loc[:, 'z'] = points['z'] - min_z
        print('Shifted data min/max:', points.min(axis=0), points.max(axis=0))
        if input('Continue? ').lower() in ['n', 'no']:
            break


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Create dataset')
    parser.add_argument('--input_json', type=str, required=True, help='Path to input LabelPC JSON file')
    parser.add_argument('--output', type=str, help='Path at which to create new dataset')
    parser.add_argument('--split', type=float, default=0.1, help='Percentage of data to tag as testing')
    parser.add_argument('--multiplier', type=int, default=1, help='# of times to sample each rack')
    parser.add_argument('--min_points', type=float, default=100000, help='Min number of points needed to create file')
    parser.add_argument('--resolution', type=int, default=512, help='Image resolution')
    parser.add_argument('--n_slices', type=int, default=1, help='Number of vertical slices')
    parser.add_argument('--min_z', type=float, default=0.1, help='Minimum z-height to keep')
    parser.add_argument('--max_z', type=float, default=8.0, help='Maximum z-height to keep')
    parser.add_argument('--count_start', type=int, default=0, help='Beginning unique ID number for output files')
    parser.add_argument('--test', action='store_true', help='If True, then run tests')
    parser.add_argument('--plot', action='store_true', help='If True, just plot the dataset')
    args = parser.parse_args()

    scan = Scan(args.input_json)

    if args.test:
        test(args.input, args.min_z, args.max_z)
    elif args.plot:
        scan.plot()
    elif args.output:
        build_dataset_from_scan(scan, args.output, args.split, args.multiplier, args.min_points,
                                args.resolution, args.n_slices, args.min_z, args.max_z, args.count_start)
    else:
        print('No output directory provided...')
