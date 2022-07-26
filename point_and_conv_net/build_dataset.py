import os
import numpy as np
from utils.data import write_to_pkl, write_to_las
from utils.rack import Rack
from utils.scan import Scan
from convnet.build_dataset import build_bitmap


def build_dataset_from_scan(scan: Scan, output_dir: str, split: float = 0.1, multiplier: int = 1,
                            min_points: int = 100000, resolution: int = 512, n_slices: int = 1,
                            min_z: float = 0.5, max_z: float = 8.0, count_start: int = 0):

    # Set up output directories
    pkl_dir_train, pkl_dir_test = os.path.join(output_dir, 'train', 'pkl'), os.path.join(output_dir, 'test', 'pkl')
    las_dir_train, las_dir_test = os.path.join(output_dir, 'train', 'las'), os.path.join(output_dir, 'test', 'las')
    json_dir_train, json_dir_test = os.path.join(output_dir, 'train', 'json'), os.path.join(output_dir, 'test', 'json')
    if not os.path.exists(pkl_dir_train):
        os.makedirs(pkl_dir_train)
    if not os.path.exists(pkl_dir_test):
        os.makedirs(pkl_dir_test)
    if not os.path.exists(las_dir_train):
        os.makedirs(las_dir_train)
    if not os.path.exists(las_dir_test):
        os.makedirs(las_dir_test)
    if not os.path.exists(json_dir_train):
        os.makedirs(json_dir_train)
    if not os.path.exists(json_dir_test):
        os.makedirs(json_dir_test)

    # Loop over racks
    count = count_start
    for rack in scan.racks:

        rack.data['min_z'] = min_z
        rack.data['max_z'] = max_z

        # Randomly shuffle rack data into train and test directories
        if split and np.random.random() > split:
            pkl_dir = pkl_dir_train
            las_dir = las_dir_train
            json_dir = json_dir_train
        else:
            pkl_dir = pkl_dir_test
            las_dir = las_dir_test
            json_dir = json_dir_test

        # Loop over each rack multiple times (data augmentation)
        for _ in range(multiplier):
            rack.augment()

            # Subsample rack from LAS file
            mask = Rack.points_in_rack(scan.pc.points, rack.buffered)
            if mask.sum() >= min_points:

                # Write points to LAS file (point cloud)
                las_file = os.path.join(las_dir, f'rack_{count}.las')
                points = scan.pc.points.loc[mask, ['x', 'y', 'z']]
                write_to_las(points.values, las_file)

                # Write points to PKL file (bitmaps)
                pkl_file = os.path.join(pkl_dir, f'rack_{count}.pkl')
                slices = build_bitmap(points, resolution, n_slices, min_z, max_z)
                write_to_pkl(slices, pkl_file)

                # Write JSON data (annotations and metadata)
                json_file = os.path.join(json_dir, f'rack_{count}.json')
                rack.save(json_file)

                count += 1


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Create dataset')
    parser.add_argument('--input_json', type=str, required=True, help='Path to input LabelPC JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path at which to create new dataset')
    parser.add_argument('--split', type=float, default=0.1, help='Percentage of data to tag as testing')
    parser.add_argument('--multiplier', type=int, default=1, help='# of times to sample each rack')
    parser.add_argument('--min_points', type=float, default=100000, help='Min number of points needed to create file')
    parser.add_argument('--resolution', type=int, default=512, help='Image resolution')
    parser.add_argument('--n_slices', type=int, default=1, help='Number of vertical slices')
    parser.add_argument('--min_z', type=float, default=0.1, help='Minimum z-height to keep')
    parser.add_argument('--max_z', type=float, default=8.0, help='Maximum z-height to keep')
    parser.add_argument('--count_start', type=int, default=0, help='Beginning unique ID number for output files')
    parser.add_argument('--plot', action='store_true', help='If True, just plot the dataset')
    args = parser.parse_args()

    scan = Scan(args.input_json)

    if args.plot:
        scan.plot()
    elif args.output:
        build_dataset_from_scan(scan, args.output, args.split, args.multiplier, args.min_points,
                                args.resolution, args.n_slices, args.min_z, args.max_z, args.count_start)
    else:
        print('No output directory provided...')
