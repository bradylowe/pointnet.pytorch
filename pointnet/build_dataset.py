from utils.rack import Rack
from utils.scan import Scan
import os


def build_dataset_from_scan(scan, output_dir, zip=False, multiplier=1, min_points=100000):

    pc_ext = 'laz' if zip else 'las'

    # Set up output directories
    las_dir, json_dir = os.path.join(output_dir, pc_ext), os.path.join(output_dir, 'json')
    if not os.path.exists(las_dir):
        os.makedirs(las_dir)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    # Make sure we don't overwrite existing files
    count = 0
    for filename in os.listdir(json_dir):
        n = int(filename[5:-5])
        if n >= count:
            count = n + 1

    # Loop over racks
    for rack in scan.racks:
        for _ in range(multiplier):
            rack.augment()

            # Subsample LAS file and save
            mask = Rack.points_in_rack(scan.pc.points, rack.buffered)
            if mask.sum() >= min_points:
                pc_file = os.path.join(las_dir, f'rack_{count}.{pc_ext}')
                scan.pc.save(pc_file, mask)

                # Write JSON data
                json_file = os.path.join(json_dir, f'rack_{count}.json')
                rack.save(json_file)

                count += 1


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Create dataset')
    parser.add_argument('--input_json', type=str, required=True, help='Path to input LabelPC JSON file')
    parser.add_argument('--output', type=str, help='Path at which to create new dataset')
    parser.add_argument('--multiplier', type=int, default=1, help='# of times to sample each rack')
    parser.add_argument('--min_points', type=float, default=100000, help='Min number of points needed to create file')
    parser.add_argument('--zip', action='store_true', help='If True, save point clouds as LAZ')
    parser.add_argument('--plot', action='store_true', help='If True, just plot the dataset')
    args = parser.parse_args()

    # Load a LabelPC JSON file with rack(s) in it
    scan = Scan(args.input_json)

    if args.plot:
        scan.plot()
    elif args.output:
        build_dataset_from_scan(scan, args.output, zip=args.zip, multiplier=args.multiplier)
    else:
        print('No output directory provided')
