import matplotlib.pyplot as plt
import numpy as np
import json
import os
import laspy


class Rack:

    jitter_outward_limit = 1.0
    jitter_inward_limit = 0.5
    buffer_amt = 1.0

    min_height = 0.5
    max_height = 8.0

    def __init__(self, data: dict):
        """Takes in a dictionary from a LabelPC shape annotation"""
        self.type = data['label']
        # Todo: un-rotate the rack
        self.vertices = np.array((np.min(data['vertices'], axis=0), np.max(data['vertices'], axis=0)))

        self.fine = self.vertices
        self.jittered = self.jitter(self.fine)
        self.buffered = self.buffer(self.jittered)

    @staticmethod
    def jitter(rack):
        """
        Returns a rack annotation that has been modified randomly.
        The points only move inward (into the rack) by at most `jitter_inward_limit`.
        The points only move outward (out of the rack) by at most `jitter_outward_limit`.
        """
        sum_limit = Rack.jitter_outward_limit + Rack.jitter_inward_limit
        jitter_1, jitter_2 = np.random.random(2) * sum_limit
        return np.array((rack[0] - Rack.jitter_outward_limit + jitter_1,
                         rack[1] + Rack.jitter_outward_limit - jitter_2))

    @staticmethod
    def buffer(rack):
        """
        Returns a rack which has been extended outward in the x- and y-directions
        by the amount of the given buffer.
        To be specific, if buffer=1, then the left side of the rack is moved
        leftward by 1 meter, the right side of the rack is moved rightward
        by 1 meter, and similarly with the top and bottom sides of the rack.
        """
        return np.array((rack[0] - Rack.buffer_amt, rack[1] + Rack.buffer_amt))

    @staticmethod
    def points_in_rack(points, rack):
        """Returns a boolean mask indicating which points are contained in this rack"""
        min_p = np.array((rack[0][0], rack[0][1], Rack.min_height))
        max_p = np.array((rack[1][0], rack[1][1], Rack.max_height))
        keep_1 = (points[:, :3] >= min_p).all(axis=1)
        keep_2 = (points[:, :3] <= max_p).all(axis=1)
        return keep_1 & keep_2

    @staticmethod
    def min_jitter_bounds(rack):
        sum_limit = Rack.jitter_outward_limit + Rack.jitter_inward_limit
        center = rack[0] - (Rack.jitter_outward_limit - Rack.jitter_inward_limit) / 2.0
        min_jitter, max_jitter = center + (0.0 - 0.5) * sum_limit, center + (1.0 - 0.5) * sum_limit
        return min_jitter, max_jitter

    @staticmethod
    def max_jitter_bounds(rack):
        sum_limit = Rack.jitter_outward_limit + Rack.jitter_inward_limit
        center = rack[1] + (Rack.jitter_outward_limit - Rack.jitter_inward_limit) / 2.0
        min_jitter, max_jitter = center + (0.0 - 0.5) * sum_limit, center + (1.0 - 0.5) * sum_limit
        return min_jitter, max_jitter

    @staticmethod
    def plot_rack(rack, color, linestyle='-'):
        x_min, y_min = rack[0]
        x_max, y_max = rack[1]

        x_points = [x_min, x_max, x_max, x_min, x_min]
        y_points = [y_min, y_min, y_max, y_max, y_min]

        plt.plot(x_points, y_points, color, linestyle=linestyle)

    def plot(self):
        """Plot the three rectangles on the same plot with different colors"""
        Rack.plot_rack(self.fine, 'green')
        Rack.plot_rack(self.jittered, 'red', '--')
        Rack.plot_rack(self.buffered, 'blue')
        Rack.plot_rack(Rack.min_jitter_bounds(self.fine), 'violet')
        Rack.plot_rack(Rack.max_jitter_bounds(self.fine), 'purple')
        plt.legend(['Fine', 'Rough', 'Buffered', 'Min Jitter', 'Max Jitter'])

    def save(self, filename):
        with open(filename, 'w') as f:
            data = {'fine': self.fine.tolist(),
                    'jittered': self.jittered.tolist(),
                    'buffered': self.buffered.tolist()}
            json.dump(data, f)

    def augment(self):
        self.jittered = self.jitter(self.fine)
        self.buffered = self.buffer(self.jittered)


class PointCloud:

    def __init__(self, filename, rgb=False, classification=False, intensity=False, user_data=False):
        self.filename = filename

        las = laspy.read(filename)
        self.points = np.vstack((las.x, las.y, las.z)).T
        self.rgb = np.vstack((las.red, las.green, las.blue)).T if rgb else None
        self.classification = las.classification if classification else None
        self.intensity = las.intensity if intensity else None
        self.user_data = las.user_data if user_data else None

    def save(self, filename, mask):
        header = laspy.LasHeader(point_format=2, version='1.2')
        las = laspy.LasData(header)

        las.x, las.y, las.z = self.points[mask].T
        if self.rgb is not None:
            las.red, las.green, las.blue = self.rgb[mask].T
        if self.classification is not None:
            las.classification = self.classification[mask]
        if self.intensity is not None:
            las.intensity = self.intensity[mask]
        if self.user_data is not None:
            las.user_data = self.user_data[mask]

        las.write(filename)

    def plot(self, mask):
        x, y = self.points[mask, :2].T
        plt.scatter(x, y, 1, 'black')


class Scan:

    def __init__(self, json_file, rgb=False, classification=False, intensity=False, user_data=False):

        # Load annotations and parse racks
        self.json_file = json_file
        with open(self.json_file, 'r') as f:
            self.json_data = json.load(f)
        self.racks = [Rack(shape) for shape in self.json_data['shapes'] if 'rack' in shape['label']]

        # Load points
        source = self.json_data['source']
        ext = source.split('.')[-1]
        self.pc_file = os.path.join(os.path.dirname(self.json_file), source).replace('JSON', ext.upper())
        self.pc = PointCloud(self.pc_file, rgb=rgb, classification=classification,
                             intensity=intensity, user_data=user_data)

    def save(self, output_dir, zip=False, multiplier=1):

        pc_ext = 'laz' if zip else 'las'

        # Set up output directories
        las_dir, json_dir = os.path.join(output_dir, pc_ext), os.path.join(output_dir, 'json')
        if not os.path.exists(las_dir):
            os.makedirs(las_dir)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        # Make sure we don't overwrite existing files
        idx_offset = 0
        for filename in os.listdir(json_dir):
            n = int(filename[5:-5])
            if n > idx_offset:
                idx_offset = n
        idx_offset += 1

        # Loop over racks
        for rack_idx, rack in enumerate(self.racks):
            for _ in range(multiplier):
                rack.augment()

                idx = rack_idx + idx_offset

                # Write JSON data
                json_file = os.path.join(json_dir, f'rack_{idx}.json')
                rack.save(json_file)

                # Subsample LAS file and save
                mask = Rack.points_in_rack(self.pc.points, rack.buffered)
                pc_file = os.path.join(las_dir, f'rack_{idx}.{pc_ext}')
                self.pc.save(pc_file, mask)

    def plot(self):
        for rack in self.racks:
            rack.plot()
            self.pc.plot(Rack.points_in_rack(self.pc.points, rack.buffered))

        # Make the plot bigger so it looks a little nicer
        buf = 3
        x_lim, y_lim = plt.xlim(), plt.ylim()
        plt.xlim([x_lim[0] - buf, x_lim[1] + buf])
        plt.ylim([y_lim[0] - buf, y_lim[1] + buf])

        # Show the plot
        plt.show()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Create dataset')
    parser.add_argument('--input_json', type=str, required=True, help='Path to input LabelPC JSON file')
    parser.add_argument('--output', type=str, help='Path at which to create new dataset')
    parser.add_argument('--multiplier', type=int, default=1, help='# of times to sample each rack')
    parser.add_argument('--zip', action='store_true', help='If True, save point clouds as LAZ')
    parser.add_argument('--plot', action='store_true', help='If True, just plot the dataset')
    args = parser.parse_args()

    # Load a LabelPC JSON file with rack(s) in it
    scan = Scan(args.input_json)

    if args.plot:
        scan.plot()
    elif args.output:
        scan.save(args.output, zip=args.zip, multiplier=args.multiplier)
    else:
        print('No output directory provided')
