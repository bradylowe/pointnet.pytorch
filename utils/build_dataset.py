import matplotlib.pyplot as plt
import numpy as np
import json
import os
import laspy


def rotation_matrix(degrees):
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def rotate(points, degrees, center):
    return (rotation_matrix(degrees) @ (points - center).T).T + center


def points_in_rectangle(points, rect, min_height=None, max_height=None):
    """Returns a boolean mask indicating which points are contained in this rect"""
    min_p = np.array((rect[0][0], rect[0][1], min_height)) if min_height is not None else np.array(rect[0])
    max_p = np.array((rect[1][0], rect[1][1], max_height)) if max_height is not None else np.array(rect[1])
    keep_1 = (points[:, :len(min_p)] >= min_p).all(axis=1)
    keep_2 = (points[:, :len(max_p)] <= max_p).all(axis=1)
    return keep_1 & keep_2


class Rack:

    jitter_outward_limit = 1.0
    jitter_inward_limit = 0.5
    buffer_amt = 1.0

    min_height = 0.15
    max_height = 8.0

    rotation_jitter_std = 1.0

    def __init__(self, data: dict):
        """Takes in a dictionary from a LabelPC shape annotation"""
        self.type = data['label']
        self.orient = data['orient']

        # Calculate the center of the vertices and un-rotate them
        self.vertices = np.array(data['vertices'])
        self.center = self.vertices.mean(axis=0)
        self.vertices[:, :2] = rotate(self.vertices[:, :2], -self.orient, self.center)

        # Calculate the correct (fine) annotation
        self.fine = np.array((np.min(data['vertices'], axis=0), np.max(data['vertices'], axis=0)))

        # Calculate a square that will inscribe the fine annotation at any angle
        buffered_max_dim = np.linalg.norm(self.fine[1] - self.center)
        self.square = np.array((self.center - buffered_max_dim, self.center + buffered_max_dim))

        self.jittered = None  # Rack with randomly jittered vertices
        self.buffered = None  # Rack with buffer around jittered rack
        self.rot_jitter = None  # Small random rotation jitter
        self.augment()

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

    def min_jitter_bounds(self):
        sum_limit = Rack.jitter_outward_limit + Rack.jitter_inward_limit
        origin = self.fine[0] - Rack.jitter_outward_limit
        return np.array((origin, origin + sum_limit))

    def max_jitter_bounds(self):
        sum_limit = Rack.jitter_outward_limit + Rack.jitter_inward_limit
        origin = self.fine[1] + Rack.jitter_outward_limit
        return np.array((origin - sum_limit, origin))

    @staticmethod
    def plot_rack(rack, angle, center, color, linestyle='-'):

        (x_min, y_min), (x_max, y_max) = rack[0], rack[1]
        vertices = np.array(((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)))

        vertices = rotate(vertices, angle, center)
        x_points, y_points = vertices.T

        plt.plot(x_points, y_points, color, linestyle=linestyle)

    def plot(self):
        """Plot the three rectangles on the same plot with different colors"""
        total_angle = self.orient + self.rot_jitter
        Rack.plot_rack(self.fine, total_angle, self.center, 'green')
        Rack.plot_rack(self.jittered, total_angle, self.center, 'red', '--')
        Rack.plot_rack(self.buffered, total_angle, self.center, 'blue')
        Rack.plot_rack(self.min_jitter_bounds(), total_angle, self.center, 'violet')
        Rack.plot_rack(self.max_jitter_bounds(), total_angle, self.center, 'purple')
        plt.legend(['Fine', 'Rough', 'Buffered', 'Min Jitter', 'Max Jitter'])

    def save(self, filename):
        with open(filename, 'w') as f:
            total_angle = self.orient + self.rot_jitter
            data = {'fine': rotate(self.fine, total_angle, self.center).tolist(),
                    'jittered': rotate(self.jittered, total_angle, self.center).tolist(),
                    'buffered': rotate(self.buffered, total_angle, self.center).tolist(),
                    'orient': total_angle}
            json.dump(data, f)

    def augment(self):
        self.jittered = self.jitter(self.fine)
        self.buffered = self.buffer(self.jittered)
        self.rot_jitter = np.random.randn() * Rack.rotation_jitter_std


class PointCloud:

    def __init__(self, filename, rgb=False, classification=False, intensity=False, user_data=False):
        self.filename = filename

        las = laspy.read(filename)
        self.points = np.vstack((las.x, las.y, las.z)).T
        self.rgb = np.vstack((las.red, las.green, las.blue)).T if rgb else None
        self.classification = las.classification if classification else None
        self.intensity = las.intensity if intensity else None
        self.user_data = las.user_data if user_data else None

    @staticmethod
    def save_xyz(points, filename):
        header = laspy.LasHeader(point_format=2, version='1.2')
        las = laspy.LasData(header)
        las.x, las.y, las.z = points.T
        las.write(filename)

    @staticmethod
    def plot(points):
        x, y = points.T[:2]
        plt.scatter(x, y, 1, 'black')


class Scan:

    def __init__(self, json_file, rgb=False, classification=False, intensity=False, user_data=False):

        # Load annotations and parse racks
        self.json_file = json_file
        with open(self.json_file, 'r') as f:
            self.json_data = json.load(f)

        # Load points
        source = self.json_data['source']
        ext = source.split('.')[-1]
        self.pc_file = os.path.join(os.path.dirname(self.json_file), source).replace('JSON', ext.upper())
        self.pc = PointCloud(self.pc_file, rgb=rgb, classification=classification,
                             intensity=intensity, user_data=user_data)

        # Load rack annotations, find points for each rack
        self.racks = [Rack(shape) for shape in self.json_data['shapes'] if 'rack' in shape['label']]
        self.rack_points = [self.pc.points[points_in_rectangle(self.pc.points,
                                                               rack.square,
                                                               Rack.min_height,
                                                               Rack.max_height)]
                            for rack in self.racks]

        # Un-rotate the point cloud subsamples to match the un-rotated racks
        for idx, points in enumerate(self.rack_points):
            rack = self.racks[idx]
            self.rack_points[idx][:, :2] = rotate(points[:, :2], -rack.orient, rack.center)

    def save(self, output_dir, laz=False, multiplier=1, min_points=100000):

        pc_ext = 'laz' if laz else 'las'

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
        for rack, points in zip(self.racks, self.rack_points):
            for _ in range(multiplier):
                rack.augment()

                # Subsample LAS file and save
                mask = points_in_rectangle(points, rack.buffered)
                if mask.sum() >= min_points:
                    # Write point cloud data
                    pc_file = os.path.join(las_dir, f'rack_{count}.{pc_ext}')
                    rotated_points = points[mask]
                    rotated_points[:, :2] = rotate(points[mask, :2], rack.orient + rack.rot_jitter, rack.center)
                    PointCloud.save_xyz(rotated_points, pc_file)

                    # Write JSON data
                    json_file = os.path.join(json_dir, f'rack_{count}.json')
                    rack.save(json_file)

                    count += 1

    def plot(self):
        for rack, points in zip(self.racks, self.rack_points):
            rack.plot()
            mask = points_in_rectangle(points, rack.buffered)
            PointCloud.plot(rotate(points[mask, :2], rack.orient + rack.rot_jitter, rack.center))

        # Make the plot bigger so it looks a little nicer
        buf = 3
        (x_min, x_max), (y_min, y_max) = plt.xlim(), plt.ylim()
        x_len, y_len = x_max - x_min, y_max - y_min
        length = max(x_len, y_len)
        plt.xlim([x_min - buf, x_min + length + buf])
        plt.ylim([y_min - buf, y_min + length + buf])

        # Show the plot
        plt.show()


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
        scan.save(args.output, laz=args.zip, multiplier=args.multiplier, min_points=args.min_points)
    else:
        print('No output directory provided')
