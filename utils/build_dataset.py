import matplotlib.pyplot as plt
import numpy as np
import json
import os
import laspy


class Rack:

    jitter_outward_limit = 1.0
    jitter_inward_limit = 0.5
    buffer_amt = 1.0

    def __init__(self, data: dict):
        """Takes in a dictionary from a LabelPC shape annotation"""
        self.type = data['label']
        # Todo: un-rotate the rack
        self.vertices = np.min(data['vertices'], axis=0), np.max(data['vertices'], axis=0)

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
        keep_1 = (points[:, :2] >= rack[0]).all(axis=1)
        keep_2 = (points[:, :2] <= rack[1]).all(axis=1)
        return keep_1 & keep_2

    @staticmethod
    def plot_rack(rack, color):
        x_min, y_min = rack[0]
        x_max, y_max = rack[1]

        x_points = [x_min, x_max, x_max, x_min, x_min]
        y_points = [y_min, y_min, y_max, y_max, y_min]

        plt.plot(x_points, y_points, color)

    def plot(self):
        """Plot the three rectangles on the same plot with different colors"""
        Rack.plot_rack(self.fine, 'green')
        Rack.plot_rack(self.jittered, 'red')
        Rack.plot_rack(self.buffered, 'blue')
        plt.legend(['Fine', 'Rough', 'Buffered', 'Min Jitter', 'Max Jitter'])

    def save(self, filename):
        with open(filename, 'w') as f:
            data = {'fine': self.fine,
                    'jittered': self.jittered,
                    'buffered': self.buffered}
            json.dump(data, f)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Create dataset')
    parser.add_argument('--multiple', action='store_true', help='Load test file with multiple racks')
    args = parser.parse_args()

    # Load a LabelPC JSON file with rack(s) in it
    if args.multiple:
        test_file = 'test/many_racks.json'
    else:
        test_file = 'test/single_rack.json'

    with open(test_file, 'r') as f:
        for shape in json.load(f)['shapes']:
            if 'rack' in shape['label']:
                r = Rack(shape)
                r.plot()
        plt.show()
