import numpy as np
import matplotlib.pyplot as plt
from utils.data import write_to_json


class Rack:

    jitter_outward_limit = 1.0
    jitter_inward_limit = 0.5
    buffer_amt = 1.0

    min_height = 0.15
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
        xyz = points.loc[:, ['x', 'y', 'z']].values
        min_p = np.array((rack[0][0], rack[0][1], Rack.min_height))
        max_p = np.array((rack[1][0], rack[1][1], Rack.max_height))
        keep_1 = (xyz >= min_p).all(axis=1)
        keep_2 = (xyz <= max_p).all(axis=1)
        return np.logical_and(keep_1, keep_2)

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
        data = {'fine': self.fine.tolist(),
                'jittered': self.jittered.tolist(),
                'buffered': self.buffered.tolist()}
        write_to_json(data, filename)

    def augment(self):
        self.jittered = self.jitter(self.fine)
        self.buffered = self.buffer(self.jittered)
