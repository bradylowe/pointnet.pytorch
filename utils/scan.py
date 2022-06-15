from utils.point_cloud import PointCloud
from utils.rack import Rack
import os
import matplotlib.pyplot as plt
from utils.data import load_from_json


class Scan:

    def __init__(self, json_file, rgb=False, classification=False, intensity=False, user_data=False):

        # Load annotations and parse racks
        self.json_file = json_file
        self.json_data = load_from_json(json_file)
        self.racks = [Rack(shape) for shape in self.json_data['shapes'] if 'rack' in shape['label']]

        # Load points
        source = self.json_data['source']
        ext = source.split('.')[-1]
        self.pc_file = os.path.join(os.path.dirname(self.json_file), source).replace('JSON', ext.upper())
        self.pc = PointCloud(self.pc_file, rgb=rgb, classification=classification,
                             intensity=intensity, user_data=user_data)

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
