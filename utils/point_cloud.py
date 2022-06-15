from utils.data import load_from_las, write_to_las
import matplotlib.pyplot as plt


class PointCloud:

    def __init__(self, filename, rgb=False, classification=False, intensity=False, user_data=False):
        self.filename = filename
        attributes = ['x', 'y', 'z']
        if rgb:
            attributes.extend(['red', 'green', 'blue'])
        if classification:
            attributes.append('class')
        if intensity:
            attributes.append('intensity')
        if user_data:
            attributes.append('user_data')
        self.points = load_from_las(filename, attributes)

    def save(self, filename, mask):
        write_to_las(self.points.loc[mask], filename)

    def plot(self, mask):
        x, y = self.points.loc[mask, ['x', 'y']].T
        plt.scatter(x, y, 1, 'black')
