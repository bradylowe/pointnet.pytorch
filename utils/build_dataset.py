import matplotlib.pyplot as plt
import numpy as np
import json


def read_shapes_from_json(filename):
    """Opens a JSON file created by LabelPC and returns the list of shapes"""
    with open(filename, 'r') as f:
        return json.load(f)['shapes']


def get_racks_from_shapes(shapes):
    """Returns the rack annotations from a list of LabelPC shape annotations"""
    return [shape for shape in shapes if 'rack' in shape['label']]


def get_min_max_vertices_from_racks(racks):
    """
    Returns a numpy array of rack vertices from a list of rack annotations.
    The arrays contain 2 vertices per rack (min_vertex, max_vertex).
    """
    return np.asarray([(np.min(rack['vertices'], axis=0), np.max(rack['vertices'], axis=0)) for rack in racks])


def add_random_jitter_to_rack(rack, inward_limit: float, outward_limit: float):
    """
    Returns a rack annotation that has been modified randomly.
    The points only move inward (into the rack) by at most `inward_limit`.
    The points only move outward (out of the rack) by at most `outward_limit`.
    """
    # Calculate random jitter within bounds
    # Apply random jitter and return result
    return np.array((rack[0] + inward_limit, rack[1] - inward_limit))


def add_buffer_to_rack(rack, buffer: float):
    """
    Returns a rack which has been extended outward in the x- and y-directions
    by the amount of the given buffer.
    To be specific, if buffer=1, then the left side of the rack is moved
    leftward by 1 meter, the right side of the rack is moved rightward
    by 1 meter, and similarly with the top and bottom sides of the rack.
    """
    return np.array((rack[0] - buffer, rack[1] + buffer))


def plot_rack(rack, color='black'):
    x_min, y_min = rack[0]
    x_max, y_max = rack[1]

    x_points = [x_min, x_max, x_max, x_min, x_min]
    y_points = [y_min, y_min, y_max, y_max, y_min]

    plt.plot(x_points, y_points, color)


if __name__ == "__main__":

    # Load a LabelPC JSON file with rack(s) in it
    test_file = 'test/single_rack.json'

    # Get the rack(s)
    shapes = read_shapes_from_json(test_file)
    racks = get_racks_from_shapes(shapes)
    racks = get_min_max_vertices_from_racks(racks)

    for rack in racks:

        # Calculate the other racks
        rough_rack = add_random_jitter_to_rack(rack, -0.5, 1.0)
        buffered_rack = add_buffer_to_rack(rough_rack, 1.0)

        # Plot racks
        plot_rack(rack, 'red')
        plot_rack(rough_rack, 'green')
        plot_rack(buffered_rack, 'blue')

    # Show the plot
    plt.legend(['Orig', 'Jittered', 'Buffered'])

    # Make the plot bigger so it looks a little nicer
    buf = 3
    x_lim, y_lim = plt.xlim(), plt.ylim()
    plt.xlim([x_lim[0] - buf, x_lim[1] + buf])
    plt.ylim([y_lim[0] - buf, y_lim[1] + buf])

    # Show the plot
    plt.show()

