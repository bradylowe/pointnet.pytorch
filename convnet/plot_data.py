import matplotlib.pyplot as plt
import numpy as np


def plot_bitmap(ax, data, cmap='Greens', interpolation='nearest', title=None, show_labels=True):
    ax.imshow(data, cmap=cmap, interpolation=interpolation)
    ax.axes.xaxis.set_visible(show_labels)
    ax.axes.yaxis.set_visible(show_labels)
    if title:
        ax.set_title(title)


def plot_arrays(data, cmap='Greens', interpolation='nearest', shape=None, titles=None, show_labels=True):

    if shape is None:
        root_len = np.sqrt(len(data))
        int_len = int(root_len)
        shape = int_len, int_len + int(root_len - int_len > 0.001)
    elif isinstance(shape, int):
        shape = shape, shape

    def get_i_j_from_idx_and_shape(idx, shape):
        return idx // shape[0], idx % shape[0]

    if titles is None:
        if len(data) == 1:
            titles = 'Channel 0'
        else:
            titles = [f'Channel {idx}' for idx in range(len(data))]

    fig, axs = plt.subplots(*shape if hasattr(shape, '__len__') else shape)
    if len(data) == 1:
        plot_bitmap(axs, data[0], cmap=cmap, interpolation=interpolation, title=titles)
    else:
        for idx, slice in enumerate(data):
            # Find correct axis to draw on
            if isinstance(shape, int):
                ax = axs[idx]
            else:
                i, j = get_i_j_from_idx_and_shape(idx, shape)
                ax = axs[j][i]

            title = titles[idx] if hasattr(titles, '__len__') else titles
            plot_bitmap(ax, slice, cmap, interpolation, title, show_labels)

    return fig, axs


if __name__ == "__main__":

    import os
    import argparse
    from convnet.dataset import LasDatasetSlices

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Dataset path")
    parser.add_argument('--output', type=str, help="Path to store PNG files")
    parser.add_argument('--show', action='store_true', help="Show the plots to the user")
    opt = parser.parse_args()

    if opt.output and not os.path.isdir(os.path.dirname(opt.output)):
        os.makedirs(os.path.dirname(opt.output))

    dataset = LasDatasetSlices(root=opt.dataset)
    for data, _ in dataset:
        fig, axs = plot_arrays(data)

        if opt.output:
            png_file = os.path.join(opt.output, '')
            fig.savefig(png_file, pad_inches=0.1, dpi=1000)

        if opt.show:
            fig.show()

        if input('Continue?  ').lower() in ['n', 'no']:
            break
