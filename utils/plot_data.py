import matplotlib.pyplot as plt
import numpy as np
import os


def plot_array(data, ax=None, cmap='Greens', interpolation='nearest'):
    if ax is None:
        plt.imshow(data, cmap=cmap, interpolation=interpolation)
        plt.show()
    else:
        ax.imshow(data, cmap=cmap, interpolation=interpolation)


def plot_arrays(data, titles=None, cmap='Greens', interpolation='nearest', shape=None, save_to=None, axis_labels=True):
    if shape is None:
        fig, axs = plt.subplots(len(data))
    elif shape == 'square' and len(data) > 3:
        n = int(np.sqrt(len(data)))
        fig, axs = plt.subplots(n+1, n+1)
    else:
        fig, axs = plt.subplots(*shape)

    i, j = 0, 0
    max_i = len(axs)
    axs_is_2d = isinstance(axs[0], np.ndarray)
    for idx, slice in enumerate(data):
        ax = axs[i][j] if axs_is_2d else axs[i]
        ax.imshow(slice, cmap=cmap, interpolation=interpolation)
        ax.axes.xaxis.set_visible(axis_labels)
        ax.axes.yaxis.set_visible(axis_labels)
        if titles:
            ax.set_title(titles[idx])

        i += 1
        if i == max_i:
            i = 0
            j += 1

    if save_to:
        if not os.path.isdir(os.path.dirname(save_to)):
            os.makedirs(os.path.dirname(save_to))
        fig.savefig(save_to, pad_inches=0.1, dpi=1000)
    else:
        fig.show()


if __name__ == "__main__":

    import argparse
    from convnet.dataset import LasDatasetSlices

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Dataset path")
    opt = parser.parse_args()

    dataset = LasDatasetSlices(root=opt.dataset)
    for data, _ in dataset:
        fig, axs = plt.subplots(len(data))
        if len(data) > 1:
            for slice, ax in zip(data, axs):
                ax.imshow(slice, cmap='Greens', interpolation='nearest')
        else:
            axs.imshow(data[0], cmap='Greens', interpolation='nearest')
        fig.show()
        input('Press <enter> to continue')
