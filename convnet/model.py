from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


def output_size(input_size, kernel, padding, stride):
    input_size = np.array(input_size, dtype=int)
    kernel = np.array(kernel, dtype=int)
    padding = np.array(padding, dtype=int)
    stride = np.array(stride, dtype=int)
    return (np.floor((input_size - kernel + 2 * padding) / stride) + 1).astype(int)


def output_size_many(input_size, layers):
    if len(layers) == 1:
        return output_size(input_size, *layers[0])
    else:
        size = output_size(input_size, *layers[0])
        return output_size_many(size, layers[1:])


def total_items_in_block(img_size, channels):
    if isinstance(img_size, np.ndarray):
        return channels * np.product(img_size)
    else:
        return channels * img_size ** 2


class SimpleConv2d(nn.Module):
    def __init__(self, image_resolution, n_slices=1):
        super().__init__()

        self.image_resolution = image_resolution
        layer_parameters_a = []
        layer_parameters_b = []

        assert output_size(image_resolution, 5, 2, 1) == output_size(image_resolution, 7, 3, 1)

        # Line detection layer
        self.conv1a = torch.nn.Conv2d(n_slices, 128, kernel_size=5, padding=2, stride=1)
        layer_parameters_a.append((5, 2, 1))
        self.maxpool1a = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        layer_parameters_a.append((2, 0, 2))
        self.dropout1a = torch.nn.Dropout(p=0.2)

        self.conv1b = torch.nn.Conv2d(n_slices, 128, kernel_size=7, padding=3, stride=1)
        layer_parameters_b.append((9, 4, 1))
        self.maxpool1b = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        layer_parameters_b.append((2, 0, 2))
        self.dropout1b = torch.nn.Dropout(p=0.2)

        # Higher level analysis of lines
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=8, padding=2, stride=2)
        layer_parameters_a.append((8, 2, 2))
        layer_parameters_b.append((8, 2, 2))
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        layer_parameters_a.append((2, 0, 2))
        layer_parameters_b.append((2, 0, 2))
        self.dropout2 = torch.nn.Dropout(p=0.2)

        # Even higher level analysis of lines
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=8, padding=4, stride=4)
        layer_parameters_a.append((8, 4, 4))
        layer_parameters_b.append((8, 4, 4))
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        layer_parameters_a.append((2, 0, 2))
        layer_parameters_b.append((2, 0, 2))
        self.dropout3 = torch.nn.Dropout(p=0.2)

        final_res_a = output_size_many(self.image_resolution, layer_parameters_a)
        final_res_b = output_size_many(self.image_resolution, layer_parameters_b)
        self.n_flattened = total_items_in_block(final_res_a, 256) + total_items_in_block(final_res_b, 256)

        self.fc1 = torch.nn.Linear(self.n_flattened, 2048)
        self.fc2 = torch.nn.Linear(2048, 512)
        self.fc3 = torch.nn.Linear(512, 128)
        self.fc4 = torch.nn.Linear(128, 4)

    def forward(self, x):
        xa = self.dropout1a(self.maxpool1a(F.relu(self.conv1a(x))))
        xb = self.dropout1b(self.maxpool1b(F.relu(self.conv1b(x))))
        xa = self.dropout2(self.maxpool2(F.relu(self.conv2(xa))))
        xb = self.dropout2(self.maxpool2(F.relu(self.conv2(xb))))
        xa = self.dropout3(self.maxpool3(F.relu(self.conv3(xa))))
        xb = self.dropout3(self.maxpool3(F.relu(self.conv3(xb))))
        x = torch.cat([xa.view(-1, self.n_flattened // 2), xb.view(-1, self.n_flattened // 2)], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


if __name__ == '__main__':

    batch_size = 32
    n_slices = 1
    image_res = 512
    model = SimpleConv2d(image_res)
    print(model.n_flattened)

    sim_data = Variable(torch.rand(batch_size, n_slices, image_res, image_res))
    output = model(sim_data)
    print(output)
