#!/usr/bin/env python3

import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.slam import SLAM, CHANNELS
from transforms import transforms


def plot_data_target(data, target, transform, savefig=False):
    print('\nTransform:', transform)

    if isinstance(data, np.ndarray):
        print('Data:', data.dtype, data.shape, np.amin(data), np.amax(data))
        print('Target:', target.dtype, target.shape,
              np.amin(target), np.amax(target))
    else:
        print('Data:', data.dtype, data.shape,
              torch.min(data).item(), torch.max(data).item())
        print('Target:', target.dtype, target.shape,
              torch.min(target).item(), torch.max(target).item())

    fig, axes = plt.subplots(2, 2)

    if isinstance(data, np.ndarray):
        data = np.moveaxis(data, -1, 0)

    for ax, img, title in zip(fig.axes, data, CHANNELS):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title)

    if transform:
        fig.suptitle(transform)

    if savefig:
        plt.savefig(os.path.expanduser(
            '~/Desktop/{}_data.png'.format(transform)), dpi=600)
    else:
        plt.show()

    if isinstance(data, np.ndarray):
        data = np.moveaxis(data, 0, -1)

    fig, ax = plt.subplots()

    ax.imshow(target)
    ax.axis('off')
    ax.set_title('Segmentation')

    if transform:
        fig.suptitle(transform)

    if savefig:
        plt.savefig(os.path.expanduser(
            '~/Desktop/{}_target.png'.format(transform)), dpi=600)
    else:
        plt.show()


# Load image
dataset = SLAM('data/SLAM')
data, target = dataset[1]

# Add grid lines (https://stackoverflow.com/a/20473192/5828163)
dx, dy = 80, 80
data[:, ::dy, :] = [0, 0, 0, 0]
data[::dx, :, :] = [0, 0, 0, 0]
target[:, ::dy] = 1
target[::dx, :] = 1

# Plot transformations
plot_data_target(data, target, 'Original')

transform_list = [
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomVerticalFlip(p=1),
    transforms.MinMaxScaling(),
    transforms.RandomElasticDeformation(
        alpha=200, sigma=10, alpha_affine=40),
    transforms.Pad(92, padding_mode='reflect'),
    transforms.RandomRotation(180),
    transforms.RandomCrop(572, 388),
    transforms.ToTensor(),
]

for transform in transform_list:
    data, target = transform(data, target)
    plot_data_target(data, target, transform.__class__.__name__)
