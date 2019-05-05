#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from datasets.slam import SLAM, CHANNELS
from transforms import transforms


np.set_printoptions(precision=2)


def plot_data_target(data, target, transform=None):
    print(transform)

    fig, axes = plt.subplots(2, 2)

    if isinstance(data, np.ndarray):
        data = np.moveaxis(data, -1, 0)

    for ax, img, title in zip(fig.axes, data, CHANNELS):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title)

    if transform:
        fig.suptitle(transform)

    plt.show()

    if isinstance(data, np.ndarray):
        data = np.moveaxis(data, 0, -1)

    fig, ax = plt.subplots()

    ax.imshow(target)
    ax.axis('off')
    ax.set_title('Segmentation')

    if transform:
        fig.suptitle(transform)

    plt.show()


dataset = SLAM('data/SLAM')
data, target = dataset[1]

plot_data_target(data, target, 'Original')

transform_list = [
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomVerticalFlip(p=1),
    transforms.Pad(92, padding_mode='reflect'),
    transforms.RandomRotation(180),
    transforms.RandomCrop(572, 388),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[118.9064605468, 5.58606197916, 8.82765065104, 101.04520195],
        std=[65.0233221789, 7.73024044040, 8.4314033739, 47.8530152470]),
]

for transform in transform_list:
    data, target = transform(data, target)
    plot_data_target(data, target, transform.__class__.__name__)
