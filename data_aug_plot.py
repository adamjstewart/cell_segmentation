#!/usr/bin/env python3

import os.path

import matplotlib.pyplot as plt
import numpy as np

from datasets.slam import SLAM
from transforms import transforms


# Load image
dataset = SLAM('data/SLAM')
data, target = dataset[1]

# Add grid lines (https://stackoverflow.com/a/20473192/5828163)
dx, dy = 80, 80
data[:, ::dy] = 0
data[::dx, :] = 0
target[:, ::dy] = 1
target[::dx, :] = 1

transform_list = [
    transforms.Original(),
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

# Plot transformations
for transform in transform_list:
    data, target = transform(data, target)

    if isinstance(data, np.ndarray):
        img = data[..., 0]
    else:
        img = data[0]

    plt.figure()
    plt.imshow(img)
    plt.savefig(os.path.expanduser(
        '~/Desktop/{}.png'.format(transform.__class__.__name__)),
        dpi=600, bbox_inches='tight', pad_inches=0)
