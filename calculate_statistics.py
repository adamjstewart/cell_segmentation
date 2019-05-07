#!/usr/bin/env python3

import numpy as np
import os

from datasets.slam import SLAM


dataset = SLAM(os.path.join('data', 'SLAM'), train=True)

running_mean = np.zeros(4)
running_median = np.zeros(4)
running_std = np.zeros(4)
running_weight = np.zeros(2)

for data, target in dataset:
    running_mean += np.mean(data, (0, 1))
    running_median += np.median(data, (0, 1))
    running_std += np.std(data, (0, 1))
    running_weight += np.bincount(target.flatten()) / target.size

mean = running_mean / len(dataset)
median = running_median / len(dataset)
std = running_std / len(dataset)
weight = 1 - running_weight / len(dataset)

print('mean=[{}, {}, {}, {}],'.format(*mean))
print('median=[{}, {}, {}, {}],'.format(*median))
print('std=[{}, {}, {}, {}]'.format(*std))
print('weight=torch.Tensor([{}, {}])'.format(*weight))
