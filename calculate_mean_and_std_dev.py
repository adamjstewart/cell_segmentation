#!/usr/bin/env python3

import numpy as np
import os

from datasets.slam import SLAM


dataset = SLAM(os.path.join('data', 'SLAM'))

running_mean = np.zeros(4)
running_std = np.zeros(4)

for data, _ in dataset:
    running_mean += np.mean(data, (0, 1))
    running_std += np.std(data, (0, 1))

mean = running_mean / len(dataset)
std = running_std / len(dataset)

print('mean=[{}, {}, {}, {}],'.format(*mean))
print('std=[{}, {}, {}, {}]'.format(*std))
