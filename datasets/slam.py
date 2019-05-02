from __future__ import print_function

import os

import numpy as np
from skimage import io
import torch.utils.data as data


class SLAM(data.Dataset):
    """Dataset for `SLAM: Simultaneous Label-free
    Autofluorescence-Multiharmonic microscopy
    <https://www.nature.com/articles/s41467-018-04470-8>`_.

    Args:
        root (string): Root directory where dataset exists.
        transform (callable, optional): A function/transform that takes in a
            numpy array and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
    """
    def __init__(self, root, transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.data = []
        self.labels = []

        for filename in os.listdir(os.path.join(self.root, 'Segmented')):
            self.labels.append(io.imread(os.path.join(
                self.root, 'Segmented', filename)))

            raw = filename.replace('_Segmented.png', '.tif')

            data = []
            for channel in ['THG', '3PF', 'SHG', '2PF']:
                data.append(io.imread(os.path.join(
                    self.root, 'Macrophage4channels', channel, raw)))

            self.data.append(np.dstack(data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]

        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
