import os

import numpy as np
from skimage import color, io
import torch.utils.data as data


CHANNELS = ['THG', '3PF', 'SHG', '2PF']


class SLAM(data.Dataset):
    """Dataset for `SLAM: Simultaneous Label-free
    Autofluorescence-Multiharmonic microscopy
    <https://www.nature.com/articles/s41467-018-04470-8>`_.

    Args:
        root (string): Root directory where dataset exists.
        transform (callable, optional): A function/transform that takes in two
            numpy arrays and returns transformed versions.
    """
    def __init__(self, root, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform

        self.data = []
        self.labels = []

        for filename in os.listdir(os.path.join(self.root, 'Segmented')):
            # Load labels, convert to grayscale, and binarize
            labels = io.imread(os.path.join(self.root, 'Segmented', filename))
            labels = color.rgb2gray(labels)
            labels = (labels < 1).astype(np.uint8)
            self.labels.append(labels)

            # Load data and stack to 4-channel image
            raw = filename.replace('_Segmented.png', '.tif')
            data = []
            for channel in CHANNELS:
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
        target = self.labels[index]

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.data)
