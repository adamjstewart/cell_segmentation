from collections.abc import Sequence
import random

import numpy as np
import skimage.transform
import torch
import torchvision


class Compose(torchvision.transforms.Compose):

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)

        return img, target


class ToTensor(torchvision.transforms.ToTensor):

    def __call__(self, img, target):
        """
        Args:
            img (numpy.ndarray): Image to be converted to tensor.
            target (numpy.ndarray): Target to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return (
            torchvision.transforms.functional.to_tensor(img),
            torch.from_numpy(target)
        )


class Normalize(torchvision.transforms.Normalize):

    def __call__(self, img, target):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be normalized.
            target (Tensor): Tensor target of size (H, W).

        Returns:
            tuple: Normalized Tensor image and target
        """
        # Only normalize img, not target
        return (
            torchvision.transforms.functional.normalize(
                img, self.mean, self.std),
            target
        )


class Pad(torchvision.transforms.Pad):

    def __call__(self, img, target):
        """
        Args:
            img (numpy.ndarray): Image to be padded.
            target (numpy.ndarray): Target.
        Returns:
            tuple: Padded image and target.
        """
        if isinstance(self.padding, int):
            pad_left = pad_right = pad_top = pad_bottom = self.padding
        if isinstance(self.padding, Sequence) and len(self.padding) == 2:
            pad_left = pad_right = self.padding[0]
            pad_top = pad_bottom = self.padding[1]
        if isinstance(self.padding, Sequence) and len(self.padding) == 4:
            pad_left = self.padding[0]
            pad_top = self.padding[1]
            pad_right = self.padding[2]
            pad_bottom = self.padding[3]

        # Only pad img, not target
        return (
            np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                   self.padding_mode),
            target
        )


class RandomCrop(torchvision.transforms.RandomCrop):

    def __init__(self, img_size, target_size):
        self.img_size = img_size
        self.target_size = target_size

    def __call__(self, img, target):
        """
        Args:
            img (numpy.ndarray): Image to be cropped.
            target (numpy.ndarray): Target to be cropped.
        Returns:
            tuple: Cropped image and target.
        """
        ih, iw, _ = img.shape
        th, tw = target.shape

        i = random.randint(0, ih - self.img_size)
        j = random.randint(0, iw - self.img_size)

        img = img[i:i + self.img_size, j:j + self.img_size]

        # Crop target to corresponding area
        target = target[i:i + self.target_size, j:j + self.target_size]

        return img, target

    def __repr__(self):
        return self.__class__.__name__ + \
            '(img_size={0}, target_size={1})'.format(
                self.img_size, self.target_size)


class RandomHorizontalFlip(torchvision.transforms.RandomHorizontalFlip):

    def __call__(self, img, target):
        """
        Args:
            img (numpy.ndarray): Image to be flipped.
            target (numpy.ndarray): Target to be flipped.

        Returns:
            tuple: Randomly flipped images.
        """
        if random.random() < self.p:
            img, target = np.fliplr(img), np.fliplr(target)

        return img, target


class RandomVerticalFlip(torchvision.transforms.RandomVerticalFlip):

    def __call__(self, img, target):
        """
        Args:
            img (numpy.ndarray): Image to be flipped.
            target (numpy.ndarray): Target to be flipped.

        Returns:
            tuple: Randomly flipped images.
        """
        if random.random() < self.p:
            img, target = np.flipud(img), np.flipud(target)

        return img, target


class RandomRotation(torchvision.transforms.RandomRotation):

    def __call__(self, img, target):
        """
        Args:
            img (numpy.ndarray): Image to be rotated.
            target (numpy.ndarray): Target to be rotated.
        Returns:
            tuple: Rotated image and target.
        """
        angle = self.get_params(self.degrees)

        return (
            skimage.transform.rotate(img, angle, mode='reflect'),
            skimage.transform.rotate(target, angle, mode='reflect')
        )