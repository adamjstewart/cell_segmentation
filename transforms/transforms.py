from collections.abc import Sequence
import random

import cv2
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
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
        # Need to swap channel axis for img
        img = torchvision.transforms.functional.to_tensor(img)
        target = torch.from_numpy(target)

        return img, target


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
        img = torchvision.transforms.functional.normalize(
            img, self.mean, self.std)

        return img, target


class MinMaxScaling:

    def __call__(self, img, target):
        mins = np.amin(img, axis=(0, 1))
        maxs = np.amax(img, axis=(0, 1))

        img = (img - mins) / (maxs - mins)

        return img, target


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
        img = np.pad(
            img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            self.padding_mode)

        return img, target


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

        img = skimage.transform.rotate(
            img, angle, mode='reflect', preserve_range=True)
        target = skimage.transform.rotate(
            target, angle, mode='reflect', preserve_range=True)

        # Cast back to original dtypes
        img = img.astype(np.float32)
        target = target.round().astype(np.int64)

        return img, target


class RandomElasticDeformation:

    def __init__(self, alpha, sigma, alpha_affine):
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine

    def __call__(self, img, target):
        # Transform img and target at the same time to ensure same deformation
        stack = np.dstack([img, target])
        stack = self.elastic_transform(stack)
        img, target = stack[..., :-1], stack[..., -1]

        # Cast target back to original dtype
        target = target.round().astype(np.int64)

        return img, target

    def elastic_transform(self, image, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.

        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis",
           in Proc. of the International Conference on Document Analysis and
           Recognition, 2003.

        Based on:
            https://www.kaggle.com/bguberfain/
                elastic-transform-for-data-augmentation
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        shape_size = shape[:2]

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size
        ])
        pts2 = pts1 + random_state.uniform(
            -self.alpha_affine, self.alpha_affine, size=pts1.shape
        ).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(
            image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), self.sigma) * self.alpha

        x, y, z = np.meshgrid(
            np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = (
            np.reshape(y + dy, (-1, 1)),
            np.reshape(x + dx, (-1, 1)),
            np.reshape(z, (-1, 1))
        )

        return map_coordinates(
            image, indices, order=1, mode='reflect').reshape(shape)


class Original:
    """Identity mapping for plotting purposes."""

    def __call__(self, img, target):
        return img, target
