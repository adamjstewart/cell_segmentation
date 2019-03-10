import os

from .sstem import ssTEM
# from .slam import SLAM


def get_dataset(dataset, root, train=True,
                transform=None, target_transform=None):
    """Loads the requested dataset.

    Parameters:
        dataset (str): the requested dataset
        root (str): the root directory containing the dataset
        train (bool, optional): If True, creates dataset from training set,
            otherwise creates from test set.
        transform (callable, optional): A function/transform that takes in a
            PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

    Returns:
        torch.utils.data.Dataset: the requested dataset
    """
    if dataset == 'ssTEM':
        return ssTEM(os.path.join(root, dataset), train,
                     transform, target_transform, download=True)
    # elif dataset == 'SLAM':
    #     return SLAM(os.path.join(root, dataset), train,
    #                 transform, target_transform, download=True)
    else:
        raise ValueError("Unsupported dataset: '{}'".format(dataset))
