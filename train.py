#!/usr/bin/env python3

"""Training script to train the u-net."""

import argparse
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose

from datasets import get_dataset
from models.unet import UNet


def set_up_parser():
    """Set up the argument parser.

    Returns:
        argparse.ArgumentParser: the argument parser
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

    group1 = parser.add_argument_group(title='dataset')
    group1.add_argument(
        '-d', '--dataset', default='ssTEM', choices=['ssTEM', 'SLAM'],
        help='dataset to train on')
    group1.add_argument(
        '-r', '--root', default='data',
        help='root data directory', metavar='DIR')

    group2 = parser.add_argument_group(title='hyperparameters')
    group2.add_argument(
        '-d', '--depth', default=4, type=int,
        help='depth of the u-net')
    group2.add_argument(
        '-e', '--epochs', default=1000, type=int,
        help='number of training epochs')
    group2.add_argument(
        '-b', '--batch-size', default=1, type=int,
        help='mini-batch size', metavar='SIZE')
    group2.add_argument(
        '-o', '--optimizer', default='sgd',
        choices=['adagrad', 'adam', 'rmsprop', 'sgd'], help='optimizer')
    group2.add_argument(
        '-l', '--learning-rate', default=0.01, type=float,
        help='learning rate', metavar='RATE')
    group2.add_argument(
        '-m', '--momentum', default=0.99, type=float, help='momentum factor')
    group2.add_argument(
        '-s', '--step-size', default=250, type=int,
        help='period of learning rate decay', metavar='SIZE')

    group3 = parser.add_argument_group(title='utility flags')
    group3.add_argument(
        '--seed', default=1, type=int,
        help='seed for random number generation')
    group3.add_argument(
        '-h', '--help', action='help', default=argparse.SUPPRESS,
        help='show this help message and exit')

    return parser


if __name__ == '__main__':
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()
    print('\nHyperparameters:', args)

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Data augmentation
    transform = Compose([
        transforms.ToTensor()
    ])
    target_transform = None

    # Data loaders
    num_workers = min(args.batch_size, multiprocessing.cpu_count())
    train_dataset = get_dataset(
        args.dataset, args.root, train=True,
        transform=transform, target_transform=target_transform)
    train_loader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, num_workers=num_workers)

    # Model
    model = UNet(depth=args.depth)

    # Checkpointing
    # TODO

    # Send to the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    params = model.parameters()
    if args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(params, args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, args.learning_rate)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(
            params, args.learning_rate, momentum=args.momentum)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            params, args.learning_rate, momentum=args.momentum)

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size)

    # For each epoch...
    for epoch in range(1, args.epochs + 1):
        print('\nEpoch:', epoch)
        scheduler.step()

        print('\nTraining...\n')
        model.train()

        # For each mini-batch...
        for batch, data, labels in enumerate(train_loader, 1):
            if batch % 100 == 0:
                print('Batch:', batch)

            # Send to the GPU
            data = data.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Calculate gradients
            loss.backward()

            # Update parameters
            optimizer.step()
