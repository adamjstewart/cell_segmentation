#!/usr/bin/env python3

"""Training script to train the u-net."""

import argparse
import multiprocessing
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import get_dataset
from models.unet import UNet
from transforms import transforms


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
        '--dataset', default='ssTEM', choices=['ssTEM', 'SLAM'],
        help='dataset to train on')
    group1.add_argument(
        '-r', '--root', default='data',
        help='root data directory', metavar='DIR')

    group2 = parser.add_argument_group(title='hyperparameters')
    group2.add_argument(
        '-d', '--depth', default=4, type=int,
        help='depth of the u-net')
    group2.add_argument(
        '-p', '--dropout', default=0.5, type=float,
        help='dropout probability')
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
        '-c', '--checkpoint', action='store_true',
        help='load existing checkpoint if it exists')
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

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Data augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomElasticDeformation(
            alpha=400, sigma=10, alpha_affine=50),
        transforms.Pad(92, padding_mode='reflect'),
        transforms.RandomRotation(180),
        transforms.RandomCrop(572, 388),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[118.9064605468, 5.58606197916, 8.82765065104, 101.04520195],
            std=[65.0233221789, 7.73024044040, 8.4314033739, 47.8530152470]),
    ])

    # Data loaders
    num_workers = min(args.batch_size, multiprocessing.cpu_count())
    train_dataset = get_dataset(
        args.dataset, args.root, train=True, transform=transform)
    train_loader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, num_workers=num_workers)

    # Model
    model = UNet(depth=args.depth, p=args.dropout)

    # Loss criterion
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

    # Checkpointing
    epoch = 1
    checkpoint_file = os.path.join('checkpoints', 'model.pth')
    if args.checkpoint and os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']

    # Send to the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    # For each epoch...
    while epoch < args.epochs:
        print('\nEpoch:', epoch)
        scheduler.step()

        # For each mini-batch...
        for batch, data, labels in enumerate(train_loader, 1):
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

        epoch += 1

        # Checkpointing
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
        }, checkpoint_file)
