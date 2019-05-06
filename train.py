#!/usr/bin/env python3

"""Training script to train the u-net."""

import argparse
import multiprocessing
import os

from sklearn import metrics
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
        '--dataset', default='SLAM', choices=['ssTEM', 'SLAM'],
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
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomElasticDeformation(
            alpha=400, sigma=10, alpha_affine=50),
        transforms.Pad(92, padding_mode='reflect'),
        transforms.RandomRotation(180),
        transforms.RandomCrop(572, 388),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[106.224838055, 5.2348620833, 7.62163486111, 86.974367638],
            std=[59.419128849, 7.1664727925, 7.462212191, 43.3211457393]),
    ])
    test_transform = transforms.Compose([
        transforms.Pad(92, padding_mode='reflect'),
        transforms.RandomCrop(572, 388),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[106.224838055, 5.2348620833, 7.62163486111, 86.974367638],
            std=[59.419128849, 7.1664727925, 7.462212191, 43.3211457393]),
    ])

    # Data loaders
    num_workers = min(args.batch_size, multiprocessing.cpu_count())
    train_dataset = get_dataset(
        args.dataset, args.root, train=True, transform=train_transform)
    test_dataset = get_dataset(
        args.dataset, args.root, train=False, transform=test_transform)
    train_loader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset, args.batch_size, shuffle=True, num_workers=num_workers)

    # Model
    model = UNet(in_channels=4, depth=args.depth, p=args.dropout)

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
    train_loss = []
    train_iou = []
    test_iou = []
    checkpoint_file = os.path.join('checkpoints', 'model.pth')
    if args.checkpoint and os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        train_iou = checkpoint['train_iou']
        test_iou = checkpoint['test_iou']

    # Send to the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # For each epoch...
    running_loss = 0
    running_train_iou = 0
    while epoch < args.epochs:
        print('\nEpoch:', epoch)

        # Training
        model.train()
        scheduler.step()

        # For each mini-batch...
        for data, labels in train_loader:
            # Send to the GPU
            data = data.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)
            predictions = torch.argmax(outputs, 1)
            running_train_iou += metrics.jaccard_similarity_score(
                labels.numpy().flatten(), predictions.numpy().flatten())

            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate gradients
            loss.backward()

            # Update parameters
            optimizer.step()

        epoch += 1

        if epoch % 10 == 1:
            train_loss.append(running_loss / 10 / len(train_loader))
            train_iou.append(running_train_iou / 10 / len(train_loader))

            running_loss = 0
            running_train_iou = 0

            print('Loss: {:.3f}'.format(train_loss[-1]))
            print('Train IoU: {:.3f}'.format(train_iou[-1]))

            # Testing
            model.eval()
            running_test_iou = 0
            with torch.no_grad():
                for data, labels in test_loader:
                    # Send to the GPU
                    data = data.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = model(data)
                    predictions = torch.argmax(outputs, 1)
                    running_test_iou += metrics.jaccard_similarity_score(
                        labels.numpy().flatten(),
                        predictions.numpy().flatten())

            test_iou.append(running_test_iou / len(test_loader))

            print('Test IoU: {:.3f}'.format(test_iou[-1]))

            # Checkpointing
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'train_iou': train_iou,
                'test_iou': test_iou,
            }, checkpoint_file)
