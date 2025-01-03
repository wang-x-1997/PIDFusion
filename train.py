#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image Fusion Network Training Script
This script implements training pipeline for an image fusion network.
"""

import argparse
import os
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import kornia
import numpy as np

from dataset import fusiondata
from net import Net
from loss import SimMaxLoss, SimMinLoss


class Config:
    """Configuration class to store all training parameters."""

    def __init__(self):
        self.parser = self._create_parser()
        self.args = None

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with training parameters."""
        parser = argparse.ArgumentParser(description='Image Fusion Network Training')

        # Dataset parameters
        parser.add_argument('--dataset', type=str, default='data', help='Dataset path')
        parser.add_argument('--input_nc', type=int, default=1, help='Input image channels')
        parser.add_argument('--output_nc', type=int, default=1, help='Output image channels')

        # Training parameters
        parser.add_argument('--batchSize', type=int, default=4, help='Training batch size')
        parser.add_argument('--testBatchSize', type=int, default=1, help='Testing batch size')
        parser.add_argument('--nEpochs', type=int, default=100, help='Number of epochs')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
        parser.add_argument('--alpha', type=float, default=0.25, help='Alpha parameter for loss')
        parser.add_argument('--lamb', type=int, default=150, help='Lambda weight for L1 loss')

        # Network parameters
        parser.add_argument('--ngf', type=int, default=64, help='Generator filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='Discriminator filters in first conv layer')

        # System parameters
        parser.add_argument('--cuda', action='store_true', help='Use CUDA')
        parser.add_argument('--threads', type=int, default=0, help='Number of threads for data loader')
        parser.add_argument('--seed', type=int, default=123, help='Random seed')
        parser.add_argument('--ema_decay', type=float, default=0.9, help='EMA decay rate')

        return parser

    def parse_args(self):
        """Parse command line arguments."""
        self.args = self.parser.parse_args()
        return self.args


class Trainer:
    """Main trainer class implementing the training pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.device = self._setup_device()
        self.model = self._setup_model()
        self.criterion = self._setup_criterion()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.dataloader = self._setup_dataloader()
        self.loss_history = {'train': []}

    def _setup_device(self) -> torch.device:
        """Setup CUDA device if available."""
        use_cuda = not self.config.args.cuda and torch.cuda.is_available()

        if self.config.args.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        # Set random seeds
        torch.manual_seed(self.config.args.seed)
        if use_cuda:
            torch.cuda.manual_seed(self.config.args.seed)

        cudnn.benchmark = True
        return torch.device("cuda" if use_cuda else "cpu")

    def _setup_model(self) -> nn.Module:
        """Initialize and setup the network model."""
        model = Net().to(self.device)
        return model

    def _setup_criterion(self) -> List:
        """Setup loss functions."""
        return [
            SimMaxLoss(metric='cos', alpha=self.config.args.alpha).cuda(),
            SimMinLoss(metric='cos').cuda(),
            SimMaxLoss(metric='cos', alpha=self.config.args.alpha).cuda()
        ]

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup Adam optimizer."""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.args.lr,
            betas=(0.9, 0.999)
        )

    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        return optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.5
        )

    def _setup_dataloader(self) -> DataLoader:
        """Setup data loader."""
        root_path = "data/"
        dataset = fusiondata(os.path.join(root_path, self.config.args.dataset))
        return DataLoader(
            dataset=dataset,
            num_workers=self.config.args.threads,
            batch_size=self.config.args.batchSize,
            shuffle=True
        )

    def train_epoch(self, epoch: int) -> None:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []

        for batch_idx, (imgA, imgB) in enumerate(self.dataloader, 1):
            # Normalize images to [0,1] range and move to device
            imgA = (imgA / 255).to(self.device)
            imgB = (imgB / 255).to(self.device)

            # Forward pass
            fused_image, loss1 = self.model(imgA, imgB)

            # Calculate total loss
            loss = (torch.norm(fused_image - imgA, p=1) +
                    100 * kornia.losses.SSIMLoss(3, reduction='mean')(fused_image, imgB) +
                    5 * loss1)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            with torch.autograd.detect_anomaly():
                loss.backward()
            self.optimizer.step()

            # Record loss
            epoch_losses.append(loss.item())
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

            # Save checkpoint every 100 epochs
            if epoch % 100 == 0:
                self._save_checkpoint(epoch)

        # Record average epoch loss
        avg_loss = np.mean(epoch_losses)
        self.loss_history['train'].append(avg_loss)

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint_path = f"./checkpoint_epoch_{epoch}.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def train(self) -> None:
        """Main training loop."""
        print("Starting training...")
        for epoch in range(1, self.config.args.nEpochs + 1):
            self.train_epoch(epoch)
            self.scheduler.step()
            print(f'Completed epoch {epoch}/{self.config.args.nEpochs}')


def main():
    """Main function."""
    # Initialize configuration
    config = Config()
    args = config.parse_args()
    print(args)

    # Initialize trainer
    trainer = Trainer(config)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()