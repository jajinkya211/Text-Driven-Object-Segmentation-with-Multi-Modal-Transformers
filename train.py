"""
Training script for Referring Expression Segmentation

Features:
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Model checkpointing
- W&B logging (optional)
- Validation during training
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import argparse
from tqdm import tqdm
import numpy as np
from pathlib import Path

from config import Config, get_config_from_args
from models import ReferringSegmentationModel, build_model
from data import RefCOCODataset, build_dataloader
from data.transforms import get_train_transforms, get_val_transforms
from utils.losses import CombinedLoss, build_loss_function
from utils.metrics import MetricsTracker, compute_iou


class Trainer:
    """
    Training pipeline for referring segmentation

    Features:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Validation during training
    - Model checkpointing
    - Logging (W&B optional)
    """

    def __init__(self, config: Config):
        self.config = config

        # Set device
        self.device = torch.device(config.training.device)

        # Set random seed
        self._set_seed(config.seed)

        # Build model
        print("Building model...")
        self.model = ReferringSegmentationModel(config)
        self.model = self.model.to(self.device)

        print(f"Model has {self._count_parameters()} parameters")

        # Build datasets
        print("\nLoading datasets...")
        self.train_dataset = RefCOCODataset(
            data_root=config.data.data_root,
            dataset=config.data.dataset,
            split='train',
            split_by=config.data.split_by,
            image_size=config.data.image_size,
            max_text_length=config.data.max_text_length,
            transform=get_train_transforms(config.data.image_size, augment=True)
        )

        self.val_dataset = RefCOCODataset(
            data_root=config.data.data_root,
            dataset=config.data.dataset,
            split='val',
            split_by=config.data.split_by,
            image_size=config.data.image_size,
            max_text_length=config.data.max_text_length,
            transform=get_val_transforms(config.data.image_size)
        )

        # Build dataloaders
        self.train_loader = build_dataloader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory
        )

        self.val_loader = build_dataloader(
            self.val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory
        )

        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Val dataset: {len(self.val_dataset)} samples")

        # Build optimizer
        self.optimizer = self._build_optimizer()

        # Build loss function
        self.criterion = CombinedLoss(
            bce_weight=config.training.bce_weight,
            dice_weight=config.training.dice_weight
        )

        # Learning rate scheduler
        self.scheduler = self._build_scheduler()

        # Mixed precision scaler
        self.scaler = GradScaler() if config.training.use_amp else None

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_iou = 0.0

        # Checkpointing
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.use_wandb = config.training.use_wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.training.wandb_project,
                    name=config.experiment_name,
                    config=config.to_dict()
                )
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not available, skipping W&B logging")
                self.use_wandb = False

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)

    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _build_optimizer(self):
        """Build optimizer"""

        # Different learning rates for different components
        vision_params = list(self.model.vision_encoder.parameters())
        language_params = list(self.model.language_encoder.parameters())
        fusion_params = list(self.model.fusion.parameters())
        decoder_params = list(self.model.decoder.parameters())
        other_params = [
            p for n, p in self.model.named_parameters()
            if not any(x in n for x in ['vision_encoder', 'language_encoder', 'fusion', 'decoder'])
        ]

        param_groups = [
            {'params': vision_params, 'lr': self.config.training.learning_rate * 0.1},
            {'params': language_params, 'lr': self.config.training.learning_rate * 0.1},
            {'params': fusion_params, 'lr': self.config.training.learning_rate},
            {'params': decoder_params, 'lr': self.config.training.learning_rate},
            {'params': other_params, 'lr': self.config.training.learning_rate}
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )

        return optimizer

    def _build_scheduler(self):
        """Build learning rate scheduler"""

        if self.config.training.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )

        elif self.config.training.scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )

        elif self.config.training.scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )

        else:
            scheduler = None

        return scheduler

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch"""

        self.model.train()

        epoch_loss = 0
        epoch_iou = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.training.num_epochs}')

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            gt_masks = batch['mask'].to(self.device)

            # Forward pass with mixed precision
            if self.config.training.use_amp:
                with autocast():
                    pred_masks, _, _ = self.model(images, input_ids, attention_mask)
                    losses = self.criterion(pred_masks, gt_masks)
                    loss = losses['total']
            else:
                pred_masks, _, _ = self.model(images, input_ids, attention_mask)
                losses = self.criterion(pred_masks, gt_masks)
                loss = losses['total']

            # Normalize loss for gradient accumulation
            loss = loss / self.config.training.accumulation_steps

            # Backward pass
            if self.config.training.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.training.accumulation_steps == 0:
                if self.config.training.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Compute IoU for monitoring
            with torch.no_grad():
                iou = compute_iou(pred_masks, gt_masks).mean()

            # Track metrics
            epoch_loss += loss.item() * self.config.training.accumulation_steps
            epoch_iou += iou.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item() * self.config.training.accumulation_steps:.4f}',
                'iou': f'{iou.item() * 100:.2f}%'
            })

            # Log to wandb
            if self.use_wandb and batch_idx % self.config.training.log_frequency == 0:
                self.wandb.log({
                    'train/loss': loss.item() * self.config.training.accumulation_steps,
                    'train/bce': losses.get('bce', 0).item(),
                    'train/dice': losses.get('dice', 0).item(),
                    'train/iou': iou.item() * 100,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'step': self.global_step
                })

            self.global_step += 1

        # Epoch metrics
        avg_loss = epoch_loss / num_batches
        avg_iou = epoch_iou / num_batches

        return {
            'loss': avg_loss,
            'iou': avg_iou * 100
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """Validation"""

        self.model.eval()

        metrics_tracker = MetricsTracker()

        pbar = tqdm(self.val_loader, desc='Validation')

        for batch in pbar:
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            gt_masks = batch['mask'].to(self.device)

            # Forward
            pred_masks, _, _ = self.model(images, input_ids, attention_mask)

            # Update metrics
            metrics_tracker.update(pred_masks, gt_masks)

        # Compute final metrics
        metrics = metrics_tracker.compute()

        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_iou': self.best_iou,
            'config': self.config.to_dict()
        }

        # Save regular checkpoint
        if epoch % self.config.training.save_frequency == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

        # Always save last checkpoint
        last_path = self.checkpoint_dir / 'last_checkpoint.pth'
        torch.save(checkpoint, last_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_iou = checkpoint['best_iou']

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self):
        """Full training loop"""

        print(f"\n{'='*60}")
        print(f"Starting training: {self.config.experiment_name}")
        print(f"{'='*60}\n")

        for epoch in range(self.current_epoch + 1, self.config.training.num_epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Learning rate step
            if self.scheduler:
                if self.config.training.scheduler_type == 'plateau':
                    self.scheduler.step(val_metrics['mean_iou'])
                else:
                    self.scheduler.step()

            # Print results
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train IoU:  {train_metrics['iou']:.2f}%")
            print(f"  Val IoU:    {val_metrics['mean_iou']:.2f}%")
            print(f"  Val Dice:   {val_metrics['mean_dice']:.2f}%")
            print(f"  P@0.5:      {val_metrics['prec@0.5']:.2f}%")

            # Log to wandb
            if self.use_wandb:
                self.wandb.log({
                    'val/mean_iou': val_metrics['mean_iou'],
                    'val/mean_dice': val_metrics['mean_dice'],
                    'val/prec@0.5': val_metrics['prec@0.5'],
                    'val/prec@0.7': val_metrics['prec@0.7'],
                    'val/prec@0.9': val_metrics['prec@0.9'],
                    'epoch': epoch
                })

            # Save checkpoint
            is_best = val_metrics['mean_iou'] > self.best_iou

            if is_best:
                self.best_iou = val_metrics['mean_iou']
                print(f"  New best model! IoU: {self.best_iou:.2f}%")

            self.save_checkpoint(epoch, is_best=is_best)

            self.current_epoch = epoch

        print(f"\n{'='*60}")
        print(f"Training complete! Best IoU: {self.best_iou:.2f}%")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Referring Segmentation Model')

    # Data
    parser.add_argument('--data_root', type=str, required=True, help='Path to RefCOCO data')
    parser.add_argument('--dataset', type=str, default='refcoco', choices=['refcoco', 'refcoco+', 'refcocog'])

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')

    # Experiment
    parser.add_argument('--experiment_name', type=str, default='refcoco_baseline')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    # Resume
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Build config
    config = get_config_from_args(args)

    # Print config
    print(config)

    # Create trainer
    trainer = Trainer(config)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
