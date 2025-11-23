"""
Evaluation metrics for referring expression segmentation
"""
import torch
import numpy as np
from typing import Optional, Tuple


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU)

    Args:
        pred: [B, 1, H, W] or [B, H, W] predictions
        target: [B, 1, H, W] or [B, H, W] ground truth
        threshold: Binarization threshold for predictions
        eps: Small epsilon to avoid division by zero

    Returns:
        iou: [B] IoU scores for each sample
    """

    # Ensure same shape
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)

    # Binarize predictions
    pred_binary = (pred > threshold).float()
    target_binary = target.float()

    # Compute intersection and union
    intersection = (pred_binary * target_binary).sum(dim=(1, 2))
    union = pred_binary.sum(dim=(1, 2)) + target_binary.sum(dim=(1, 2)) - intersection

    # IoU
    iou = (intersection + eps) / (union + eps)

    return iou


def compute_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Dice coefficient (F1 score)

    Args:
        pred: [B, 1, H, W] or [B, H, W] predictions
        target: [B, 1, H, W] or [B, H, W] ground truth
        threshold: Binarization threshold
        eps: Small epsilon

    Returns:
        dice: [B] Dice scores
    """

    # Ensure same shape
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)

    # Binarize
    pred_binary = (pred > threshold).float()
    target_binary = target.float()

    # Compute intersection
    intersection = (pred_binary * target_binary).sum(dim=(1, 2))
    cardinality = pred_binary.sum(dim=(1, 2)) + target_binary.sum(dim=(1, 2))

    # Dice
    dice = (2.0 * intersection + eps) / (cardinality + eps)

    return dice


def compute_precision_recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute precision and recall

    Args:
        pred: [B, 1, H, W] or [B, H, W] predictions
        target: [B, 1, H, W] or [B, H, W] ground truth
        threshold: Binarization threshold
        eps: Small epsilon

    Returns:
        precision: [B] Precision scores
        recall: [B] Recall scores
    """

    # Ensure same shape
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)

    # Binarize
    pred_binary = (pred > threshold).float()
    target_binary = target.float()

    # True positives, false positives, false negatives
    tp = (pred_binary * target_binary).sum(dim=(1, 2))
    fp = (pred_binary * (1 - target_binary)).sum(dim=(1, 2))
    fn = ((1 - pred_binary) * target_binary).sum(dim=(1, 2))

    # Precision and recall
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return precision, recall


def compute_precision_at_k(
    ious: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute Precision@K (percentage of samples with IoU > threshold)

    Args:
        ious: [N] IoU scores
        threshold: IoU threshold

    Returns:
        Precision@K as percentage (0-100)
    """

    precision = (ious > threshold).float().mean() * 100
    return precision.item()


def compute_mean_iou(
    ious: torch.Tensor
) -> float:
    """
    Compute mean IoU

    Args:
        ious: [N] IoU scores

    Returns:
        Mean IoU
    """

    return ious.mean().item()


class MetricsTracker:
    """
    Tracks metrics during training/evaluation

    Accumulates predictions and computes aggregate metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.ious = []
        self.dices = []
        self.precisions = []
        self.recalls = []

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5
    ):
        """
        Update metrics with new batch

        Args:
            pred: [B, 1, H, W] predictions
            target: [B, 1, H, W] ground truth
            threshold: Binarization threshold
        """

        # Compute metrics
        iou = compute_iou(pred, target, threshold)
        dice = compute_dice(pred, target, threshold)
        precision, recall = compute_precision_recall(pred, target, threshold)

        # Store
        self.ious.append(iou.cpu())
        self.dices.append(dice.cpu())
        self.precisions.append(precision.cpu())
        self.recalls.append(recall.cpu())

    def compute(self) -> dict:
        """
        Compute aggregate metrics

        Returns:
            Dictionary of metrics
        """

        # Concatenate all batches
        ious = torch.cat(self.ious)
        dices = torch.cat(self.dices)
        precisions = torch.cat(self.precisions)
        recalls = torch.cat(self.recalls)

        # Compute statistics
        metrics = {
            'mean_iou': ious.mean().item() * 100,
            'median_iou': ious.median().item() * 100,
            'mean_dice': dices.mean().item() * 100,
            'mean_precision': precisions.mean().item() * 100,
            'mean_recall': recalls.mean().item() * 100,
            'prec@0.5': compute_precision_at_k(ious, 0.5),
            'prec@0.6': compute_precision_at_k(ious, 0.6),
            'prec@0.7': compute_precision_at_k(ious, 0.7),
            'prec@0.8': compute_precision_at_k(ious, 0.8),
            'prec@0.9': compute_precision_at_k(ious, 0.9),
        }

        # F1 score
        if metrics['mean_precision'] + metrics['mean_recall'] > 0:
            metrics['f1'] = (
                2 * metrics['mean_precision'] * metrics['mean_recall'] /
                (metrics['mean_precision'] + metrics['mean_recall'])
            )
        else:
            metrics['f1'] = 0.0

        return metrics

    def get_summary_string(self) -> str:
        """Get formatted summary of metrics"""

        metrics = self.compute()

        summary = [
            "Evaluation Results:",
            f"  Mean IoU:        {metrics['mean_iou']:.2f}%",
            f"  Median IoU:      {metrics['median_iou']:.2f}%",
            f"  Mean Dice:       {metrics['mean_dice']:.2f}%",
            f"  Mean Precision:  {metrics['mean_precision']:.2f}%",
            f"  Mean Recall:     {metrics['mean_recall']:.2f}%",
            f"  F1 Score:        {metrics['f1']:.2f}%",
            "",
            "Precision @ IoU thresholds:",
            f"  P@0.5:           {metrics['prec@0.5']:.2f}%",
            f"  P@0.6:           {metrics['prec@0.6']:.2f}%",
            f"  P@0.7:           {metrics['prec@0.7']:.2f}%",
            f"  P@0.8:           {metrics['prec@0.8']:.2f}%",
            f"  P@0.9:           {metrics['prec@0.9']:.2f}%",
        ]

        return '\n'.join(summary)


def compute_boundary_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    dilation: int = 2
) -> float:
    """
    Compute boundary accuracy

    Measures accuracy near object boundaries.

    Args:
        pred: [B, 1, H, W] predictions
        target: [B, 1, H, W] ground truth
        threshold: Binarization threshold
        dilation: Boundary dilation size

    Returns:
        Boundary accuracy
    """

    import torch.nn.functional as F

    # Binarize
    pred_binary = (pred > threshold).float()

    # Extract boundaries using morphological operations
    kernel_size = 2 * dilation + 1

    # Target boundary
    target_dilated = F.max_pool2d(
        target,
        kernel_size=kernel_size,
        stride=1,
        padding=dilation
    )

    target_eroded = -F.max_pool2d(
        -target,
        kernel_size=kernel_size,
        stride=1,
        padding=dilation
    )

    target_boundary = (target_dilated - target_eroded) > 0

    # Compute accuracy on boundary pixels
    boundary_pixels = target_boundary.sum()

    if boundary_pixels > 0:
        correct = ((pred_binary == target) * target_boundary).sum()
        accuracy = (correct / boundary_pixels).item()
    else:
        accuracy = 0.0

    return accuracy


def evaluate_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> dict:
    """
    Evaluate a single batch comprehensively

    Args:
        pred: [B, 1, H, W] predictions
        target: [B, 1, H, W] ground truth
        threshold: Binarization threshold

    Returns:
        Dictionary of metrics
    """

    iou = compute_iou(pred, target, threshold)
    dice = compute_dice(pred, target, threshold)
    precision, recall = compute_precision_recall(pred, target, threshold)

    metrics = {
        'iou': iou.mean().item() * 100,
        'dice': dice.mean().item() * 100,
        'precision': precision.mean().item() * 100,
        'recall': recall.mean().item() * 100,
    }

    # F1
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = (
            2 * metrics['precision'] * metrics['recall'] /
            (metrics['precision'] + metrics['recall'])
        )
    else:
        metrics['f1'] = 0.0

    return metrics
