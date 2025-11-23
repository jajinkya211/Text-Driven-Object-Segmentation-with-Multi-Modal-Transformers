"""
Loss functions for referring expression segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation

    Dice coefficient measures overlap between prediction and ground truth.
    Dice Loss = 1 - Dice Coefficient

    Effective for handling class imbalance in segmentation tasks.

    Args:
        smooth: Smoothing factor to avoid division by zero
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss

        Args:
            pred: [B, 1, H, W] predictions (after sigmoid)
            target: [B, 1, H, W] ground truth binary masks

        Returns:
            Dice loss
        """

        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice loss
        loss = 1.0 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    Focal Loss down-weights easy examples and focuses on hard examples.

    Args:
        alpha: Weighting factor (0-1)
        gamma: Focusing parameter (typically 2)
        reduction: Reduction method
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss

        Args:
            pred: [B, 1, H, W] predictions (before sigmoid)
            target: [B, 1, H, W] ground truth

        Returns:
            Focal loss
        """

        # Apply sigmoid
        pred_sigmoid = torch.sigmoid(pred)

        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )

        # Compute focal weight
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        # Focal loss
        loss = alpha_t * focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class IoULoss(nn.Module):
    """
    IoU (Intersection over Union) Loss

    Directly optimizes IoU metric.

    Args:
        smooth: Smoothing factor
        reduction: Reduction method
    """

    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU loss

        Args:
            pred: [B, 1, H, W] predictions (after sigmoid)
            target: [B, 1, H, W] ground truth

        Returns:
            IoU loss
        """

        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

        # IoU
        iou = (intersection + self.smooth) / (union + self.smooth)

        # IoU loss
        loss = 1.0 - iou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for better boundary prediction

    Emphasizes errors near object boundaries.

    Args:
        theta: Boundary thickness
    """

    def __init__(self, theta: int = 3):
        super().__init__()
        self.theta = theta

    def _compute_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract boundary from mask using morphological operations"""

        # Simple boundary extraction using max pooling and subtraction
        kernel_size = 2 * self.theta + 1

        max_pool = F.max_pool2d(
            mask,
            kernel_size=kernel_size,
            stride=1,
            padding=self.theta
        )

        min_pool = -F.max_pool2d(
            -mask,
            kernel_size=kernel_size,
            stride=1,
            padding=self.theta
        )

        boundary = max_pool - min_pool

        return boundary

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary loss

        Args:
            pred: [B, 1, H, W] predictions
            target: [B, 1, H, W] ground truth

        Returns:
            Boundary loss
        """

        # Extract boundaries
        pred_boundary = self._compute_boundary(pred)
        target_boundary = self._compute_boundary(target)

        # BCE on boundaries
        loss = F.binary_cross_entropy(pred_boundary, target_boundary)

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function

    Combines multiple losses with configurable weights.
    Typically uses BCE + Dice for referring segmentation.

    Args:
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        iou_weight: Weight for IoU loss
        focal_weight: Weight for Focal loss
        boundary_weight: Weight for Boundary loss
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        iou_weight: float = 0.0,
        focal_weight: float = 0.0,
        boundary_weight: float = 0.0
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight

        # Initialize loss functions
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.iou_loss = IoULoss()
        self.focal_loss = FocalLoss()
        self.boundary_loss = BoundaryLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute combined loss

        Args:
            pred: [B, 1, H, W] predictions (after sigmoid)
            target: [B, 1, H, W] ground truth

        Returns:
            Dictionary with total loss and individual losses
        """

        losses = {}
        total_loss = 0

        # BCE Loss
        if self.bce_weight > 0:
            bce = self.bce_loss(pred, target)
            losses['bce'] = bce
            total_loss += self.bce_weight * bce

        # Dice Loss
        if self.dice_weight > 0:
            dice = self.dice_loss(pred, target)
            losses['dice'] = dice
            total_loss += self.dice_weight * dice

        # IoU Loss
        if self.iou_weight > 0:
            iou = self.iou_loss(pred, target)
            losses['iou'] = iou
            total_loss += self.iou_weight * iou

        # Focal Loss (requires logits, so skip if pred is after sigmoid)
        if self.focal_weight > 0:
            # Note: This assumes pred is logits, not after sigmoid
            # If pred is after sigmoid, this will not work correctly
            focal = self.focal_loss(pred, target)
            losses['focal'] = focal
            total_loss += self.focal_weight * focal

        # Boundary Loss
        if self.boundary_weight > 0:
            boundary = self.boundary_loss(pred, target)
            losses['boundary'] = boundary
            total_loss += self.boundary_weight * boundary

        losses['total'] = total_loss

        return losses


def build_loss_function(
    loss_type: str = 'combined',
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    **kwargs
) -> nn.Module:
    """
    Factory function to build loss function

    Args:
        loss_type: Type of loss ('bce', 'dice', 'iou', 'focal', 'combined')
        bce_weight: Weight for BCE in combined loss
        dice_weight: Weight for Dice in combined loss
        **kwargs: Additional arguments

    Returns:
        Loss function module
    """

    if loss_type == 'bce':
        return nn.BCELoss()

    elif loss_type == 'dice':
        return DiceLoss(**kwargs)

    elif loss_type == 'iou':
        return IoULoss(**kwargs)

    elif loss_type == 'focal':
        return FocalLoss(**kwargs)

    elif loss_type == 'combined':
        return CombinedLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
