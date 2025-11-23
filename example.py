"""
Quick example script demonstrating the full pipeline

This script shows how to:
1. Create a simple config
2. Build the model
3. Create a dummy dataset
4. Train for a few iterations
5. Run inference

Note: This uses dummy data for demonstration.
For real training, use RefCOCO dataset with train.py
"""
import torch
import numpy as np
from PIL import Image

from config import Config
from models import ReferringSegmentationModel
from inference import ReferringSegmentationInference


def create_dummy_data(batch_size=4):
    """Create dummy data for testing"""

    # Random images [B, 3, 480, 480]
    images = torch.randn(batch_size, 3, 480, 480)

    # Random masks [B, 1, 480, 480]
    masks = torch.randint(0, 2, (batch_size, 1, 480, 480)).float()

    # Random text tokens [B, 20]
    input_ids = torch.randint(0, 1000, (batch_size, 20))

    # Attention mask (all ones for dummy data)
    attention_mask = torch.ones(batch_size, 20)

    return {
        'image': images,
        'mask': masks,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


def example_model_creation():
    """Example: Create and inspect model"""

    print("=" * 60)
    print("Example 1: Model Creation")
    print("=" * 60)

    # Create config
    config = Config()

    # Build model
    model = ReferringSegmentationModel(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model, config


def example_forward_pass(model, config):
    """Example: Forward pass with dummy data"""

    print("\n" + "=" * 60)
    print("Example 2: Forward Pass")
    print("=" * 60)

    # Create dummy batch
    batch = create_dummy_data(batch_size=2)

    # Move to device (CPU for this example)
    device = torch.device('cpu')
    model = model.to(device)

    # Forward pass
    model.eval()

    with torch.no_grad():
        pred_masks, attn_weights, features = model(
            batch['image'].to(device),
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
            return_features=True
        )

    print(f"\nInput shapes:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Input IDs: {batch['input_ids'].shape}")
    print(f"  GT Masks: {batch['mask'].shape}")

    print(f"\nOutput shapes:")
    print(f"  Predicted Masks: {pred_masks.shape}")
    print(f"  Attention Weights: {attn_weights.shape}")

    print("\nForward pass successful!")

    return pred_masks


def example_loss_computation(pred_masks, gt_masks):
    """Example: Compute loss"""

    print("\n" + "=" * 60)
    print("Example 3: Loss Computation")
    print("=" * 60)

    from utils.losses import CombinedLoss

    # Create loss function
    criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0)

    # Compute loss
    losses = criterion(pred_masks, gt_masks)

    print(f"\nLosses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

    return losses['total']


def example_metrics(pred_masks, gt_masks):
    """Example: Compute metrics"""

    print("\n" + "=" * 60)
    print("Example 4: Metrics Computation")
    print("=" * 60)

    from utils.metrics import compute_iou, compute_dice

    # Compute metrics
    iou = compute_iou(pred_masks, gt_masks)
    dice = compute_dice(pred_masks, gt_masks)

    print(f"\nMetrics:")
    print(f"  Mean IoU: {iou.mean().item() * 100:.2f}%")
    print(f"  Mean Dice: {dice.mean().item() * 100:.2f}%")


def example_training_step():
    """Example: Single training step"""

    print("\n" + "=" * 60)
    print("Example 5: Training Step")
    print("=" * 60)

    # Setup
    config = Config()
    model = ReferringSegmentationModel(config)
    device = torch.device('cpu')
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Loss
    from utils.losses import CombinedLoss
    criterion = CombinedLoss()

    # Training mode
    model.train()

    # Get batch
    batch = create_dummy_data(batch_size=2)

    # Forward
    pred_masks, _, _ = model(
        batch['image'].to(device),
        batch['input_ids'].to(device),
        batch['attention_mask'].to(device)
    )

    # Compute loss
    losses = criterion(pred_masks, batch['mask'].to(device))
    loss = losses['total']

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"\nTraining step completed!")
    print(f"Loss: {loss.item():.4f}")


def example_save_load():
    """Example: Save and load checkpoint"""

    print("\n" + "=" * 60)
    print("Example 6: Save and Load Checkpoint")
    print("=" * 60)

    import tempfile
    import os

    # Create model
    config = Config()
    model = ReferringSegmentationModel(config)

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pth') as f:
        checkpoint_path = f.name

    # Save
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config.to_dict()
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved to: {checkpoint_path}")

    # Load
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Checkpoint loaded successfully!")

    # Cleanup
    os.remove(checkpoint_path)


def main():
    """Run all examples"""

    print("\n")
    print("*" * 60)
    print("REFERRING EXPRESSION SEGMENTATION - EXAMPLES")
    print("*" * 60)

    # Example 1: Model creation
    model, config = example_model_creation()

    # Example 2: Forward pass
    batch = create_dummy_data(batch_size=2)
    pred_masks = example_forward_pass(model, config)

    # Example 3: Loss computation
    loss = example_loss_computation(pred_masks, batch['mask'])

    # Example 4: Metrics
    example_metrics(pred_masks, batch['mask'])

    # Example 5: Training step
    example_training_step()

    # Example 6: Save/load
    example_save_load()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

    print("\n📝 Next Steps:")
    print("  1. Download RefCOCO dataset (see README.md)")
    print("  2. Train model: python train.py --data_root /path/to/data")
    print("  3. Evaluate: python evaluate.py --checkpoint checkpoints/best_model.pth")
    print("  4. Inference: python inference.py --checkpoint ... --image ... --text ...")
    print("")


if __name__ == '__main__':
    main()
