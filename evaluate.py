"""
Evaluation script for Referring Expression Segmentation

Evaluates a trained model on test sets and generates visualizations
"""
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from models import ReferringSegmentationModel, load_pretrained_model
from data import RefCOCODataset, build_dataloader
from data.transforms import get_val_transforms
from utils.metrics import MetricsTracker, evaluate_batch


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device='cuda',
    verbose=True
):
    """
    Evaluate model on dataset

    Args:
        model: Trained model
        dataloader: DataLoader
        device: Device to use
        verbose: Print results

    Returns:
        Dictionary of metrics
    """

    model.eval()
    metrics_tracker = MetricsTracker()

    pbar = tqdm(dataloader, desc='Evaluating') if verbose else dataloader

    for batch in pbar:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        gt_masks = batch['mask'].to(device)

        # Forward
        pred_masks, _, _ = model(images, input_ids, attention_mask)

        # Update metrics
        metrics_tracker.update(pred_masks, gt_masks)

    # Compute final metrics
    metrics = metrics_tracker.compute()

    if verbose:
        print("\n" + metrics_tracker.get_summary_string())

    return metrics


def visualize_predictions(
    model,
    dataset,
    num_samples=10,
    output_dir='./results',
    device='cuda'
):
    """
    Visualize predictions

    Args:
        model: Trained model
        dataset: Dataset
        num_samples: Number of samples to visualize
        output_dir: Output directory
        device: Device
    """

    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(indices):
        sample = dataset[idx]

        # Prepare input
        image = sample['image'].unsqueeze(0).to(device)
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        gt_mask = sample['mask']
        expression = sample['expression']

        # Predict
        with torch.no_grad():
            pred_mask, attn_weights, _ = model(image, input_ids, attention_mask)

        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        img_np = image[0].cpu() * std + mean
        img_np = img_np.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        # Convert masks to numpy
        gt_mask_np = gt_mask[0].cpu().numpy()
        pred_mask_np = pred_mask[0, 0].cpu().numpy()

        # Compute IoU
        iou = ((pred_mask_np > 0.5) & (gt_mask_np > 0.5)).sum() / \
              ((pred_mask_np > 0.5) | (gt_mask_np > 0.5)).sum()

        # Plot
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'Image\n"{expression}"', fontsize=8)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_mask_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth', fontsize=8)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_mask_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Prediction\nIoU: {iou:.3f}', fontsize=8)
        axes[i, 2].axis('off')

        # Overlay
        overlay = img_np.copy()
        overlay[pred_mask_np > 0.5] = overlay[pred_mask_np > 0.5] * 0.5 + np.array([1, 0, 0]) * 0.5

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay', fontsize=8)
        axes[i, 3].axis('off')

    plt.tight_layout()

    # Save
    output_path = output_dir / 'predictions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualizations to {output_path}")

    plt.close()


def visualize_attention(
    model,
    dataset,
    sample_idx=0,
    output_dir='./results',
    device='cuda'
):
    """
    Visualize attention weights

    Args:
        model: Trained model
        dataset: Dataset
        sample_idx: Sample index
        output_dir: Output directory
        device: Device
    """

    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get sample
    sample = dataset[sample_idx]

    # Prepare input
    image = sample['image'].unsqueeze(0).to(device)
    input_ids = sample['input_ids'].unsqueeze(0).to(device)
    attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
    expression = sample['expression']

    # Get tokenizer to decode tokens
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    tokens = [t for t, m in zip(tokens, attention_mask[0].cpu().numpy()) if m == 1]

    # Predict
    with torch.no_grad():
        pred_mask, attn_weights, _ = model(image, input_ids, attention_mask)

    # Attention weights: [B, num_heads, H*W, L]
    attn_weights = attn_weights[0].cpu().numpy()  # [num_heads, H*W, L]

    # Average across heads
    attn_avg = attn_weights.mean(axis=0)  # [H*W, L]

    # Reshape to spatial dimensions
    H = W = int(np.sqrt(attn_avg.shape[0]))
    attn_spatial = attn_avg.reshape(H, W, -1)  # [H, W, L]

    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img_np = image[0].cpu() * std + mean
    img_np = img_np.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    # Plot attention for top words
    num_words = min(6, len(tokens) - 2)  # Exclude [CLS] and [SEP]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(num_words):
        token_idx = i + 1  # Skip [CLS]

        attn_map = attn_spatial[:, :, token_idx]

        # Resize to image size
        from scipy.ndimage import zoom
        scale_h = img_np.shape[0] / attn_map.shape[0]
        scale_w = img_np.shape[1] / attn_map.shape[1]

        attn_map_resized = zoom(attn_map, (scale_h, scale_w), order=1)

        # Overlay attention on image
        axes[i].imshow(img_np)
        axes[i].imshow(attn_map_resized, alpha=0.5, cmap='jet')
        axes[i].set_title(f'"{tokens[token_idx]}"', fontsize=12)
        axes[i].axis('off')

    plt.suptitle(f'Attention Visualization\n"{expression}"', fontsize=14)
    plt.tight_layout()

    # Save
    output_path = output_dir / 'attention.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved attention visualization to {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Referring Segmentation Model')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config (optional)')

    # Data
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='refcoco')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'testA', 'testB', 'test'])

    # Evaluation
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')

    # Visualization
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--num_vis', type=int, default=10, help='Number of visualizations')
    parser.add_argument('--output_dir', type=str, default='./results')

    args = parser.parse_args()

    # Load config
    if args.config:
        config = Config.load(args.config)
    else:
        # Load from checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'config' in checkpoint:
            config = Config.from_dict(checkpoint['config'])
        else:
            # Use default config
            from config import get_default_config
            config = get_default_config()

    # Update data settings
    config.data.data_root = args.data_root
    config.data.dataset = args.dataset

    print("Loading model...")
    device = torch.device(args.device)

    model = load_pretrained_model(args.checkpoint, config, device)

    print("Loading dataset...")
    dataset = RefCOCODataset(
        data_root=args.data_root,
        dataset=args.dataset,
        split=args.split,
        split_by=config.data.split_by,
        image_size=config.data.image_size,
        max_text_length=config.data.max_text_length,
        transform=get_val_transforms(config.data.image_size)
    )

    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"Evaluating on {len(dataset)} samples from {args.dataset} {args.split}...\n")

    # Evaluate
    metrics = evaluate_model(model, dataloader, device)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f'results_{args.dataset}_{args.split}.txt'

    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results: {args.dataset} {args.split}\n")
        f.write("=" * 60 + "\n\n")

        for key, value in metrics.items():
            f.write(f"{key}: {value:.2f}\n")

    print(f"\nResults saved to {results_file}")

    # Visualizations
    if args.visualize:
        print("\nGenerating visualizations...")

        visualize_predictions(
            model,
            dataset,
            num_samples=args.num_vis,
            output_dir=output_dir,
            device=device
        )

        visualize_attention(
            model,
            dataset,
            sample_idx=0,
            output_dir=output_dir,
            device=device
        )


if __name__ == '__main__':
    main()
