"""
Inference script for Referring Expression Segmentation

Run inference on custom images with text descriptions
"""
import torch
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import Config
from models import load_pretrained_model
from transformers import BertTokenizer


class ReferringSegmentationInference:
    """
    Inference wrapper for referring segmentation

    Easy-to-use interface for running inference on images with text queries.
    """

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Initialize inference

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use
        """

        self.device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load config
        if 'config' in checkpoint:
            self.config = Config.from_dict(checkpoint['config'])
        else:
            from config import get_default_config
            self.config = get_default_config()

        # Load model
        print("Loading model...")
        self.model = load_pretrained_model(checkpoint_path, self.config, self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            self.config.model.language_model
        )

        # Image preprocessing
        self.image_size = self.config.data.image_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        print("Model loaded successfully!")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image

        Args:
            image: PIL Image

        Returns:
            Preprocessed image tensor [1, 3, H, W]
        """

        # Resize
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # To tensor
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # [3, H, W]

        # Normalize
        image_tensor = (image_tensor - self.mean) / self.std

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)  # [1, 3, H, W]

        return image_tensor

    def preprocess_text(self, text: str) -> dict:
        """
        Preprocess text

        Args:
            text: Text query

        Returns:
            Dictionary with input_ids and attention_mask
        """

        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config.data.max_text_length,
            return_tensors='pt'
        )

        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask']
        }

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        text: str,
        threshold: float = 0.5,
        return_attention: bool = False
    ) -> dict:
        """
        Run inference

        Args:
            image: PIL Image
            text: Text query
            threshold: Threshold for binarizing mask
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with predictions
        """

        # Preprocess
        image_tensor = self.preprocess_image(image).to(self.device)
        text_dict = self.preprocess_text(text)
        input_ids = text_dict['input_ids'].to(self.device)
        attention_mask = text_dict['attention_mask'].to(self.device)

        # Predict
        pred_mask, attn_weights, _ = self.model(
            image_tensor,
            input_ids,
            attention_mask
        )

        # Process output
        pred_mask_np = pred_mask[0, 0].cpu().numpy()  # [H, W]

        # Binarize
        pred_mask_binary = (pred_mask_np > threshold).astype(np.uint8)

        result = {
            'mask': pred_mask_np,
            'mask_binary': pred_mask_binary,
            'mask_pil': Image.fromarray((pred_mask_np * 255).astype(np.uint8)),
            'confidence': pred_mask_np.max()
        }

        if return_attention:
            result['attention_weights'] = attn_weights.cpu().numpy()

        return result

    def visualize(
        self,
        image: Image.Image,
        text: str,
        threshold: float = 0.5,
        output_path: str = None
    ):
        """
        Predict and visualize

        Args:
            image: PIL Image
            text: Text query
            threshold: Threshold
            output_path: Path to save visualization
        """

        # Predict
        result = self.predict(image, text, threshold)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title(f'Input Image\n"{text}"')
        axes[0].axis('off')

        # Predicted mask
        axes[1].imshow(result['mask'], cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Predicted Mask\n(Confidence: {result["confidence"]:.3f})')
        axes[1].axis('off')

        # Overlay
        overlay = np.array(image.resize((self.image_size, self.image_size))) / 255.0
        mask_rgb = np.stack([result['mask']] * 3, axis=-1)

        # Red overlay
        overlay = overlay * (1 - mask_rgb) + mask_rgb * np.array([1, 0, 0])
        overlay = np.clip(overlay, 0, 1)

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()

        plt.close()

        return result


def main():
    parser = argparse.ArgumentParser(description='Inference for Referring Segmentation')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')

    # Input
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--text', type=str, required=True, help='Text query')

    # Settings
    parser.add_argument('--threshold', type=float, default=0.5, help='Mask threshold')
    parser.add_argument('--device', type=str, default='cuda')

    # Output
    parser.add_argument('--output', type=str, default=None, help='Output path for visualization')
    parser.add_argument('--save_mask', type=str, default=None, help='Path to save mask')

    args = parser.parse_args()

    # Load model
    inference = ReferringSegmentationInference(args.checkpoint, args.device)

    # Load image
    print(f"Loading image from {args.image}...")
    image = Image.open(args.image).convert('RGB')

    print(f"Query: \"{args.text}\"")
    print("Running inference...")

    # Predict
    result = inference.predict(image, args.text, args.threshold)

    print(f"Confidence: {result['confidence']:.3f}")

    # Visualize
    output_path = args.output or 'inference_result.png'

    inference.visualize(
        image,
        args.text,
        args.threshold,
        output_path
    )

    # Save mask if requested
    if args.save_mask:
        result['mask_pil'].save(args.save_mask)
        print(f"Saved mask to {args.save_mask}")


if __name__ == '__main__':
    main()
