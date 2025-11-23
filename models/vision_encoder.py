"""
Vision Encoder with ResNet-101 + FPN for multi-scale feature extraction
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from typing import Dict, Optional


class VisionEncoder(nn.Module):
    """
    Multi-scale vision encoder using ResNet + FPN

    Architecture:
    - ResNet-101 backbone (pretrained on ImageNet)
    - Feature Pyramid Network for multi-scale features
    - Output: 5 feature maps at different resolutions

    Features extracted at:
    - C2: 1/4 resolution (256 channels)
    - C3: 1/8 resolution (512 channels)
    - C4: 1/16 resolution (1024 channels)
    - C5: 1/32 resolution (2048 channels)

    FPN produces:
    - P2, P3, P4, P5 (all 256 channels by default)

    Args:
        backbone: Backbone architecture ('resnet101', 'resnet50')
        pretrained: Whether to use ImageNet pretrained weights
        trainable_layers: Number of trainable layers (0-5)
        feature_dim: Output feature dimension
    """

    def __init__(
        self,
        backbone: str = 'resnet101',
        pretrained: bool = True,
        trainable_layers: int = 5,
        feature_dim: int = 256
    ):
        super().__init__()

        self.backbone_name = backbone
        self.feature_dim = feature_dim

        if backbone in ['resnet101', 'resnet50']:
            # Load ResNet with FPN
            self.backbone = resnet_fpn_backbone(
                backbone,
                pretrained=pretrained,
                trainable_layers=trainable_layers,
                returned_layers=[1, 2, 3, 4],
                extra_blocks=None
            )

            # FPN output is already 256 channels
            self.fpn_output_dim = 256

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Additional conv layers for feature refinement
        self.refine_conv = nn.ModuleDict({
            '0': self._make_refine_block(self.fpn_output_dim, feature_dim),
            '1': self._make_refine_block(self.fpn_output_dim, feature_dim),
            '2': self._make_refine_block(self.fpn_output_dim, feature_dim),
            '3': self._make_refine_block(self.fpn_output_dim, feature_dim),
        })

        # Initialize weights
        self._init_weights()

    def _make_refine_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create refinement block for features"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _init_weights(self):
        """Initialize weights for refinement layers"""
        for m in self.refine_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from images

        Args:
            images: [B, 3, H, W] RGB images (normalized)

        Returns:
            features: Dict with multi-scale feature maps
              - '0' (p2): [B, feature_dim, H/4, W/4]
              - '1' (p3): [B, feature_dim, H/8, W/8]
              - '2' (p4): [B, feature_dim, H/16, W/16]
              - '3' (p5): [B, feature_dim, H/32, W/32]
        """

        # Extract features through backbone + FPN
        features = self.backbone(images)

        # Refine features
        refined_features = {}
        for key, feat in features.items():
            refined_features[key] = self.refine_conv[key](feat)

        return refined_features

    def get_output_channels(self) -> int:
        """Get number of output channels"""
        return self.feature_dim


class SwinVisionEncoder(nn.Module):
    """
    Alternative vision encoder using Swin Transformer

    Swin Transformer provides hierarchical features similar to CNN backbones
    but with better long-range modeling capabilities.

    Args:
        model_name: Swin model variant
        pretrained: Whether to use pretrained weights
        feature_dim: Output feature dimension
    """

    def __init__(
        self,
        model_name: str = 'swin_base_patch4_window7_224',
        pretrained: bool = True,
        feature_dim: int = 256
    ):
        super().__init__()

        # Load Swin Transformer
        from transformers import AutoModel, AutoImageProcessor

        self.model_name = model_name
        self.feature_dim = feature_dim

        # Load pretrained model
        if pretrained:
            self.backbone = AutoModel.from_pretrained(f'microsoft/{model_name}')
        else:
            from transformers import SwinConfig
            config = SwinConfig()
            self.backbone = AutoModel.from_config(config)

        # Swin output dimensions (hierarchical features)
        # For swin_base: [128, 256, 512, 1024] channels at [H/4, H/8, H/16, H/32]
        self.swin_dims = [128, 256, 512, 1024]

        # Projection layers to match feature_dim
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, feature_dim, 1, bias=False),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            )
            for dim in self.swin_dims
        ])

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features using Swin Transformer

        Args:
            images: [B, 3, H, W] RGB images

        Returns:
            features: Dict with multi-scale feature maps
        """

        # Extract features
        outputs = self.backbone(images, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # Swin outputs are in format [B, H*W, C]
        # Need to reshape to [B, C, H, W]
        features = {}

        # Get features at different scales (stages 1-4)
        for i, (hidden_state, proj) in enumerate(zip(hidden_states[1:], self.projections)):
            B, HW, C = hidden_state.shape

            # Calculate spatial dimensions
            H = W = int(HW ** 0.5)

            # Reshape and project
            feat = hidden_state.transpose(1, 2).view(B, C, H, W)
            feat = proj(feat)

            features[str(i)] = feat

        return features

    def get_output_channels(self) -> int:
        """Get number of output channels"""
        return self.feature_dim


def build_vision_encoder(
    backbone: str = 'resnet101',
    pretrained: bool = True,
    feature_dim: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function to build vision encoder

    Args:
        backbone: Backbone architecture
        pretrained: Use pretrained weights
        feature_dim: Output feature dimension
        **kwargs: Additional arguments

    Returns:
        Vision encoder module
    """

    if backbone in ['resnet50', 'resnet101']:
        return VisionEncoder(
            backbone=backbone,
            pretrained=pretrained,
            feature_dim=feature_dim,
            **kwargs
        )
    elif 'swin' in backbone:
        return SwinVisionEncoder(
            model_name=backbone,
            pretrained=pretrained,
            feature_dim=feature_dim
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
