"""
Complete Referring Expression Segmentation Model

Integrates vision encoder, language encoder, multi-modal fusion, and segmentation decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .vision_encoder import VisionEncoder, build_vision_encoder
from .language_encoder import LanguageEncoder, build_language_encoder
from .multimodal_fusion import MultiModalFusion, build_fusion_module
from .segmentation_decoder import SegmentationDecoder, build_decoder


class ReferringSegmentationModel(nn.Module):
    """
    Complete referring expression segmentation model

    Integrates all components:
    - Vision encoder: Extracts multi-scale visual features
    - Language encoder: Encodes referring expression
    - Multi-modal fusion: Fuses vision and language
    - Segmentation decoder: Produces binary mask

    Args:
        config: Configuration object with model parameters
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Vision encoder
        self.vision_encoder = build_vision_encoder(
            backbone=config.model.vision_backbone,
            pretrained=True,
            feature_dim=config.model.vision_feature_dim
        )

        # Language encoder
        self.language_encoder = build_language_encoder(
            bert_model=config.model.language_model,
            output_dim=config.model.language_output_dim,
            use_lstm=True,
            lstm_hidden_dim=config.model.lstm_hidden_dim,
            lstm_num_layers=config.model.lstm_num_layers,
            dropout=config.model.fusion_dropout
        )

        # Multi-modal fusion
        self.fusion = build_fusion_module(
            fusion_type='cross_attention',
            embed_dim=config.model.fusion_embed_dim,
            num_heads=config.model.fusion_num_heads,
            num_layers=config.model.fusion_num_layers,
            dropout=config.model.fusion_dropout
        )

        # Spatial projection for vision features
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(config.model.vision_feature_dim, config.model.fusion_embed_dim, 1),
            nn.BatchNorm2d(config.model.fusion_embed_dim),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = build_decoder(
            decoder_type='standard',
            in_channels=config.model.fusion_embed_dim,
            decoder_channels=config.model.decoder_channels
        )

        # Additional refinement
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass

        Args:
            images: [B, 3, H, W] RGB images (normalized)
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask
            return_features: Whether to return intermediate features

        Returns:
            pred_mask: [B, 1, H, W] predicted segmentation mask (with sigmoid)
            attn_weights: [B, num_heads, H*W, L] attention weights (optional)
            features: Dict of intermediate features (optional)
        """

        # Extract visual features (multi-scale)
        vision_features = self.vision_encoder(images)  # Dict with keys '0', '1', '2', '3'

        # Extract language features
        word_features, sent_feature = self.language_encoder(
            input_ids, attention_mask
        )  # [B, L, 256], [B, 256]

        # Use intermediate-scale features (typically '2' corresponds to 1/16 resolution)
        # This is a good balance between spatial resolution and semantic information
        vis_feat = vision_features['2']  # [B, 256, H/16, W/16]

        B, C, H, W = vis_feat.shape

        # Project vision features
        vis_feat = self.spatial_proj(vis_feat)  # [B, fusion_dim, H, W]

        # Reshape vision features for attention: [B, C, H, W] -> [B, H*W, C]
        vis_feat_flat = vis_feat.view(B, self.config.model.fusion_embed_dim, H * W)
        vis_feat_flat = vis_feat_flat.transpose(1, 2)  # [B, H*W, fusion_dim]

        # Create language mask (True for padding tokens)
        lang_padding_mask = ~attention_mask.bool()

        # Cross-modal fusion
        fused_vis, fused_lang, attn_weights = self.fusion(
            vis_feat_flat,
            word_features,
            lang_mask=lang_padding_mask
        )

        # Reshape back to spatial: [B, H*W, C] -> [B, C, H, W]
        fused_vis = fused_vis.transpose(1, 2).view(B, self.config.model.fusion_embed_dim, H, W)

        # Decode to mask
        pred_mask = self.decoder(fused_vis)  # [B, 1, H', W']

        # Upsample to original image size if needed
        if pred_mask.shape[2:] != images.shape[2:]:
            pred_mask = F.interpolate(
                pred_mask,
                size=images.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Refine prediction
        pred_mask = pred_mask + self.refine(pred_mask)

        # Apply sigmoid
        pred_mask = torch.sigmoid(pred_mask)

        if return_features:
            features = {
                'vision_features': vision_features,
                'word_features': word_features,
                'sent_feature': sent_feature,
                'fused_vis': fused_vis,
                'fused_lang': fused_lang
            }
            return pred_mask, attn_weights, features
        else:
            return pred_mask, attn_weights, None

    def get_trainable_parameters(self):
        """Get trainable parameters grouped by component"""
        param_groups = {
            'vision_encoder': self.vision_encoder.parameters(),
            'language_encoder': self.language_encoder.parameters(),
            'fusion': self.fusion.parameters(),
            'decoder': self.decoder.parameters()
        }
        return param_groups


class LightweightReferringSegmentationModel(nn.Module):
    """
    Lightweight version of referring segmentation model

    Uses simpler components for faster inference and lower memory usage.
    Good for deployment or resource-constrained scenarios.

    Args:
        config: Configuration object
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Use ResNet50 instead of ResNet101
        self.vision_encoder = build_vision_encoder(
            backbone='resnet50',
            pretrained=True,
            feature_dim=256
        )

        # Use simple BERT without LSTM
        self.language_encoder = build_language_encoder(
            bert_model='bert-base-uncased',
            output_dim=256,
            use_lstm=False
        )

        # Single-layer fusion
        self.fusion = build_fusion_module(
            fusion_type='cross_attention',
            embed_dim=256,
            num_heads=8,
            num_layers=1,
            dropout=0.1
        )

        # Spatial projection
        self.spatial_proj = nn.Conv2d(256, 256, 1)

        # Simple decoder
        self.decoder = build_decoder(
            decoder_type='standard',
            in_channels=256,
            decoder_channels=[128, 64, 32, 16]
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            images: [B, 3, H, W]
            input_ids: [B, L]
            attention_mask: [B, L]

        Returns:
            pred_mask: [B, 1, H, W]
            attn_weights: Attention weights
        """

        # Extract features
        vision_features = self.vision_encoder(images)
        word_features, sent_feature = self.language_encoder(input_ids, attention_mask)

        # Use P4 level
        vis_feat = vision_features['2']
        B, C, H, W = vis_feat.shape

        # Project and flatten
        vis_feat = self.spatial_proj(vis_feat)
        vis_feat_flat = vis_feat.view(B, 256, H * W).transpose(1, 2)

        # Fusion
        lang_padding_mask = ~attention_mask.bool()
        fused_vis, _, attn_weights = self.fusion(
            vis_feat_flat,
            word_features,
            lang_mask=lang_padding_mask
        )

        # Reshape and decode
        fused_vis = fused_vis.transpose(1, 2).view(B, 256, H, W)
        pred_mask = self.decoder(fused_vis)

        # Upsample and sigmoid
        if pred_mask.shape[2:] != images.shape[2:]:
            pred_mask = F.interpolate(
                pred_mask,
                size=images.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        pred_mask = torch.sigmoid(pred_mask)

        return pred_mask, attn_weights


def build_model(config, lightweight: bool = False) -> nn.Module:
    """
    Factory function to build referring segmentation model

    Args:
        config: Configuration object
        lightweight: Whether to build lightweight version

    Returns:
        Model instance
    """

    if lightweight:
        model = LightweightReferringSegmentationModel(config)
    else:
        model = ReferringSegmentationModel(config)

    return model


def load_pretrained_model(checkpoint_path: str, config, device: str = 'cuda') -> nn.Module:
    """
    Load pretrained model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration object
        device: Device to load model on

    Returns:
        Loaded model
    """

    model = build_model(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model
