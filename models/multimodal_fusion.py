"""
Multi-modal fusion modules for combining vision and language features
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class CrossModalAttention(nn.Module):
    """
    Cross-attention between vision and language

    Allows language features to attend to relevant image regions
    and vision features to attend to relevant words.

    Uses multi-head attention mechanism with residual connections
    and layer normalization.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Vision to Language attention (vision queries language)
        self.v2l_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Language to Vision attention (language queries vision)
        self.l2v_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward networks
        self.ffn_v = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        self.ffn_l = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm_v1 = nn.LayerNorm(embed_dim)
        self.norm_v2 = nn.LayerNorm(embed_dim)
        self.norm_l1 = nn.LayerNorm(embed_dim)
        self.norm_l2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        lang_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cross-modal fusion

        Args:
            vision_features: [B, H*W, embed_dim] flattened spatial features
            language_features: [B, L, embed_dim] word features
            lang_mask: [B, L] attention mask for padding (True for padding)

        Returns:
            fused_vision: [B, H*W, embed_dim] language-aware visual features
            fused_language: [B, L, embed_dim] vision-aware language features
            v2l_weights: [B, num_heads, H*W, L] attention weights
        """

        # Language to Vision (language attends to vision)
        l2v_out, _ = self.l2v_attention(
            query=language_features,
            key=vision_features,
            value=vision_features,
            key_padding_mask=None  # Vision has no padding
        )

        # Residual connection and norm
        language_features = self.norm_l1(language_features + l2v_out)

        # FFN
        l_ffn_out = self.ffn_l(language_features)
        language_features = self.norm_l2(language_features + l_ffn_out)

        # Vision to Language (vision attends to language)
        v2l_out, v2l_weights = self.v2l_attention(
            query=vision_features,
            key=language_features,
            value=language_features,
            key_padding_mask=lang_mask  # Mask out padding tokens
        )

        # Residual connection and norm
        vision_features = self.norm_v1(vision_features + v2l_out)

        # FFN
        v_ffn_out = self.ffn_v(vision_features)
        vision_features = self.norm_v2(vision_features + v_ffn_out)

        return vision_features, language_features, v2l_weights


class MultiModalFusion(nn.Module):
    """
    Multi-layer multi-modal fusion with cross-attention

    Stacks multiple CrossModalAttention layers for deeper interaction
    between vision and language modalities.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of fusion layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Stack of cross-modal attention layers
        self.layers = nn.ModuleList([
            CrossModalAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        lang_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-layer cross-modal fusion

        Args:
            vision_features: [B, H*W, embed_dim] flattened spatial features
            language_features: [B, L, embed_dim] word features
            lang_mask: [B, L] attention mask for padding

        Returns:
            fused_vision: [B, H*W, embed_dim] fused visual features
            fused_language: [B, L, embed_dim] fused language features
            attn_weights: [B, num_heads, H*W, L] final layer attention weights
        """

        attn_weights = None

        # Apply each fusion layer
        for layer in self.layers:
            vision_features, language_features, attn_weights = layer(
                vision_features,
                language_features,
                lang_mask
            )

        return vision_features, language_features, attn_weights


class DynamicMultiModalFusion(nn.Module):
    """
    Dynamic fusion that adapts based on input

    Uses gating mechanisms to control information flow between modalities.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Cross-attention
        self.cross_attention = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Gating mechanisms
        self.vision_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        self.language_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        lang_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dynamic multi-modal fusion with gating

        Args:
            vision_features: [B, H*W, embed_dim]
            language_features: [B, L, embed_dim]
            lang_mask: [B, L] attention mask

        Returns:
            fused_vision: [B, H*W, embed_dim]
            fused_language: [B, L, embed_dim]
            attn_weights: Attention weights
        """

        # Store original features
        vision_orig = vision_features
        language_orig = language_features

        # Cross-modal attention
        vision_fused, language_fused, attn_weights = self.cross_attention(
            vision_features,
            language_features,
            lang_mask
        )

        # Compute gates
        vision_gate_input = torch.cat([vision_orig, vision_fused], dim=-1)
        vision_gate = self.vision_gate(vision_gate_input)

        language_gate_input = torch.cat([language_orig, language_fused], dim=-1)
        language_gate = self.language_gate(language_gate_input)

        # Apply gates
        vision_output = vision_gate * vision_fused + (1 - vision_gate) * vision_orig
        language_output = language_gate * language_fused + (1 - language_gate) * language_orig

        return vision_output, language_output, attn_weights


class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for conditioning vision on language

    Language features generate scale (gamma) and shift (beta) parameters
    to modulate visual features.

    Simpler alternative to attention-based fusion.

    Args:
        vision_dim: Vision feature dimension
        language_dim: Language feature dimension
    """

    def __init__(
        self,
        vision_dim: int = 256,
        language_dim: int = 256
    ):
        super().__init__()

        # Generate FiLM parameters from language
        self.film_generator = nn.Sequential(
            nn.Linear(language_dim, vision_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(vision_dim * 2, vision_dim * 2)
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        language_feature: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply FiLM modulation

        Args:
            vision_features: [B, C, H, W] or [B, H*W, C] visual features
            language_feature: [B, language_dim] sentence-level language feature

        Returns:
            modulated_vision: Same shape as vision_features
        """

        # Generate FiLM parameters
        film_params = self.film_generator(language_feature)  # [B, vision_dim * 2]

        # Split into gamma (scale) and beta (shift)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # Each [B, vision_dim]

        # Handle different input shapes
        if vision_features.dim() == 4:
            # [B, C, H, W] format
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            beta = beta.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        elif vision_features.dim() == 3:
            # [B, H*W, C] format
            gamma = gamma.unsqueeze(1)  # [B, 1, C]
            beta = beta.unsqueeze(1)  # [B, 1, C]

        # Apply FiLM: gamma * x + beta
        modulated = gamma * vision_features + beta

        return modulated


def build_fusion_module(
    fusion_type: str = 'cross_attention',
    embed_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 3,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to build fusion module

    Args:
        fusion_type: Type of fusion ('cross_attention', 'dynamic', 'film')
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of fusion layers
        dropout: Dropout probability
        **kwargs: Additional arguments

    Returns:
        Fusion module
    """

    if fusion_type == 'cross_attention':
        if num_layers == 1:
            return CrossModalAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            return MultiModalFusion(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )

    elif fusion_type == 'dynamic':
        return DynamicMultiModalFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

    elif fusion_type == 'film':
        return FiLMFusion(
            vision_dim=embed_dim,
            language_dim=embed_dim
        )

    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
