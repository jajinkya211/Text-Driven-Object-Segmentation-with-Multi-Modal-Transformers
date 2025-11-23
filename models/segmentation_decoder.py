"""
Segmentation decoder for producing binary masks from multi-modal features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


class SegmentationDecoder(nn.Module):
    """
    Progressive upsampling decoder for segmentation

    Takes fused multi-modal features and produces binary mask
    through progressive upsampling with skip connections.

    Architecture:
    - Progressive upsampling (x2 each step)
    - Optional skip connections from encoder
    - Refinement at each scale
    - Final binary mask prediction

    Args:
        in_channels: Input feature channels
        decoder_channels: List of channel dimensions for each upsampling stage
        use_skip_connections: Whether to use skip connections
    """

    def __init__(
        self,
        in_channels: int = 256,
        decoder_channels: List[int] = [128, 64, 32, 16],
        use_skip_connections: bool = False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.decoder_channels = decoder_channels
        self.use_skip_connections = use_skip_connections

        # Build upsampling blocks
        self.up_blocks = nn.ModuleList()

        prev_channels = in_channels
        for out_channels in decoder_channels:
            self.up_blocks.append(
                self._make_upsampling_block(prev_channels, out_channels)
            )
            prev_channels = out_channels

        # Final prediction head
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[-1], 1, 1),  # Binary mask
        )

    def _make_upsampling_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create upsampling block with conv layers"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        fused_features: torch.Tensor,
        skip_connections: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Decode features to segmentation mask

        Args:
            fused_features: [B, in_channels, H/32, W/32] fused multi-modal features
            skip_connections: Optional list of encoder features for skip connections

        Returns:
            mask: [B, 1, H, W] binary segmentation mask (before sigmoid)
        """

        x = fused_features

        # Progressive upsampling
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x)

            # Add skip connection if available
            if self.use_skip_connections and skip_connections is not None:
                if i < len(skip_connections) and skip_connections[i] is not None:
                    skip = skip_connections[i]
                    # Resize skip to match x if needed
                    if skip.shape[2:] != x.shape[2:]:
                        skip = F.interpolate(
                            skip,
                            size=x.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    x = x + skip

        # Final prediction
        mask = self.final_conv(x)  # [B, 1, H, W]

        return mask


class AdvancedSegmentationDecoder(nn.Module):
    """
    Advanced decoder with ASPP (Atrous Spatial Pyramid Pooling)

    Uses multi-scale convolutions to capture context at different scales
    before final prediction.

    Args:
        in_channels: Input feature channels
        decoder_channels: Channel dimensions for decoder stages
        aspp_rates: Dilation rates for ASPP
    """

    def __init__(
        self,
        in_channels: int = 256,
        decoder_channels: List[int] = [128, 64, 32, 16],
        aspp_rates: List[int] = [6, 12, 18]
    ):
        super().__init__()

        self.in_channels = in_channels
        self.decoder_channels = decoder_channels

        # ASPP module
        self.aspp = ASPP(in_channels, aspp_rates)

        # Decoder blocks
        self.up_blocks = nn.ModuleList()
        prev_channels = in_channels

        for out_channels in decoder_channels:
            self.up_blocks.append(
                self._make_upsampling_block(prev_channels, out_channels)
            )
            prev_channels = out_channels

        # Final prediction
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[-1], 1, 1)
        )

    def _make_upsampling_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create upsampling block"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        fused_features: torch.Tensor,
        skip_connections: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Decode with ASPP

        Args:
            fused_features: [B, in_channels, H, W]
            skip_connections: Optional skip connections

        Returns:
            mask: [B, 1, H', W'] segmentation mask
        """

        # Apply ASPP for multi-scale context
        x = self.aspp(fused_features)

        # Progressive upsampling
        for up_block in self.up_blocks:
            x = up_block(x)

        # Final prediction
        mask = self.final_conv(x)

        return mask


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling

    Captures multi-scale contextual information using parallel
    atrous convolutions with different dilation rates.

    Args:
        in_channels: Input channels
        rates: List of dilation rates
        out_channels: Output channels
    """

    def __init__(
        self,
        in_channels: int,
        rates: List[int] = [6, 12, 18],
        out_channels: int = 256
    ):
        super().__init__()

        self.branches = nn.ModuleList()

        # 1x1 convolution branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )

        # Atrous convolution branches
        for rate in rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Fusion
        num_branches = len(rates) + 2  # +1 for 1x1, +1 for global pool
        self.fusion = nn.Sequential(
            nn.Conv2d(
                out_channels * num_branches,
                out_channels,
                1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ASPP

        Args:
            x: [B, C, H, W] input features

        Returns:
            out: [B, out_channels, H, W] multi-scale features
        """

        H, W = x.shape[2:]

        # Apply each branch
        branch_outputs = []

        for branch in self.branches:
            branch_outputs.append(branch(x))

        # Global pooling branch
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(
            global_feat,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        branch_outputs.append(global_feat)

        # Concatenate and fuse
        concat = torch.cat(branch_outputs, dim=1)
        out = self.fusion(concat)

        return out


class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder for segmentation

    Uses transformer decoder layers to refine features before upsampling.
    Can attend to multi-scale features from encoder.

    Args:
        in_channels: Input channels
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_channels = in_channels

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=in_channels * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(inplace=True)
        )

        # Final prediction
        self.final_conv = nn.Conv2d(16, 1, 1)

    def forward(
        self,
        fused_features: torch.Tensor,
        memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Transformer-based decoding

        Args:
            fused_features: [B, C, H, W] fused features
            memory: Optional memory from encoder

        Returns:
            mask: [B, 1, H', W'] segmentation mask
        """

        B, C, H, W = fused_features.shape

        # Flatten spatial dimensions
        x = fused_features.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]

        # Transformer decoding
        if memory is not None:
            x = self.transformer_decoder(x, memory)
        else:
            x = self.transformer_decoder(x, x)

        # Reshape back to spatial
        x = x.transpose(1, 2).view(B, C, H, W)

        # Upsample
        x = self.upsample(x)

        # Final prediction
        mask = self.final_conv(x)

        return mask


def build_decoder(
    decoder_type: str = 'standard',
    in_channels: int = 256,
    decoder_channels: List[int] = [128, 64, 32, 16],
    **kwargs
) -> nn.Module:
    """
    Factory function to build decoder

    Args:
        decoder_type: Type of decoder ('standard', 'aspp', 'transformer')
        in_channels: Input channels
        decoder_channels: Decoder channel dimensions
        **kwargs: Additional arguments

    Returns:
        Decoder module
    """

    if decoder_type == 'standard':
        return SegmentationDecoder(
            in_channels=in_channels,
            decoder_channels=decoder_channels,
            **kwargs
        )

    elif decoder_type == 'aspp':
        return AdvancedSegmentationDecoder(
            in_channels=in_channels,
            decoder_channels=decoder_channels,
            **kwargs
        )

    elif decoder_type == 'transformer':
        return TransformerDecoder(
            in_channels=in_channels,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")
