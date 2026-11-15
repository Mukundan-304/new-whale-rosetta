"""CNN-Transformer Acoustic Representation Learning Model.

Combines a 3-stage Conv2D feature extractor for local time-frequency patterns
with a 2-block Multi-Head Self-Attention Transformer encoder for long-range
temporal dependencies over acoustic tokens.

Follows SimCLR principles:
- The encoder extracts acoustic representations (h).
- The projection head projects representations into contrastive space (z) during training.
- Primary embeddings are extracted from the encoder BEFORE the projection head.
"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class CNNFeatureExtractor(nn.Module):
    """Local time-frequency feature extractor.

    Architecture:
    - Conv2D: 32 filters, 3x3 kernel, padding=1
    - Batch Normalization
    - ReLU activation
    - MaxPooling2D: kernel (2, 2)
    - Conv2D: 64 filters, 3x3 kernel, padding=1
    - Batch Normalization
    - ReLU activation
    - MaxPooling2D: kernel (2, 2)
    - Conv2D: 128 filters, 3x3 kernel, padding=1
    - Batch Normalization
    - ReLU activation
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        # Stage 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # Stage 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # Stage 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Input:  (B, 1, F=128, T=934)
        Stage 1: (B, 32, F=64, T=467)
        Stage 2: (B, 64, F=32, T=233)
        Stage 3: (B, 128, F=32, T=233)
        """
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal tokens."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class CNNTransformerAcousticModel(nn.Module):
    """Full CNN-Transformer acoustic representation model.

    Features:
    - 3-layer CNN preserving temporal resolution (233 tokens from 934 STFT frames).
    - Frequency aggregation via adaptive average pooling across frequency bins.
    - Projection of token channels to Transformer hidden dimension d_model.
    - 2 Transformer encoder blocks with 4 attention heads, LayerNorm, GELU, and residual connections.
    - Temporal mean pooling generating fixed-dimensional encoder representation (h).
    - Projection head (z) for SimCLR NT-Xent contrastive learning.
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        proj_dim: int = 128,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.d_model = d_model
        self.proj_dim = proj_dim

        # 1. Local time-frequency extractor
        self.cnn = CNNFeatureExtractor(in_channels=in_channels)

        # 2. CNN-to-Transformer conversion:
        # Aggregate frequency dimension (collapsing 32 frequency bins to 1)
        # while strictly preserving temporal tokens (233 tokens)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.token_proj = nn.Linear(128, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=500, dropout=dropout)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # 4. Encoder temporal pooling normalization
        self.encoder_norm = nn.LayerNorm(d_model)

        # 5. Projection Head for SimCLR contrastive training (Section 4/6)
        # Maps encoder representations h into contrastive embedding space z
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def extract_temporal_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Extract sequence of temporal tokens prior to pooling."""
        cnn_features = self.cnn(x)                       # (B, 128, F', T')
        freq_pooled = self.freq_pool(cnn_features)       # (B, 128, 1, T')
        tokens = freq_pooled.squeeze(2).permute(0, 2, 1) # (B, T', 128)
        tokens = self.token_proj(tokens)                 # (B, T', d_model)
        tokens = self.pos_encoder(tokens)                # (B, T', d_model)
        trans_tokens = self.transformer(tokens)          # (B, T', d_model)
        return trans_tokens

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract encoder acoustic representation before the projection head.

        This is the PRIMARY Stage 10 representation.
        Returns:
            L2-normalized embedding tensor of shape (B, d_model).
        """
        trans_tokens = self.extract_temporal_tokens(x)   # (B, T', d_model)
        pooled = trans_tokens.mean(dim=1)                # (B, d_model)
        normed = self.encoder_norm(pooled)               # (B, d_model)
        return F.normalize(normed, p=2, dim=-1)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Map encoder representation h to contrastive space z."""
        z = self.proj_head(h)
        return F.normalize(z, p=2, dim=-1)

    def forward(self, x: torch.Tensor, return_projection: bool = True) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input spectrogram (B, 1, 128, 934).
            return_projection: If True, returns projected contrastive embedding z (for training).
                               If False, returns encoder representation h (before projection head).
        """
        h = self.encode(x)
        if return_projection:
            return self.project(h)
        return h
