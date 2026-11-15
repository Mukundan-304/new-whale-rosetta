"""Unit tests for Stage 10: CNN-Transformer Acoustic Representation Learning."""

from __future__ import annotations

import unittest
import torch
import torch.nn.functional as F

from watkins_stage10.cnn_transformer import (
    CNNFeatureExtractor,
    CNNTransformerAcousticModel,
    PositionalEncoding,
)


class WatkinsStage10ArchitectureTests(unittest.TestCase):
    def test_cnn_feature_extractor_preserves_temporal_resolution(self) -> None:
        cnn = CNNFeatureExtractor(in_channels=1)
        # Input log-Mel: (B=2, C=1, F=128, T=934)
        x = torch.randn(2, 1, 128, 934)
        out = cnn(x)

        # After stage 1: F=64, T=467
        # After stage 2: F=32, T=233
        # After stage 3: F=32, T=233, Channels=128
        self.assertEqual(out.shape, (2, 128, 32, 233))

    def test_positional_encoding_shape(self) -> None:
        pe = PositionalEncoding(d_model=128, max_len=500, dropout=0.0)
        tokens = torch.zeros(2, 233, 128)
        encoded = pe(tokens)

        self.assertEqual(encoded.shape, (2, 233, 128))
        self.assertFalse(torch.allclose(encoded[0, 0], encoded[0, 1]))

    def test_encoder_extracts_representations_before_projection_head(self) -> None:
        model = CNNTransformerAcousticModel(
            in_channels=1,
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            proj_dim=128,
        )

        x = torch.randn(3, 1, 128, 934)
        # Encoder representation (before projection head)
        encoder_h = model.encode(x)
        self.assertEqual(encoder_h.shape, (3, 128))

        # Projected contrastive representation (for NT-Xent training)
        projected_z = model.project(encoder_h)
        self.assertEqual(projected_z.shape, (3, 128))

        # Both should be L2-normalized
        for norm in torch.norm(encoder_h, p=2, dim=-1):
            self.assertAlmostEqual(norm.item(), 1.0, places=5)
        for norm in torch.norm(projected_z, p=2, dim=-1):
            self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_extract_temporal_tokens_shape(self) -> None:
        model = CNNTransformerAcousticModel(d_model=128, proj_dim=128)
        x = torch.randn(2, 1, 128, 934)
        tokens = model.extract_temporal_tokens(x)

        # Sequence shape: (batch, seq_len=233, d_model=128)
        self.assertEqual(tokens.shape, (2, 233, 128))

    def test_parameter_count_within_lightweight_budget(self) -> None:
        model = CNNTransformerAcousticModel(d_model=128, proj_dim=128)
        total_params = sum(p.numel() for p in model.parameters())

        # Budget check: lightweight CNN-Transformer should be under 1.5 million parameters
        self.assertLess(total_params, 1_500_000)
        self.assertGreater(total_params, 100_000)


if __name__ == "__main__":
    unittest.main()
