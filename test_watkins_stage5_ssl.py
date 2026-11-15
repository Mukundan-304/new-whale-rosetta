from __future__ import annotations

import unittest

import torch

from watkins_stage5_evaluate import _sample_pairs
from watkins_stage5_ssl import augment_waveforms, nt_xent_loss


class WatkinsStage5SSLTests(unittest.TestCase):
    def test_nt_xent_produces_finite_loss_and_gradients(self) -> None:
        first = torch.randn(4, 16, requires_grad=True)
        second = first.detach().clone() + 0.05 * torch.randn(4, 16)
        second.requires_grad_()

        loss = nt_xent_loss(first, second, temperature=0.1)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(first.grad)
        self.assertGreater(float(first.grad.abs().sum()), 0.0)
        self.assertIsNotNone(second.grad)
        self.assertGreater(float(second.grad.abs().sum()), 0.0)

    def test_nt_xent_rejects_singleton_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            nt_xent_loss(torch.ones(1, 4), torch.ones(1, 4), temperature=0.1)

    def test_augmentations_are_finite_and_seed_reproducible(self) -> None:
        waveforms = torch.zeros(3, 2400)
        waveforms[:, 500:1500] = 0.2
        arguments = {
            "sample_rate": 48000,
            "max_shift_ms": 1.0,
            "gain_db": 2.0,
            "min_noise_snr_db": 35.0,
            "max_noise_snr_db": 40.0,
        }
        first = augment_waveforms(
            waveforms, torch.Generator().manual_seed(7), **arguments
        )
        repeated = augment_waveforms(
            waveforms, torch.Generator().manual_seed(7), **arguments
        )

        self.assertEqual(first.shape, waveforms.shape)
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, waveforms))

    def test_evaluation_pairs_exclude_segments_from_same_recording(self) -> None:
        metadata = [
            {
                "species_label": "species-a",
                "recording_id": f"recording-{index}",
                "location_domain": "north" if index < 2 else "south",
            }
            for index in range(4)
        ] + [
            {
                "species_label": "species-b",
                "recording_id": f"other-{index}",
                "location_domain": "north" if index < 2 else "south",
            }
            for index in range(4)
        ]
        pairs = _sample_pairs(metadata, seed=29, pairs_per_type=5)

        self.assertTrue(pairs["same_species"])
        self.assertTrue(pairs["different_species"])
        self.assertTrue(pairs["same_species_same_location"])
        self.assertTrue(pairs["same_species_different_location"])
        for group in pairs.values():
            for left, right in group:
                self.assertNotEqual(
                    metadata[left]["recording_id"], metadata[right]["recording_id"]
                )


if __name__ == "__main__":
    unittest.main()
