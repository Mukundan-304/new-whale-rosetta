from __future__ import annotations

import unittest

import numpy as np

from watkins_stage6.run_stage6_discovery import _cluster_statistics


class WatkinsStage6DiscoveryTests(unittest.TestCase):
    def test_cluster_statistics_separate_noise_from_clusters(self) -> None:
        result = _cluster_statistics(np.asarray([-1, -1, 0, 0, 1], dtype=np.int32))

        self.assertEqual(result["cluster_count"], 2)
        self.assertEqual(result["noise_count"], 2)
        self.assertAlmostEqual(result["noise_percentage"], 40.0)
        self.assertEqual(result["assigned_count"], 3)
        self.assertEqual(
            result["cluster_sizes"],
            [
                {"cluster": 0, "segments": 2, "percentage": 40.0},
                {"cluster": 1, "segments": 1, "percentage": 20.0},
            ],
        )

    def test_all_noise_has_zero_clusters_and_full_noise_percentage(self) -> None:
        result = _cluster_statistics(np.full(4, -1, dtype=np.int32))

        self.assertEqual(result["cluster_count"], 0)
        self.assertEqual(result["noise_count"], 4)
        self.assertEqual(result["noise_percentage"], 100.0)
        self.assertEqual(result["assigned_percentage"], 0.0)


if __name__ == "__main__":
    unittest.main()
