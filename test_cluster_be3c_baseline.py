import unittest

import numpy as np

from cluster_be3c_baseline import _aggregate_recordings, _cluster_representation


class Be3cClusteringTests(unittest.TestCase):
    def test_chunk_aggregation_keeps_recording_split_and_averages_features(self) -> None:
        rows = [
            {
                "species_code": "BE3C",
                "record_number": "a",
                "split": "train",
                "location_group": "site-a",
                "parquet_file": "part.parquet",
                "parquet_row_index": "0",
                "source_sample_rate_hz": "48000",
                "spectral_power_fraction_above_14khz": "0.2",
            },
            {
                "species_code": "BE3C",
                "record_number": "a",
                "split": "train",
                "location_group": "site-a",
                "parquet_file": "part.parquet",
                "parquet_row_index": "0",
                "source_sample_rate_hz": "48000",
                "spectral_power_fraction_above_14khz": "0.2",
            },
            {
                "species_code": "BE3C",
                "record_number": "b",
                "split": "validation",
                "location_group": "site-b",
                "parquet_file": "part.parquet",
                "parquet_row_index": "1",
                "source_sample_rate_hz": "48000",
                "spectral_power_fraction_above_14khz": "0.1",
            },
        ]
        clap = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        full_band = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

        records, clap_records, full_band_records = _aggregate_recordings(
            rows, clap, full_band
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["chunk_count"], 2)
        np.testing.assert_allclose(clap_records[0], [1, 0])
        np.testing.assert_allclose(full_band_records[0], [2, 3])

    def test_clustering_fits_train_and_assigns_validation_once(self) -> None:
        rows = [
            {
                "split": "train",
                "location_group": "a",
                "source_sample_rate_hz": "48000",
                "record_number": f"train-{index}",
                "chunk_count": 1,
                "spectral_power_fraction_above_14khz": "0.1",
            }
            for index in range(4)
        ] + [
            {
                "split": "validation",
                "location_group": "b",
                "source_sample_rate_hz": "48000",
                "record_number": "validation-1",
                "chunk_count": 1,
                "spectral_power_fraction_above_14khz": "0.2",
            }
        ]
        features = np.asarray([[0, 0], [0, 0.1], [5, 5], [5, 5.1], [5, 5.2]])

        scores, assignments = _cluster_representation(
            "full_band", features, rows, max_clusters=2
        )

        self.assertEqual(len(scores), 1)
        self.assertEqual(len(assignments), len(rows))
        self.assertEqual(
            sum(row["split"] == "validation" for row in assignments), 1
        )
        self.assertEqual({row["selected_cluster_count"] for row in assignments}, {2})


if __name__ == "__main__":
    unittest.main()
