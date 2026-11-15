import unittest

import numpy as np

from make_be3c_cluster_previews import _choose_cluster_representatives


class Be3cClusterPreviewTests(unittest.TestCase):
    def test_selects_nearest_train_and_validation_examples(self) -> None:
        rows = [
            {
                "record_number": "train-near",
                "split": "train",
                "location_group": "site-a",
                "parquet_file": "part.parquet",
                "parquet_row_index": "0",
                "source_sample_rate_hz": "48000",
            },
            {
                "record_number": "train-far",
                "split": "train",
                "location_group": "site-a",
                "parquet_file": "part.parquet",
                "parquet_row_index": "1",
                "source_sample_rate_hz": "48000",
            },
            {
                "record_number": "validation-near",
                "split": "validation",
                "location_group": "site-b",
                "parquet_file": "part.parquet",
                "parquet_row_index": "2",
                "source_sample_rate_hz": "48000",
            },
            {
                "record_number": "validation-other-site",
                "split": "validation",
                "location_group": "site-c",
                "parquet_file": "part.parquet",
                "parquet_row_index": "3",
                "source_sample_rate_hz": "48000",
            },
        ]
        features = np.asarray([[0, 0], [10, 10], [0.1, 0.1], [0.2, 0.2]])
        assignments = [
            {"representation": "clap", "record_number": "train-near", "cluster": "0"},
            {"representation": "clap", "record_number": "train-far", "cluster": "1"},
            {
                "representation": "clap",
                "record_number": "validation-near",
                "cluster": "0",
            },
            {
                "representation": "clap",
                "record_number": "validation-other-site",
                "cluster": "0",
            },
        ]

        result = _choose_cluster_representatives(
            "clap", rows, features, assignments
        )

        self.assertEqual(len(result), 4)
        train_cluster_zero = next(
            row
            for row in result
            if row["split"] == "train" and row["cluster"] == 0
        )
        self.assertEqual(train_cluster_zero["record_number"], "train-near")
        validation_cluster_zero = next(
            row
            for row in result
            if row["split"] == "validation" and row["cluster"] == 0
        )
        self.assertEqual(
            validation_cluster_zero["record_number"], "validation-near"
        )
        self.assertIn(
            "site-c",
            {
                row["location_group"]
                for row in result
                if row["split"] == "validation"
            },
        )


if __name__ == "__main__":
    unittest.main()
