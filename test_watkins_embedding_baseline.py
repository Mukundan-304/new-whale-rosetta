import unittest

import numpy as np

from watkins_embedding_baseline import (
    _assign_location_groups,
    _association_metrics,
    _cluster_split,
    _sampled_silhouette,
    _species_label,
)


class WatkinsEmbeddingBaselineTests(unittest.TestCase):
    def test_location_groups_hold_sites_and_duplicate_payloads_together(self) -> None:
        rows = [
            {
                "parquet_file": "part.parquet",
                "parquet_row_index": index,
                "_entities": entities,
            }
            for index, entities in enumerate(
                [
                    {("name", "site-a")},
                    {("name", "site-a"), ("audio", "duplicate")},
                    {("audio", "duplicate")},
                    {("name", "site-b")},
                    set(),
                ]
            )
        ]

        groups, validation = _assign_location_groups(rows, validation_fraction=0.2)

        group_for_row = {
            index: group for group, indices in groups.items() for index in indices
        }
        self.assertEqual(group_for_row[0], group_for_row[1])
        self.assertEqual(group_for_row[1], group_for_row[2])
        self.assertNotEqual(group_for_row[3], group_for_row[0])
        self.assertNotIn(group_for_row[4], validation)
        self.assertTrue(validation)

    def test_validation_prefers_more_groups_when_size_is_equally_close(self) -> None:
        group_sizes = [4, 3, 2, 1, 1, 1]
        rows = []
        for group_index, group_size in enumerate(group_sizes):
            rows.extend(
                {
                    "parquet_file": "part.parquet",
                    "parquet_row_index": len(rows),
                    "_entities": {("name", f"site-{group_index}")},
                }
                for _ in range(group_size)
            )

        groups, validation = _assign_location_groups(
            rows, validation_fraction=5 / len(rows)
        )

        self.assertEqual(sum(len(groups[group]) for group in validation), 5)
        self.assertEqual(len(validation), 4)

    def test_validation_split_balances_species_across_location_groups(self) -> None:
        rows = []
        for species in ("A", "B", "C"):
            for site in range(2):
                rows.extend(
                    {
                        "parquet_file": "part.parquet",
                        "parquet_row_index": len(rows),
                        "species_label": species,
                        "imbalance_tier": "rare",
                        "_entities": {("name", f"{species}-site-{site}")},
                    }
                    for _ in range(2)
                )

        groups, validation = _assign_location_groups(
            rows, validation_fraction=0.5
        )
        validation_indices = {
            index for group in validation for index in groups[group]
        }
        validation_species = {
            rows[index]["species_label"] for index in validation_indices
        }
        training_species = {
            row["species_label"]
            for index, row in enumerate(rows)
            if index not in validation_indices
        }

        self.assertEqual(validation_species, {"A", "B", "C"})
        self.assertEqual(training_species, {"A", "B", "C"})

    def test_species_label_marks_missing_labels_without_inventing_a_species(self) -> None:
        self.assertEqual(_species_label({"species_code": "BE3C"}), "BE3C")
        self.assertEqual(
            _species_label({"display_name": "No code"}),
            "[UNLABELED] No code",
        )
        self.assertTrue(
            _species_label({}).startswith("[UNLABELED] missing species label")
        )

    def test_silhouette_is_skipped_when_sample_has_invalid_cluster_count(self) -> None:
        features = np.asarray([[0], [1], [2]], dtype=np.float32)
        self.assertIsNone(_sampled_silhouette(features, np.asarray([0, 0, 0])))
        self.assertIsNone(_sampled_silhouette(features, np.asarray([0, 1, 2])))
        self.assertIsInstance(
            _sampled_silhouette(features, np.asarray([0, 0, 1])), float
        )

    def test_cluster_fit_and_tier_evaluation_keep_site_heldout_groups_separate(
        self,
    ) -> None:
        rows = [
            {
                "split": "train",
                "location_group": "site-a",
                "source_sample_rate_hz": "48000",
                "record_number": f"train-{index}",
                "species_label": "COMMON",
                "imbalance_tier": "dominant",
            }
            for index in range(8)
        ] + [
            {
                "split": "validation",
                "location_group": "site-b",
                "source_sample_rate_hz": "48000",
                "record_number": f"validation-{index}",
                "species_label": "RARE",
                "imbalance_tier": "rare",
            }
            for index in range(3)
        ]
        features = np.asarray(
            [
                [0, 0],
                [0, 0.1],
                [0.1, 0],
                [0.1, 0.1],
                [5, 5],
                [5, 5.1],
                [5.1, 5],
                [5.1, 5.1],
                [0.05, 0.05],
                [5.05, 5.05],
                [5.02, 5.02],
            ],
            dtype=np.float32,
        )

        scores, assignments = _cluster_split(
            "full_band", features, rows, max_clusters=2
        )
        metrics = _association_metrics(
            "full_band", assignments, features
        )

        self.assertEqual(len(scores), 1)
        self.assertEqual(len(assignments), len(rows))
        self.assertEqual(
            {row["location_group"] for row in assignments if row["split"] == "train"},
            {"site-a"},
        )
        self.assertEqual(
            {row["location_group"] for row in assignments if row["split"] == "validation"},
            {"site-b"},
        )
        self.assertEqual(
            {(row["split"], row["imbalance_tier"]) for row in metrics},
            {("train", "dominant"), ("validation", "rare")},
        )


if __name__ == "__main__":
    unittest.main()
