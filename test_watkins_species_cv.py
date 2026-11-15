import unittest

from watkins_species_cv import _make_splits, _score


class WatkinsSpeciesCvTests(unittest.TestCase):
    def test_grouped_splits_keep_groups_disjoint(self) -> None:
        labels = ["A", "A", "B", "B", "A", "B"]
        groups = ["site-1", "site-1", "site-2", "site-2", "site-3", "site-3"]

        for train_indices, test_indices in _make_splits(
            labels, groups, split_count=3, seed=29
        ):
            train_groups = {groups[index] for index in train_indices}
            test_groups = {groups[index] for index in test_indices}
            self.assertFalse(train_groups & test_groups)

    def test_score_reports_species_unseen_in_training_as_coverage_loss(self) -> None:
        metrics = _score(
            y_true=["A", "A", "B", "C"],
            y_predicted=["A", "B", "B", "A"],
            train_labels={"A", "B"},
        )

        self.assertEqual(metrics["seen_record_count"], 3)
        self.assertEqual(metrics["unseen_species_record_count"], 1)
        self.assertEqual(metrics["species_coverage"], 0.75)
        self.assertEqual(metrics["accuracy_all_records"], 0.5)
        self.assertAlmostEqual(metrics["accuracy_seen_species"], 2 / 3)

    def test_score_rejects_empty_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty validation"):
            _score([], [], {"A"})


if __name__ == "__main__":
    unittest.main()
