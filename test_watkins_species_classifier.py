import unittest

from watkins_species_classifier import _classification_metrics


class WatkinsSpeciesClassifierTests(unittest.TestCase):
    def test_metrics_separate_unseen_species_from_seen_species_performance(self) -> None:
        summary, per_species = _classification_metrics(
            y_true=["A", "A", "B", "C"],
            y_predicted=["A", "B", "B", "A"],
            train_labels=["A", "A", "B"],
        )

        self.assertEqual(summary["train_species_count"], 2)
        self.assertEqual(summary["validation_unseen_species_count"], 1)
        self.assertEqual(summary["validation_records_seen_in_train_species"], 3)
        self.assertEqual(summary["validation_records_from_unseen_species"], 1)
        self.assertEqual(summary["validation_species_coverage"], 0.75)
        self.assertEqual(summary["validation_accuracy_all_labeled"], 0.5)
        self.assertAlmostEqual(summary["validation_accuracy_seen_species"], 2 / 3)
        self.assertEqual(
            {row["species_label"]: row["seen_in_training"] for row in per_species},
            {"A": True, "B": True, "C": False},
        )

    def test_metrics_reject_mismatched_predictions(self) -> None:
        with self.assertRaisesRegex(ValueError, "counts must match"):
            _classification_metrics(["A"], [], ["A"])


if __name__ == "__main__":
    unittest.main()
