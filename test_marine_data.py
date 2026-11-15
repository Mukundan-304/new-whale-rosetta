import csv
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from marine_data import (
    ParquetAudioDataset,
    SpeciesBalancedBatchSampler,
    analyze_dataset,
    discover_schema,
    evaluate_cluster_assignments,
    grouped_split_indices,
)


def make_wav() -> bytes:
    buffer = BytesIO()
    sf.write(buffer, np.zeros(800, dtype=np.float32), 8000, format="WAV")
    return buffer.getvalue()


class MarineDataTests(unittest.TestCase):
    def test_infers_fields_and_audits_quality_without_changing_source(self) -> None:
        audio = make_wav()
        rows = [
            {
                "species": "A",
                "audio_payload": {"data": audio, "uri": "a.wav"},
                "record_id": "one",
                "observation_date": "2000-01-01",
                "location": {"name": ["site-1"]},
                "sound": {"sample_rate": 8000},
            },
            {
                "species": "A",
                "audio_payload": {"data": audio, "uri": "a-copy.wav"},
                "record_id": "one",
                "observation_date": "2000-01-02",
                "location": {"name": ["site-1"]},
                "sound": {"sample_rate": 8000},
            },
            {
                "species": "B",
                "audio_payload": {"data": b"not a wav", "uri": None},
                "record_id": "three",
                "observation_date": None,
                "location": {"name": ["site-2"]},
                "sound": {"sample_rate": 8000},
            },
            {
                "species": None,
                "audio_payload": None,
                "record_id": "four",
                "observation_date": None,
                "location": None,
                "sound": None,
            },
        ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data"
            output = root / "reports"
            source.mkdir()
            shard = source / "part.parquet"
            pq.write_table(pa.Table.from_pylist(rows), shard)
            original_bytes = shard.read_bytes()

            schema = discover_schema(shard)
            self.assertEqual(schema.label, "species")
            self.assertEqual(schema.audio, "audio_payload")
            self.assertEqual(schema.record_id, "record_id")

            result = analyze_dataset(source, output, batch_size=2)
            self.assertEqual(result["record_count"], 4)
            self.assertEqual(result["quality"]["duplicate_record_ids"], 1)
            self.assertEqual(result["quality"]["duplicate_audio_payloads"], 1)
            self.assertEqual(result["quality"]["audio_corrupt"], 1)
            self.assertEqual(result["quality"]["audio_missing"], 1)
            self.assertEqual(shard.read_bytes(), original_bytes)

            with (output / "species_distribution.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                distribution = list(csv.DictReader(stream))
            species_a = next(row for row in distribution if row["species_label"] == "A")
            self.assertEqual(species_a["record_count"], "2")
            self.assertEqual(species_a["duration_available_count"], "2")

            self.assertTrue((output / "dataset_quality_report.csv").is_file())
            self.assertTrue((output / "species_distribution.png").is_file())
            self.assertTrue(
                (output / "species_duration_distribution.png").is_file()
            )
            self.assertTrue((output / "imbalance_strategy_report.md").is_file())
            with (output / "dataset_quality_report.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                quality = {
                    row["metric"]: row["value"] for row in csv.DictReader(stream)
                }
            self.assertEqual(quality["audio_missing"], "1")
            self.assertEqual(quality["audio_empty"], "0")
            self.assertEqual(quality["audio_corrupt"], "1")

            dataset = ParquetAudioDataset([shard], columns=["species", "audio_payload"])
            self.assertEqual(len(dataset), 4)
            self.assertEqual(dataset[0]["species"], "A")

    def test_balanced_sampler_is_class_balanced_without_repeated_indices(self) -> None:
        labels = ["common"] * 4 + ["rare"]
        sampler = SpeciesBalancedBatchSampler(
            labels, batch_size=2, classes_per_batch=2, seed=7
        )
        batches = list(sampler)
        used_indices = [index for batch in batches for index in batch]
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(set(used_indices)), len(used_indices))
        self.assertEqual(2, len(batches[0]))
        self.assertEqual({labels[index] for index in batches[0]}, {"common", "rare"})
        self.assertEqual(len(sampler), len(batches))

    def test_group_split_keeps_groups_together_and_missing_groups_in_train(self) -> None:
        groups = ["site-a", "site-a", "site-b", "site-b", "site-c", None]
        train, validation = grouped_split_indices(groups, seed=1)
        self.assertIn(5, train)
        self.assertTrue(set(train).isdisjoint(validation))
        for group in {"site-a", "site-b", "site-c"}:
            group_indices = {i for i, value in enumerate(groups) if value == group}
            self.assertTrue(
                group_indices.issubset(set(train))
                or group_indices.issubset(set(validation))
            )

    def test_cluster_evaluation_reports_tiers_and_domain_association(self) -> None:
        result = evaluate_cluster_assignments(
            ["A", "A", "B", "B", "C", "C"],
            [0, 0, 1, 1, -1, 2],
            {"A": "dominant", "B": "moderate", "C": "rare"},
            ["site-1", "site-1", "site-2", "site-2", "site-3", "site-3"],
        )
        self.assertEqual([row["tier"] for row in result], ["common", "moderate", "rare"])
        self.assertIn("cluster_domain_nmi", result[0])


if __name__ == "__main__":
    unittest.main()
