import tempfile
import unittest
from pathlib import Path

import numpy as np

from split_watkins_dataset import (
    _species_coverage,
    assign_grouped_splits,
    validate_split_rows,
)


def _recording(
    recording_id: str,
    species: str,
    group: str,
    segment_count: int = 1,
) -> dict:
    return {
        "recording_id": recording_id,
        "species_label": species,
        "source_audio_sha256": group,
        "location_group": group,
        "segments": [f"{recording_id}:{index}" for index in range(segment_count)],
        "segment_rows": [],
    }


class WatkinsSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recordings = [
            _recording(f"record-{index}", f"species-{index % 3}", f"group-{index}")
            for index in range(30)
        ]

    def test_group_separation_and_reproducibility(self) -> None:
        first, _ = assign_grouped_splits(
            self.recordings, "source_audio_sha256", seed=29, fold_count=6
        )
        second, _ = assign_grouped_splits(
            self.recordings, "source_audio_sha256", seed=29, fold_count=6
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "validation", "test"})
        split_fractions = {
            split: sum(value == split for value in first.values()) / len(first)
            for split in ("train", "validation", "test")
        }
        self.assertGreaterEqual(split_fractions["train"], 0.55)
        self.assertLessEqual(split_fractions["train"], 0.80)
        for split in ("validation", "test"):
            self.assertGreaterEqual(split_fractions[split], 0.10)
            self.assertLessEqual(split_fractions[split], 0.25)

    def test_duplicate_source_audio_stays_in_one_split(self) -> None:
        records = [dict(record) for record in self.recordings]
        records[1]["source_audio_sha256"] = records[0]["source_audio_sha256"]
        splits, _ = assign_grouped_splits(
            records, "source_audio_sha256", seed=7, fold_count=6
        )
        self.assertEqual(splits["record-0"], splits["record-1"])

    def test_rare_species_are_retained_without_forced_replication(self) -> None:
        records = self.recordings + [
            _recording("rare-record", "rare-species", "rare-group"),
            _recording("feasible-rare-1", "feasible-rare", "feasible-group-1"),
            _recording("feasible-rare-2", "feasible-rare", "feasible-group-2"),
            _recording("feasible-rare-3", "feasible-rare", "feasible-group-3"),
        ]
        splits, _ = assign_grouped_splits(
            records, "source_audio_sha256", seed=11, fold_count=6
        )
        self.assertEqual(len(splits), len(records))
        self.assertEqual(len(set(splits.values())), 3)
        rare = next(
            item
            for item in _species_coverage(records, splits)
            if item["species"] == "rare-species"
        )
        self.assertEqual(rare["independent_audio_groups"], 1)
        self.assertEqual(len(rare["splits_absent"]), 2)
        feasible_splits = {
            splits[record["recording_id"]]
            for record in records
            if record["species_label"] == "feasible-rare"
        }
        self.assertEqual(feasible_splits, {"train", "validation", "test"})

    def test_validation_rejects_duplicate_segment_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "source.parquet"
            feature = root / "feature.npz"
            audio.write_bytes(b"placeholder")
            np.savez(feature, log_mel=np.zeros((1, 2, 2), dtype=np.float16))
            base = {
                "recording_id": "record-1",
                "source_audio_sha256": "a" * 64,
                "audio_path": str(audio),
                "feature_path": str(feature),
                "audio_parquet_row_index": "0",
                "feature_index": "0",
            }
            rows = {
                "train": [
                    dict(base, segment_id="segment-1", source_audio_sha256="a" * 64)
                ],
                "validation": [
                    dict(base, segment_id="segment-1", source_audio_sha256="b" * 64)
                ],
                "test": [],
            }
            with self.assertRaisesRegex(ValueError, "Duplicate segment_id"):
                validate_split_rows(rows, "source_audio_sha256")

    def test_validation_rejects_missing_files_and_group_leakage(self) -> None:
        row = {
            "segment_id": "segment-1",
            "recording_id": "record-1",
            "source_audio_sha256": "a" * 64,
            "location_group": "site-1",
            "audio_path": r"Z:\missing\source.parquet",
            "feature_path": r"Z:\missing\feature.npz",
            "audio_parquet_row_index": "0",
            "feature_index": "0",
        }
        with self.assertRaisesRegex(FileNotFoundError, "Missing audio_path"):
            validate_split_rows(
                {"train": [row], "validation": [], "test": []},
                "location_group",
            )

    def test_validation_rejects_duplicate_audio_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "source.parquet"
            feature = root / "feature.npz"
            audio.write_bytes(b"placeholder")
            np.savez(
                feature,
                log_mel=np.zeros((2, 2, 2), dtype=np.float16),
            )
            train_row = {
                "segment_id": "segment-1",
                "recording_id": "record-1",
                "source_audio_sha256": "a" * 64,
                "location_group": "site-1",
                "audio_path": str(audio),
                "feature_path": str(feature),
                "audio_parquet_row_index": "0",
                "feature_index": "0",
            }
            test_row = {
                **train_row,
                "segment_id": "segment-2",
                "recording_id": "record-2",
                "location_group": "site-2",
                "audio_parquet_row_index": "1",
                "feature_index": "1",
            }
            with self.assertRaisesRegex(
                ValueError, "source_audio_sha256 values"
            ):
                validate_split_rows(
                    {
                        "train": [train_row],
                        "validation": [],
                        "test": [test_row],
                    },
                    "location_group",
                )


if __name__ == "__main__":
    unittest.main()
