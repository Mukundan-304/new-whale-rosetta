"""Focused tests for Stage 4 manifest and resume helpers."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from watkins_pretrained_embeddings import (
    _prepare_store,
    _select_smoke_indices,
)


class WatkinsPretrainedEmbeddingTests(unittest.TestCase):
    def test_prepare_store_resumes_only_matching_manifest_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "train.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["segment_id", "split"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"segment_id": "a", "split": "train"},
                        {"segment_id": "b", "split": "train"},
                    ]
                )
            rows = [
                {"segment_id": "a", "split": "train"},
                {"segment_id": "b", "split": "train"},
            ]
            embeddings, completed, progress_path = _prepare_store(
                root / "out", rows, manifest, 4, "fingerprint-1"
            )
            embeddings[0] = np.asarray([1, 2, 3, 4], dtype=np.float32)
            embeddings.flush()
            completed[0] = True
            completed.flush()
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress["fingerprint"], "fingerprint-1")

            resumed_embeddings, resumed_completed, _ = _prepare_store(
                root / "out", rows, manifest, 4, "fingerprint-1"
            )
            self.assertTrue(bool(resumed_completed[0]))
            self.assertFalse(bool(resumed_completed[1]))
            np.testing.assert_array_equal(
                resumed_embeddings[0], np.asarray([1, 2, 3, 4], dtype=np.float32)
            )
            with self.assertRaisesRegex(ValueError, "mismatch"):
                _prepare_store(root / "out", rows, manifest, 4, "fingerprint-2")
            for value in (
                embeddings,
                completed,
                resumed_embeddings,
                resumed_completed,
            ):
                value._mmap.close()

    def test_smoke_selection_uses_distinct_recordings_and_is_deterministic(self) -> None:
        manifests = {
            "train": [
                {"recording_id": "r1"},
                {"recording_id": "r1"},
                {"recording_id": "r2"},
                {"recording_id": "r3"},
            ],
            "validation": [
                {"recording_id": "v1"},
                {"recording_id": "v2"},
            ],
            "test": [{"recording_id": "t1"}],
        }
        selected = _select_smoke_indices(manifests, 2, 29)
        self.assertEqual(selected, _select_smoke_indices(manifests, 2, 29))
        for split, indices in selected.items():
            self.assertLessEqual(len(indices), 2)
            self.assertEqual(
                len({manifests[split][index]["recording_id"] for index in indices}),
                len(indices),
            )


if __name__ == "__main__":
    unittest.main()
