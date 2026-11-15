"""Unit tests for Stage 9: Temporal Sequence Analysis."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from watkins_stage9.run_stage9_temporal import (
    analyze_temporal_novelty_context,
    analyze_transitions_and_sequences,
    verify_sequence_continuity,
)


class WatkinsStage9TemporalTests(unittest.TestCase):
    def test_verify_sequence_continuity_identifies_multi_segment(self) -> None:
        synthetic_df = pd.DataFrame(
            {
                "recording_id": [1, 2, 2, 3, 3, 3],
                "segment_index": [0, 0, 1, 0, 1, 2],
            }
        )
        stats = verify_sequence_continuity(synthetic_df)

        self.assertEqual(stats["total_recordings"], 3)
        self.assertEqual(stats["single_segment_recordings"], 1)
        self.assertEqual(stats["multi_segment_recordings"], 2)
        self.assertTrue(stats["all_strictly_contiguous"])
        self.assertEqual(stats["max_segments_in_recording"], 3)

    def test_verify_sequence_continuity_detects_non_contiguous_gaps(self) -> None:
        synthetic_df = pd.DataFrame(
            {
                "recording_id": [1, 1],
                "segment_index": [0, 2],  # Missing index 1
            }
        )
        stats = verify_sequence_continuity(synthetic_df)
        self.assertFalse(stats["all_strictly_contiguous"])

    def test_analyze_transitions_calculates_self_loops_and_trigrams(self) -> None:
        synthetic_df = pd.DataFrame(
            {
                "recording_id": [101, 101, 101, 101, 102, 102],
                "cluster": [5, 5, 4, 5, 8, 8],
                "species_code": ["SP1", "SP1", "SP1", "SP1", "SP2", "SP2"],
                "location_domain": ["loc1"] * 6,
                "recording_duration_seconds": [20.0, 20.0, 20.0, 20.0, 10.0, 10.0],
                "novelty_score": [0.4, 0.4, 0.5, 0.4, 0.3, 0.3],
                "novelty_flag": [False, False, True, False, False, False],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics, trans_df, seq_df, rec_df = analyze_transitions_and_sequences(
                synthetic_df, Path(tmp_dir)
            )

            # In rec 101: (5,5), (5,4), (4,5) -> 3 transitions (1 self, 2 state changes)
            # In rec 102: (8,8) -> 1 transition (1 self)
            # Total transitions = 4 (2 self, 2 state changes)
            self.assertEqual(metrics["total_transitions"], 4)
            self.assertEqual(metrics["self_transitions"], 2)
            self.assertEqual(metrics["state_changes"], 2)
            self.assertAlmostEqual(metrics["self_transition_ratio"], 0.5)

            # Check trigrams: in rec 101: (5,5,4), (5,4,5)
            # Total trigrams = 2, alternating trigram (5,4,5) = 1
            self.assertEqual(metrics["total_trigrams"], 2)
            self.assertEqual(metrics["alternating_trigrams"], 1)

            self.assertTrue((Path(tmp_dir) / "temporal_transitions.csv").exists())
            self.assertTrue((Path(tmp_dir) / "common_sequences.csv").exists())
            self.assertTrue((Path(tmp_dir) / "recording_sequence_summary.csv").exists())

    def test_analyze_temporal_novelty_context(self) -> None:
        synthetic_df = pd.DataFrame(
            {
                "recording_id": [1, 1, 1, 2, 2, 2],
                "segment_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
                "cluster": [1, 1, 2, 3, 3, 3],
                "novelty_score": [0.6, 0.4, 0.4, 0.7, 0.3, 0.3],
                "novelty_flag": [True, False, False, True, False, False],
            }
        )
        res = analyze_temporal_novelty_context(synthetic_df)

        roles = res["position_roles"]
        self.assertEqual(roles["first"]["count"], 2)
        self.assertEqual(roles["first"]["flagged_count"], 2)
        self.assertAlmostEqual(roles["first"]["flag_rate_pct"], 100.0)
        self.assertEqual(roles["middle"]["flagged_count"], 0)
        self.assertEqual(roles["last"]["flagged_count"], 0)


if __name__ == "__main__":
    unittest.main()
