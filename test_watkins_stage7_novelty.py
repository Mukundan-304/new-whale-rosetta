from __future__ import annotations

import unittest

from watkins_stage7.detect_novelty import _priority_for_segment
from watkins_stage7.extract_stage7_audio import _frame_window


class WatkinsStage7NoveltyTests(unittest.TestCase):
    def test_priority_requires_noise_and_novelty_for_high(self) -> None:
        self.assertEqual(_priority_for_segment(-1, True), ("high", "noise"))
        self.assertEqual(_priority_for_segment(-1, False), ("review", "noise"))
        self.assertEqual(_priority_for_segment(4, True), ("medium", "clustered"))
        self.assertEqual(_priority_for_segment(4, False), ("normal", "clustered"))
        self.assertEqual(
            _priority_for_segment(None, True),
            ("medium", "not_clustered_by_stage6"),
        )
        self.assertEqual(
            _priority_for_segment(None, False),
            ("not_clustered", "not_clustered_by_stage6"),
        )

    def test_audio_frame_window_uses_source_rate_and_valid_duration(self) -> None:
        self.assertEqual(_frame_window(1.5, 0.75, 4, 20), (6, 3))

    def test_audio_frame_window_rejects_out_of_range_offsets(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            _frame_window(5.0, 1.0, 4, 20)


if __name__ == "__main__":
    unittest.main()
