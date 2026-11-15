import unittest

import numpy as np

from be3c_embedding_baseline import (
    FULL_BAND_FEATURE_COUNT,
    TARGET_SAMPLE_RATE,
    _audio_chunks,
    _full_band_features,
    _resample_audio,
)


class Be3cEmbeddingBaselineTests(unittest.TestCase):
    def test_resampling_uses_audio_header_rate_and_returns_target_rate_length(self) -> None:
        audio = np.zeros(8000, dtype=np.float32)
        resampled = _resample_audio(audio, 8000)
        self.assertEqual(len(resampled), TARGET_SAMPLE_RATE)

    def test_chunks_cover_audio_once_without_overlap(self) -> None:
        audio = np.arange(TARGET_SAMPLE_RATE * 12, dtype=np.float32)
        chunks = _audio_chunks(audio, clip_seconds=10)
        self.assertEqual(len(chunks), 2)
        self.assertEqual((chunks[0][1], chunks[0][2]), (0, 10))
        self.assertEqual((chunks[1][1], chunks[1][2]), (10, 12))
        self.assertEqual(sum(len(chunk[3]) for chunk in chunks), len(audio))

    def test_full_band_feature_vector_is_finite_and_fixed_width(self) -> None:
        seconds = np.arange(TARGET_SAMPLE_RATE, dtype=np.float32) / TARGET_SAMPLE_RATE
        audio = np.sin(2 * np.pi * 18_000 * seconds).astype(np.float32)
        features = _full_band_features(audio)
        self.assertEqual(features.shape, (FULL_BAND_FEATURE_COUNT,))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(float(np.max(features)), 0.0)


if __name__ == "__main__":
    unittest.main()
