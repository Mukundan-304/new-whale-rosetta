import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from preprocess_watkins_audio import (
    PreprocessingConfig,
    _log_mel_spectrogram,
    _mel_filterbank,
    _downmix_to_mono,
    _peak_normalize,
    _resample_channels,
    _segment_audio,
    run_preprocessing,
)


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class WatkinsPreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config = PreprocessingConfig(
            data_dir=self.root / "source",
            output_dir=self.root / "processed",
            target_sample_rate_hz=8000,
            segment_duration_seconds=2.0,
            n_fft=256,
            hop_length=64,
            n_mels=16,
            f_min_hz=20.0,
            f_max_hz=4000.0,
            log_power_floor=1e-12,
            normalization_method="peak",
            normalization_target_peak=0.99,
            records_per_partition=2,
            feature_dtype="float16",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _create_dataset(self) -> None:
        self.config.data_dir.mkdir()
        rate = 8000
        short = np.sin(
            2 * np.pi * 440 * np.arange(rate, dtype=np.float32) / rate
        )
        stereo = np.column_stack((short, short * 0.5))
        typical = np.sin(
            2 * np.pi * 700 * np.arange(rate * 2 + rate // 2, dtype=np.float32)
            / rate
        )
        long = np.sin(
            2 * np.pi * 1200 * np.arange(rate * 4 + rate // 2, dtype=np.float32)
            / rate
        )
        records = [
            {
                "record_number": "short-record",
                "species_code": "BE3C",
                "display_name": "Pilot whale",
                "scientific_name": "Globicephala melas",
                "mid": "whoi0039",
                "class_id": 39,
                "observation_date": "2000-01-01",
                "location": {
                    "name": ["Test Bay"],
                    "coordinates": [{"lat": 1.0, "lon": 2.0}],
                },
                "sound": {
                    "sample_rate": 16000,
                    "channel": {"recorded": 2},
                },
                "signal": {"quality": 1},
                "audio": {"bytes": _wav_bytes(stereo, rate)},
            },
            {
                "record_number": "typical-record",
                "species_code": "BA2A",
                "display_name": "Sperm whale",
                "scientific_name": "Physeter macrocephalus",
                "mid": "whoi0051",
                "class_id": 51,
                "observation_date": "2000-01-02",
                "location": {"name": ["Other Bay"], "coordinates": []},
                "sound": {"sample_rate": rate, "channel": {}},
                "signal": {},
                "audio": {"bytes": _wav_bytes(typical, rate)},
            },
            {
                "record_number": "long-record",
                "species_code": "BE3C",
                "display_name": "Pilot whale",
                "scientific_name": "Globicephala melas",
                "mid": "whoi0039",
                "class_id": 39,
                "observation_date": "2000-01-03",
                "location": {"name": ["Test Bay"], "coordinates": []},
                "sound": {"sample_rate": rate, "channel": {}},
                "signal": {},
                "audio": {"bytes": _wav_bytes(long, rate)},
            },
            {
                "record_number": "broken-record",
                "species_code": "BE3C",
                "display_name": "Pilot whale",
                "scientific_name": "Globicephala melas",
                "mid": "whoi0039",
                "class_id": 39,
                "observation_date": "2000-01-04",
                "location": {"name": [], "coordinates": []},
                "sound": {"sample_rate": rate, "channel": {}},
                "signal": {},
                "audio": {"bytes": b"not a wav file"},
            },
        ]
        table = pa.Table.from_pylist(records)
        pq.write_table(table, self.config.data_dir / "train-00000.parquet")

    def test_feature_shape_and_finite_values(self) -> None:
        filters, centers = _mel_filterbank(8000, 256, 16, 20, 4000)
        self.assertEqual(filters.shape, (16, 129))
        self.assertEqual(centers.shape, (16,))

        config = self.config
        segment = np.ones(config.segment_samples, dtype=np.float32) * 0.1
        features = _log_mel_spectrogram(segment, config, filters)

        self.assertEqual(features.shape, (16, config.frame_count))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features.dtype, np.float16)

    def test_stereo_resampling_and_short_segment_padding(self) -> None:
        source = np.ones((800, 2), dtype=np.float32)
        resampled = _resample_channels(source, 8000, 8000)
        self.assertEqual(resampled.shape, (800, 2))
        mono = _downmix_to_mono(resampled)
        self.assertEqual(mono.shape, (800,))
        np.testing.assert_allclose(mono, 1.0)
        segments = list(_segment_audio(mono, segment_samples=1600))
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][1], 800)
        self.assertEqual(segments[0][2].shape, (1600,))
        self.assertTrue(np.all(segments[0][2][800:] == 0))

    def test_peak_normalization_preserves_silence_and_reports_gain(self) -> None:
        waveform = np.asarray([0.0, -0.25, 0.5], dtype=np.float32)
        normalized, gain = _peak_normalize(waveform, 0.99)
        self.assertAlmostEqual(gain, 1.98, places=5)
        self.assertAlmostEqual(float(np.max(np.abs(normalized))), 0.99, places=5)
        np.testing.assert_allclose(normalized, waveform * gain)

        silence, silence_gain = _peak_normalize(
            np.zeros(3, dtype=np.float32), 0.99
        )
        self.assertEqual(silence_gain, 1.0)
        self.assertTrue(np.all(silence == 0))

    def test_full_run_logs_corruption_and_reuses_cache(self) -> None:
        self._create_dataset()

        first = run_preprocessing(self.config)
        self.assertEqual(first["record_count"], 3)
        self.assertEqual(first["segment_count"], 6)
        self.assertEqual(first["failure_count"], 1)
        with first["manifest_path"].open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {row["recording_id"] for row in rows},
            {"short-record", "typical-record", "long-record"},
        )
        self.assertTrue(
            any(
                row["recording_id"] == "short-record"
                and row["segment_padded"] == "True"
                for row in rows
            )
        )
        self.assertTrue(
            all(row["species_label"] for row in rows)
        )
        self.assertTrue(
            all(row["normalization_method"] == "peak" for row in rows)
        )
        self.assertTrue(
            all(
                abs(float(row["normalized_recording_peak_amplitude"]) - 0.99)
                < 1e-5
                for row in rows
                if row["recording_peak_amplitude"] != "0.0"
            )
        )
        with np.load(
            self.config.output_dir
            / "features"
            / rows[0]["feature_partition"]
        ) as stored:
            self.assertTrue(
                np.isfinite(stored["log_mel"]).all()
            )

        second = run_preprocessing(self.config)
        with (self.config.output_dir / "preprocessing_validation.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            report = {
                row["metric"]: row["value"] for row in csv.DictReader(stream)
            }
        self.assertEqual(second["segment_count"], 6)
        self.assertEqual(report["cache_partitions_written_this_run"], "0")
        self.assertGreater(int(report["cache_partitions_reused"]), 0)
        self.assertTrue(
            (self.config.output_dir / "spectrogram_previews" / "short_segment.png").is_file()
        )
        self.assertTrue(
            json.loads(
                (self.config.output_dir / "preprocessing_config_used.json").read_text(
                    encoding="utf-8"
                )
            )["config_digest"]
        )

    def test_config_rejects_mel_range_above_nyquist(self) -> None:
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            PreprocessingConfig(
                **{**self.config.__dict__, "f_max_hz": 5000}
            ).validate()


if __name__ == "__main__":
    unittest.main()
