import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from audit_be3c_audio import _audio_metrics, audit_be3c_audio


def make_wav(
    duration_seconds: float, sample_rate: int = 8000, frequency: int = 440
) -> bytes:
    sample_count = int(duration_seconds * sample_rate)
    time = np.arange(sample_count) / sample_rate
    waveform = 0.2 * np.sin(2 * np.pi * frequency * time)
    import io

    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    return buffer.getvalue()


class Be3cAudioAuditTests(unittest.TestCase):
    def test_audio_metrics_report_duration_and_metadata_mismatch(self) -> None:
        result = _audio_metrics(make_wav(1), metadata_sample_rate=16000)
        self.assertEqual(result["audio_status"], "decoded")
        self.assertEqual(result["duration_seconds"], 1)
        self.assertEqual(result["sample_rate_hz"], 8000)
        self.assertTrue(result["sample_rate_mismatch"])
        self.assertFalse(result["silent_by_rms_threshold"])
        self.assertEqual(result["spectral_power_fraction_above_14khz"], 0)

    def test_audit_matches_split_and_writes_quality_and_listening_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            output_dir = root / "reports"
            data_dir.mkdir()
            shard = data_dir / "train-00000-of-00001.parquet"
            audio = make_wav(1)
            rows = [
                {
                    "species_code": "BE3C",
                    "record_number": "train-001",
                    "sound": {"sample_rate": 8000},
                    "audio": {"bytes": audio, "path": None},
                },
                {
                    "species_code": "BE3C",
                    "record_number": "validation-001",
                    "sound": {"sample_rate": 8000},
                    "audio": {
                        "bytes": make_wav(1, frequency=880),
                        "path": None,
                    },
                },
                {
                    "species_code": "OTHER",
                    "record_number": "other-001",
                    "sound": {"sample_rate": 8000},
                    "audio": {"bytes": audio, "path": None},
                },
            ]
            pq.write_table(pa.Table.from_pylist(rows), shard)
            output_dir.mkdir()
            with (output_dir / "be3c_location_split.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "parquet_file",
                        "parquet_row_index",
                        "split",
                        "location_group",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "parquet_file": shard.name,
                            "parquet_row_index": 0,
                            "split": "train",
                            "location_group": "group-a",
                        },
                        {
                            "parquet_file": shard.name,
                            "parquet_row_index": 1,
                            "split": "validation",
                            "location_group": "group-b",
                        },
                    ]
                )

            result = audit_be3c_audio(data_dir, output_dir)

            self.assertEqual(result["record_count"], 2)
            self.assertEqual(result["status_counts"], {"decoded": 2})
            with (output_dir / "be3c_audio_quality.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                quality = list(csv.DictReader(stream))
            self.assertEqual([row["split"] for row in quality], ["train", "validation"])
            with (output_dir / "be3c_manual_listening.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                previews = list(csv.DictReader(stream))
            self.assertEqual(len(previews), 4)
            self.assertTrue(
                all(
                    (output_dir / "be3c_audio_qc_examples" / row["wav_file"]).is_file()
                    for row in previews
                )
            )
            self.assertTrue((output_dir / "be3c_audio_quality_report.md").is_file())

    def test_audit_reports_corrupt_audio_without_claiming_decode_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            output_dir = root / "reports"
            data_dir.mkdir()
            output_dir.mkdir()
            shard = data_dir / "train-00000-of-00001.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "species_code": "BE3C",
                            "record_number": "broken-001",
                            "sound": {"sample_rate": 8000},
                            "audio": {"bytes": b"not a wav", "path": None},
                        }
                    ]
                ),
                shard,
            )
            with (output_dir / "be3c_location_split.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "parquet_file",
                        "parquet_row_index",
                        "split",
                        "location_group",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "parquet_file": shard.name,
                        "parquet_row_index": 0,
                        "split": "train",
                        "location_group": "group-a",
                    }
                )

            result = audit_be3c_audio(data_dir, output_dir)

            self.assertEqual(result["status_counts"], {"decode_error": 1})
            self.assertIn("no audio decoded", result["summary_report"].read_text())
            with (output_dir / "be3c_manual_listening.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])


if __name__ == "__main__":
    unittest.main()
