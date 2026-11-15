from __future__ import annotations

import argparse
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = Path(__file__).resolve().parent / "cluster_novelty_results.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "audio_samples"


def _frame_window(
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int,
    total_frames: int,
) -> tuple[int, int]:
    if (
        not np.isfinite(start_seconds)
        or not np.isfinite(duration_seconds)
        or start_seconds < 0
        or duration_seconds <= 0
        or sample_rate <= 0
        or total_frames <= 0
    ):
        raise ValueError("Audio crop requires finite non-negative start and positive duration/rate.")
    start_frame = int(round(start_seconds * sample_rate))
    end_frame = min(total_frames, start_frame + int(round(duration_seconds * sample_rate)))
    if start_frame >= total_frames or end_frame <= start_frame:
        raise ValueError("Requested segment offset is outside the source recording.")
    return start_frame, end_frame - start_frame


def _read_parquet_record(audio_path: Path, row_index: int) -> dict[str, Any]:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Embedded-audio Parquet file not found: {audio_path}")
    if row_index < 0:
        raise ValueError("Parquet row index must be non-negative.")
    parquet = pq.ParquetFile(audio_path)
    row_offset = 0
    for batch in parquet.iter_batches(
        columns=["record_number", "audio"], batch_size=16
    ):
        if row_offset <= row_index < row_offset + batch.num_rows:
            return batch.slice(row_index - row_offset, 1).to_pylist()[0]
        row_offset += batch.num_rows
    raise IndexError(f"Parquet row {row_index} was not found in {audio_path}.")


def extract_segment(
    segment_id: str,
    results_csv: Path = DEFAULT_RESULTS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    metadata = pd.read_csv(
        results_csv,
        dtype={"segment_id": "string", "recording_id": "string"},
        usecols=[
            "segment_id",
            "recording_id",
            "audio_path",
            "audio_storage_type",
            "audio_parquet_row_index",
            "segment_start_seconds",
            "segment_valid_seconds",
            "source_sample_rate_hz",
            "channel_count",
        ],
    )
    matches = metadata.loc[metadata["segment_id"] == segment_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one row for segment {segment_id!r}, found {len(matches)}."
        )
    row = matches.iloc[0]
    if str(row["audio_storage_type"]) != "audio_embedded_in_parquet":
        raise ValueError(
            f"Unsupported audio storage type: {row['audio_storage_type']!r}"
        )
    audio_path = Path(str(row["audio_path"]))
    audio_row = int(row["audio_parquet_row_index"])
    record = _read_parquet_record(audio_path, audio_row)
    if str(record.get("record_number", "")).strip() != str(row["recording_id"]).strip():
        raise ValueError(
            f"Recording ID mismatch for {segment_id}: expected {row['recording_id']}, "
            f"found {record.get('record_number')}"
        )
    audio_struct = record.get("audio") or {}
    payload = audio_struct.get("bytes")
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
        raise ValueError(f"Missing embedded audio bytes for {segment_id}.")
    with sf.SoundFile(BytesIO(payload)) as source:
        source_rate = int(source.samplerate)
        channels = int(source.channels)
        source_frames = int(source.frames)
        if source_rate != int(row["source_sample_rate_hz"]):
            raise ValueError(
                f"Sample-rate mismatch for {segment_id}: Parquet={source_rate}, "
                f"metadata={row['source_sample_rate_hz']}."
            )
        if channels != int(row["channel_count"]):
            raise ValueError(
                f"Channel-count mismatch for {segment_id}: Parquet={channels}, "
                f"metadata={row['channel_count']}."
            )
        start_frame, frame_count = _frame_window(
            float(row["segment_start_seconds"]),
            float(row["segment_valid_seconds"]),
            source_rate,
            source_frames,
        )
        source.seek(start_frame)
        samples = source.read(frames=frame_count, dtype="float32", always_2d=True)
    if samples.shape != (frame_count, channels) or not np.isfinite(samples).all():
        raise ValueError(f"Decoded crop is invalid for {segment_id}: {samples.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", segment_id)
    wav_path = output_dir / f"{safe_id}.wav"
    sf.write(wav_path, samples, source_rate, subtype="PCM_16")
    sidecar = {
        "segment_id": segment_id,
        "recording_id": str(row["recording_id"]),
        "source_parquet": str(audio_path),
        "source_parquet_row": audio_row,
        "segment_start_seconds": float(row["segment_start_seconds"]),
        "segment_valid_seconds": float(row["segment_valid_seconds"]),
        "source_sample_rate_hz": source_rate,
        "source_channels": channels,
        "written_frames": int(len(samples)),
        "output_encoding": "PCM_16 WAV crop; original Parquet audio is unchanged",
    }
    wav_path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )
    return wav_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract one scored segment from audio embedded in a Parquet row."
    )
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    path = extract_segment(
        arguments.segment_id, arguments.results_csv, arguments.output_dir
    )
    print(path)


if __name__ == "__main__":
    main()
