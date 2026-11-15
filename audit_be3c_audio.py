"""Audit every BE3C audio payload and export listening previews."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import welch


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "watkins-marine-mammal-full-cuts" / "data"
OUTPUT_DIR = PROJECT_DIR / "be3c_feasibility"
SPECIES_CODE = "BE3C"
MAX_PREVIEW_SECONDS = 10
CLIP_THRESHOLD = 0.999
SILENCE_RMS_THRESHOLD = 1e-6


def _read_manifest(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"parquet_file", "parquet_row_index", "split", "location_group"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        manifest: dict[tuple[str, int], dict[str, str]] = {}
        for row in reader:
            key = (row["parquet_file"], int(row["parquet_row_index"]))
            if key in manifest:
                raise ValueError(f"Duplicate source row in split manifest: {key}")
            manifest[key] = row
    return manifest


def _audio_metrics(
    payload: bytes, metadata_sample_rate: int | None
) -> dict[str, Any]:
    with sf.SoundFile(BytesIO(payload)) as audio_file:
        sample_rate = audio_file.samplerate
        frames = audio_file.frames
        channels = audio_file.channels
        samples = audio_file.read(dtype="float32", always_2d=True)
    if samples.size == 0:
        raise ValueError("Decoded audio has no samples")
    if not np.isfinite(samples).all():
        raise ValueError("Decoded audio contains non-finite samples")

    absolute_samples = np.abs(samples)
    peak = float(np.max(absolute_samples))
    mono = samples.mean(axis=1)
    frequencies, power = welch(
        mono, fs=sample_rate, nperseg=min(8192, len(mono))
    )
    total_power = float(np.sum(power))
    above_14khz_power = float(np.sum(power[frequencies > 14_000]))
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return {
        "audio_status": "decoded",
        "duration_seconds": frames / sample_rate,
        "sample_rate_hz": sample_rate,
        "metadata_sample_rate_hz": metadata_sample_rate,
        "sample_rate_mismatch": (
            metadata_sample_rate is not None
            and metadata_sample_rate != sample_rate
        ),
        "channels": channels,
        "frames": frames,
        "peak_amplitude": peak,
        "rms_amplitude": rms,
        "clipped_sample_fraction": float(
            np.mean(absolute_samples >= CLIP_THRESHOLD)
        ),
        "silent_by_rms_threshold": rms <= SILENCE_RMS_THRESHOLD,
        "dominant_frequency_hz": float(frequencies[int(np.argmax(power))]),
        "spectral_power_fraction_above_14khz": (
            above_14khz_power / total_power if total_power else 0.0
        ),
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot write an empty report: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames or list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def audit_be3c_audio(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    species_code: str = SPECIES_CODE,
) -> dict[str, Any]:
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet shards found in {data_dir}")
    manifest_path = output_dir / "be3c_location_split.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    manifest = _read_manifest(manifest_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    quality_rows: list[dict[str, Any]] = []
    seen_positions: set[tuple[str, int]] = set()
    payload_hashes: dict[str, list[str]] = defaultdict(list)
    record_number_counts: Counter[str] = Counter()

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        required_columns = {"species_code", "record_number", "sound", "audio"}
        if not required_columns.issubset(parquet_file.schema_arrow.names):
            raise ValueError(
                f"{parquet_path.name} is missing required columns "
                f"{sorted(required_columns - set(parquet_file.schema_arrow.names))}"
            )
        species_column_index = next(
            index
            for index in range(parquet_file.metadata.num_columns)
            if parquet_file.metadata.schema.column(index).path == "species_code"
        )
        row_offset = 0
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            row_group = parquet_file.metadata.row_group(row_group_index)
            statistics = row_group.column(species_column_index).statistics
            if (
                statistics is not None
                and statistics.has_min_max
                and not (statistics.min <= species_code <= statistics.max)
            ):
                row_offset += row_group.num_rows
                continue
            row_group_offset = 0
            for batch in parquet_file.iter_batches(
                batch_size=64,
                row_groups=[row_group_index],
                columns=["species_code", "record_number", "sound", "audio"],
            ):
                for row_index, record in enumerate(batch.to_pylist()):
                    if record.get("species_code") != species_code:
                        continue
                    row_in_group = row_group_offset + row_index
                    source_position = (
                        parquet_path.name,
                        row_offset + row_in_group,
                    )
                    split_record = manifest.get(source_position)
                    if split_record is None:
                        raise ValueError(
                            f"BE3C source row is missing from the split manifest: "
                            f"{source_position}"
                        )
                    seen_positions.add(source_position)
                    record_number = str(record.get("record_number") or "")
                    if record_number:
                        record_number_counts[record_number] += 1
                    sound = record.get("sound") or {}
                    payload = ((record.get("audio") or {}).get("bytes"))
                    result: dict[str, Any] = {
                        "species_code": species_code,
                        "record_number": record_number,
                        "split": split_record["split"],
                        "location_group": split_record["location_group"],
                        "parquet_file": parquet_path.name,
                        "parquet_row_index": source_position[1],
                        "metadata_sample_rate_hz": sound.get("sample_rate"),
                        "audio_status": "missing",
                        "duration_seconds": "",
                        "sample_rate_hz": "",
                        "sample_rate_mismatch": "",
                        "channels": "",
                        "frames": "",
                        "peak_amplitude": "",
                        "rms_amplitude": "",
                        "clipped_sample_fraction": "",
                        "silent_by_rms_threshold": "",
                        "dominant_frequency_hz": "",
                        "spectral_power_fraction_above_14khz": "",
                        "sha256": "",
                        "error": "",
                        "_row_group_index": row_group_index,
                        "_row_index_in_group": row_in_group,
                    }
                    if isinstance(payload, (bytes, bytearray, memoryview)) and payload:
                        payload = bytes(payload)
                        digest = hashlib.sha256(payload).hexdigest()
                        result["sha256"] = digest
                        payload_hashes[digest].append(record_number)
                        try:
                            result.update(
                                _audio_metrics(payload, sound.get("sample_rate"))
                            )
                        except (OSError, RuntimeError, ValueError) as error:
                            result["audio_status"] = "decode_error"
                            result["error"] = f"{type(error).__name__}: {error}"
                    elif isinstance(payload, (bytes, bytearray, memoryview)):
                        result["audio_status"] = "empty"
                    elif payload is not None:
                        result["audio_status"] = "unsupported_audio_payload"
                        result["error"] = (
                            "Expected non-empty audio bytes, got "
                            f"{type(payload).__name__}"
                        )
                    quality_rows.append(result)
                row_group_offset += batch.num_rows
            row_offset += row_group.num_rows
        print(f"Scanned {parquet_path.name}", flush=True)

    missing_manifest_positions = set(manifest) - seen_positions
    if missing_manifest_positions:
        raise ValueError(
            f"{len(missing_manifest_positions)} manifest rows were not found in "
            f"the source Parquet files; first: {min(missing_manifest_positions)}"
        )
    if not quality_rows:
        raise ValueError(f"No records found for species_code={species_code}")

    duplicate_hashes = {
        digest: record_numbers
        for digest, record_numbers in payload_hashes.items()
        if len(record_numbers) > 1
    }
    duplicate_record_numbers = {
        record_number
        for record_number, count in record_number_counts.items()
        if count > 1
    }
    for row in quality_rows:
        row["exact_duplicate_payload"] = len(
            payload_hashes.get(row["sha256"], [])
        ) > 1
        row["duplicate_record_number"] = (
            row["record_number"] in duplicate_record_numbers
        )

    report_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in quality_rows
    ]
    _write_csv(output_dir / "be3c_audio_quality.csv", report_rows)

    decoded_rows = [
        row for row in quality_rows if row["audio_status"] == "decoded"
    ]
    duration_values = [row["duration_seconds"] for row in decoded_rows]
    duration_percentiles = (
        np.percentile(duration_values, [10, 50, 90])
        if duration_values
        else None
    )
    duration_summary = (
        f"{duration_percentiles[0]:.2f} / {duration_percentiles[1]:.2f} / "
        f"{duration_percentiles[2]:.2f}; range "
        f"{min(duration_values):.2f}–{max(duration_values):.2f}"
        if duration_percentiles is not None
        else "unavailable (no audio decoded)"
    )
    status_counts = Counter(row["audio_status"] for row in quality_rows)
    sample_rate_counts = Counter(row["sample_rate_hz"] for row in decoded_rows)
    mismatch_rows = [row for row in decoded_rows if row["sample_rate_mismatch"]]
    clipped_rows = [
        row for row in decoded_rows if row["clipped_sample_fraction"] > 0
    ]
    materially_clipped_rows = [
        row for row in decoded_rows if row["clipped_sample_fraction"] >= 0.01
    ]
    silent_rows = [
        row for row in decoded_rows if row["silent_by_rms_threshold"]
    ]
    high_band_rows = [
        row
        for row in decoded_rows
        if row["spectral_power_fraction_above_14khz"] > 0.01
    ]
    strong_high_band_rows = [
        row
        for row in decoded_rows
        if row["spectral_power_fraction_above_14khz"] > 0.10
    ]
    high_band_dominant_rows = [
        row for row in decoded_rows if row["dominant_frequency_hz"] > 14_000
    ]
    exact_duplicate_records = sum(
        len(record_numbers) for record_numbers in duplicate_hashes.values()
    )

    preview_rows: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        split_rows = [row for row in decoded_rows if row["split"] == split]
        if not split_rows:
            continue
        durations = [row["duration_seconds"] for row in split_rows]
        median_duration = float(np.median(durations))
        typical = min(
            split_rows,
            key=lambda row: (
                abs(row["duration_seconds"] - median_duration),
                row["record_number"],
            ),
        )
        longest = max(
            split_rows,
            key=lambda row: (row["duration_seconds"], row["record_number"]),
        )
        for selection, selected in (("typical", typical), ("longest", longest)):
            filename = f"{split}_{selection}_{selected['record_number']}.wav"
            source = data_dir / selected["parquet_file"]
            audio_row = pq.ParquetFile(source).read_row_group(
                selected["_row_group_index"], columns=["audio"]
            ).column("audio")[selected["_row_index_in_group"]].as_py()
            with sf.SoundFile(BytesIO(audio_row["bytes"])) as audio_file:
                preview_frames = min(
                    audio_file.frames,
                    int(audio_file.samplerate * MAX_PREVIEW_SECONDS),
                )
                waveform = audio_file.read(
                    frames=preview_frames,
                    dtype="float32",
                    always_2d=True,
                )
                sample_rate = audio_file.samplerate
            preview_dir = output_dir / "be3c_audio_qc_examples"
            preview_dir.mkdir(parents=True, exist_ok=True)
            sf.write(
                preview_dir / filename,
                waveform,
                sample_rate,
                subtype="PCM_16",
            )
            preview_rows.append(
                {
                    "split": split,
                    "selection": selection,
                    "record_number": selected["record_number"],
                    "location_group": selected["location_group"],
                    "duration_seconds": selected["duration_seconds"],
                    "sample_rate_hz": selected["sample_rate_hz"],
                    "preview_seconds": len(waveform) / sample_rate,
                    "wav_file": filename,
                }
            )
    _write_csv(
        output_dir / "be3c_manual_listening.csv",
        preview_rows,
        fieldnames=[
            "split",
            "selection",
            "record_number",
            "location_group",
            "duration_seconds",
            "sample_rate_hz",
            "preview_seconds",
            "wav_file",
        ],
    )

    train_count = sum(row["split"] == "train" for row in quality_rows)
    validation_count = sum(row["split"] == "validation" for row in quality_rows)
    report_lines = [
        "# BE3C audio quality check",
        "",
        f"- Records checked against the location split: {len(quality_rows):,}.",
        f"- Train / validation: {train_count:,} / {validation_count:,}.",
        f"- Audio decode status: {dict(status_counts)}.",
        f"- Duration, seconds (10th / median / 90th percentile): {duration_summary}.",
        f"- Recordings shorter than 1 second: "
        f"{sum(value < 1 for value in duration_values):,}; shorter than 3 seconds: "
        f"{sum(value < 3 for value in duration_values):,}.",
        f"- Header sample rates: {dict(sorted(sample_rate_counts.items()))}.",
        f"- Header/metadata sample-rate mismatches: {len(mismatch_rows)}"
        + (
            f" ({', '.join(row['record_number'] for row in mismatch_rows)})."
            if mismatch_rows
            else "."
        ),
        f"- Silent by RMS threshold ({SILENCE_RMS_THRESHOLD:g}): {len(silent_rows)}.",
        f"- Any samples at or above {CLIP_THRESHOLD:g} full scale: "
        f"{len(clipped_rows)} recordings; recordings with at least 1% such samples: "
        f"{len(materially_clipped_rows)}.",
        f"- Recordings with >1% of measured spectral power above 14 kHz: "
        f"{len(high_band_rows)}; with >10%: {len(strong_high_band_rows)}; "
        f"dominant spectral peak above 14 kHz: {len(high_band_dominant_rows)}.",
        f"- Exact duplicate audio payloads: {len(duplicate_hashes)} duplicate groups "
        f"({exact_duplicate_records} records involved).",
        f"- Repeated non-empty record numbers: "
        f"{sum(count - 1 for count in record_number_counts.values() if count > 1)}.",
        "",
        "These checks confirm files decode and flag basic signal issues; they cannot "
        "confirm that clips contain identifiable whale calls or are suitable for a "
        "specific research question. Spectral power is not a call detector. Because "
        "many recordings contain measurable energy above 14 kHz, do not rely on a "
        "pretrained representation that discards that band as the only baseline. "
        "Listen to the provided examples before training.",
        "",
        "The original Parquet files were read only and were not modified.",
        "",
    ]
    report_path = output_dir / "be3c_audio_quality_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "record_count": len(quality_rows),
        "status_counts": dict(status_counts),
        "sample_rate_mismatches": len(mismatch_rows),
        "silent_count": len(silent_rows),
        "duplicate_payload_groups": len(duplicate_hashes),
        "duplicate_record_numbers": sum(
            count - 1 for count in record_number_counts.values() if count > 1
        ),
        "quality_report": output_dir / "be3c_audio_quality.csv",
        "summary_report": report_path,
        "preview_manifest": output_dir / "be3c_manual_listening.csv",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode BE3C audio, report basic quality, and export listenable clips."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--species-code", default=SPECIES_CODE)
    args = parser.parse_args()
    result = audit_be3c_audio(args.data_dir, args.output_dir, args.species_code)
    print(f"Checked {result['record_count']:,} recordings.")
    print(f"Audio statuses: {result['status_counts']}")
    print(f"Quality report: {result['summary_report']}")
    print(f"Per-record CSV: {result['quality_report']}")
    print(f"Listening list: {result['preview_manifest']}")


if __name__ == "__main__":
    main()
