"""Extract frozen CLAP and full-band acoustic features for the BE3C split."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from scipy.signal import resample_poly, welch


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "watkins-marine-mammal-full-cuts" / "data"
OUTPUT_DIR = PROJECT_DIR / "be3c_feasibility"
SPECIES_CODE = "BE3C"
MODEL_ID = "laion/clap-htsat-unfused"
TARGET_SAMPLE_RATE = 48_000
CLIP_SECONDS = 10
FULL_BAND_FEATURE_COUNT = 64


def _read_csv_by_source_row(
    path: Path,
) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows: dict[tuple[str, int], dict[str, str]] = {}
        for row in csv.DictReader(stream):
            key = (row["parquet_file"], int(row["parquet_row_index"]))
            if key in rows:
                raise ValueError(f"Duplicate source row in {path}: {key}")
            rows[key] = row
    return rows


def _resample_audio(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate <= 0:
        raise ValueError(f"Invalid source sample rate: {source_rate}")
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_rate == TARGET_SAMPLE_RATE:
        return mono
    ratio = Fraction(TARGET_SAMPLE_RATE, source_rate)
    return resample_poly(mono, ratio.numerator, ratio.denominator).astype(
        np.float32
    )


def _audio_chunks(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    clip_seconds: int = CLIP_SECONDS,
) -> list[tuple[int, float, float, np.ndarray]]:
    if sample_rate <= 0 or clip_seconds <= 0:
        raise ValueError("sample_rate and clip_seconds must be positive")
    if audio.size == 0:
        return []
    frames_per_clip = sample_rate * clip_seconds
    chunks = []
    for chunk_index, start_frame in enumerate(range(0, len(audio), frames_per_clip)):
        end_frame = min(start_frame + frames_per_clip, len(audio))
        chunks.append(
            (
                chunk_index,
                start_frame / sample_rate,
                end_frame / sample_rate,
                audio[start_frame:end_frame],
            )
        )
    return chunks


def _full_band_features(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    feature_count: int = FULL_BAND_FEATURE_COUNT,
) -> np.ndarray:
    if audio.size == 0 or sample_rate <= 0 or feature_count <= 0:
        raise ValueError("Audio and feature configuration must be non-empty")
    frequencies, power = welch(
        audio,
        fs=sample_rate,
        nperseg=min(8192, len(audio)),
    )
    upper_frequency = sample_rate / 2
    edges = np.geomspace(50, upper_frequency, feature_count + 1)
    bin_power = np.zeros(feature_count, dtype=np.float64)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (frequencies >= lower) & (frequencies < upper)
        if index == feature_count - 1:
            mask |= frequencies == upper_frequency
        if np.any(mask):
            bin_power[index] = float(np.sum(power[mask]))
        else:
            nearest = int(np.argmin(np.abs(frequencies - (lower + upper) / 2)))
            bin_power[index] = float(power[nearest])
    log_power = 10 * np.log10(np.maximum(bin_power, 1e-20))
    return (log_power - np.max(log_power)).astype(np.float32)


def _load_clap() -> tuple[Any, Any, torch.device]:
    try:
        from transformers import ClapModel, ClapProcessor
    except ImportError as error:
        raise RuntimeError(
            "CLAP dependencies are missing. Install "
            "`requirements-be3c-baseline.txt` after confirming PyTorch is installed."
        ) from error

    processor = ClapProcessor.from_pretrained(MODEL_ID)
    model = ClapModel.from_pretrained(MODEL_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, processor, device


def _embed_batch(
    model: Any,
    processor: Any,
    device: torch.device,
    waveforms: list[np.ndarray],
) -> np.ndarray:
    inputs = processor(
        audio=waveforms,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        audio_outputs = model.get_audio_features(**model_inputs)
    embeddings = audio_outputs.pooler_output
    if embeddings is None:
        raise RuntimeError("CLAP did not return pooled audio embeddings")
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return embeddings.cpu().numpy().astype(np.float32, copy=False)


def extract_be3c_baseline(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    batch_size: int = 8,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    split_path = output_dir / "be3c_location_split.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_path}")
    split_rows = _read_csv_by_source_row(split_path)
    quality_path = output_dir / "be3c_audio_quality.csv"
    quality_rows = (
        _read_csv_by_source_row(quality_path) if quality_path.is_file() else {}
    )
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet shards found in {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor, device = _load_clap()
    waveforms: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    spectral_rows: list[np.ndarray] = []
    embedding_rows: list[np.ndarray] = []
    processed_positions: set[tuple[str, int]] = set()

    def flush_batch() -> None:
        if not waveforms:
            return
        batch_embeddings = _embed_batch(model, processor, device, waveforms)
        if len(batch_embeddings) != len(metadata_rows) - len(embedding_rows):
            raise RuntimeError("CLAP output count did not match the input batch")
        embedding_rows.extend(batch_embeddings)
        waveforms.clear()

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        required_columns = {"species_code", "record_number", "audio"}
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
                and not (statistics.min <= SPECIES_CODE <= statistics.max)
            ):
                row_offset += row_group.num_rows
                continue
            row_group_offset = 0
            for batch in parquet_file.iter_batches(
                batch_size=64,
                row_groups=[row_group_index],
                columns=["species_code", "record_number", "audio"],
            ):
                for row_index, record in enumerate(batch.to_pylist()):
                    source_row_index = row_offset + row_group_offset + row_index
                    source_key = (parquet_path.name, source_row_index)
                    if record.get("species_code") != SPECIES_CODE:
                        continue
                    split_record = split_rows.get(source_key)
                    if split_record is None:
                        raise ValueError(
                            f"BE3C source row is missing from the split manifest: "
                            f"{source_key}"
                        )
                    payload = ((record.get("audio") or {}).get("bytes"))
                    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
                        raise ValueError(f"Missing audio payload at {source_key}")
                    with sf.SoundFile(BytesIO(payload)) as audio_file:
                        source_rate = audio_file.samplerate
                        audio = audio_file.read(
                            dtype="float32", always_2d=True
                        ).mean(axis=1)
                    audio = _resample_audio(audio, source_rate)
                    record_number = str(record.get("record_number") or "")
                    source_quality = quality_rows.get(source_key, {})
                    for chunk_index, start, end, chunk in _audio_chunks(audio):
                        waveforms.append(chunk)
                        spectral_rows.append(_full_band_features(chunk))
                        metadata_rows.append(
                            {
                                "species_code": SPECIES_CODE,
                                "record_number": record_number,
                                "split": split_record["split"],
                                "location_group": split_record["location_group"],
                                "parquet_file": parquet_path.name,
                                "parquet_row_index": source_row_index,
                                "source_sample_rate_hz": source_rate,
                                "clap_sample_rate_hz": TARGET_SAMPLE_RATE,
                                "chunk_index": chunk_index,
                                "start_seconds": start,
                                "end_seconds": end,
                                "chunk_seconds": end - start,
                                "spectral_power_fraction_above_14khz": (
                                    source_quality.get(
                                        "spectral_power_fraction_above_14khz", ""
                                    )
                                ),
                            }
                        )
                    processed_positions.add(source_key)
                    if len(waveforms) >= batch_size:
                        flush_batch()
                row_group_offset += batch.num_rows
            row_offset += row_group.num_rows
        print(f"Scanned {parquet_path.name}", flush=True)
    flush_batch()

    missing_positions = set(split_rows) - processed_positions
    if missing_positions:
        raise ValueError(
            f"{len(missing_positions)} split rows were not embedded; "
            f"first: {min(missing_positions)}"
        )
    if not metadata_rows:
        raise ValueError(f"No records found for species_code={SPECIES_CODE}")
    if len(embedding_rows) != len(metadata_rows):
        raise RuntimeError(
            f"Embedding count {len(embedding_rows)} did not match chunk count "
            f"{len(metadata_rows)}"
        )

    metadata_path = output_dir / "be3c_embedding_chunks.csv"
    with metadata_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metadata_rows[0]))
        writer.writeheader()
        writer.writerows(metadata_rows)

    embeddings_path = output_dir / "be3c_embedding_features.npz"
    np.savez_compressed(
        embeddings_path,
        clap_embeddings=np.stack(embedding_rows),
        full_band_features=np.stack(spectral_rows),
        model_id=np.asarray(MODEL_ID),
        clap_sample_rate=np.asarray(TARGET_SAMPLE_RATE),
        full_band_frequency_range_hz=np.asarray([50, TARGET_SAMPLE_RATE / 2]),
    )

    split_counts = Counter(row["split"] for row in metadata_rows)
    sample_rate_counts: Counter[str] = Counter()
    seen_recordings: set[tuple[str, str]] = set()
    for row in metadata_rows:
        source_key = (row["parquet_file"], row["parquet_row_index"])
        if source_key not in seen_recordings:
            sample_rate_counts[row["source_sample_rate_hz"]] += 1
            seen_recordings.add(source_key)
    report_lines = [
        "# BE3C frozen representation baseline",
        "",
        f"- Source recordings embedded: {len(processed_positions):,}.",
        f"- Ten-second-or-shorter non-overlapping chunks: {len(metadata_rows):,}.",
        f"- Chunk train / validation counts: "
        f"{split_counts['train']:,} / {split_counts['validation']:,}.",
        f"- Source recording sample rates: "
        f"{dict(sorted(sample_rate_counts.items()))}.",
        f"- Frozen pretrained encoder: `{MODEL_ID}`; no model parameters were trained.",
        f"- CLAP input sample rate: {TARGET_SAMPLE_RATE:,} Hz; the model processor "
        "uses approximately 50 Hz–14 kHz.",
        f"- A companion {FULL_BAND_FEATURE_COUNT}-bin log-power representation spans "
        f"50 Hz–{TARGET_SAMPLE_RATE / 2 / 1000:g} kHz after resampling.",
        "- Chunks retain the source recording, location-held-out split, and location "
        "group so all chunks from a recording remain in its original split.",
        "- The full-band features retain the 14–24 kHz region that CLAP does not "
        "represent. They are acoustic baseline features, not learned embeddings.",
        "- Resampling cannot restore frequencies above a source recording's "
        "original Nyquist limit.",
        "- Source WAVs are resampled in memory only. Original Parquet files are "
        "unchanged.",
        "",
        "These features are ready for a train-only clustering baseline and separate "
        "validation assignment. Species labels are metadata for later evaluation, "
        "not model targets.",
        "",
    ]
    report_path = output_dir / "be3c_embedding_baseline_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "recording_count": len(processed_positions),
        "chunk_count": len(metadata_rows),
        "split_counts": dict(split_counts),
        "embeddings_path": embeddings_path,
        "metadata_path": metadata_path,
        "report_path": report_path,
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frozen CLAP and full-band features for BE3C."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result = extract_be3c_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(f"Embedded {result['recording_count']:,} recordings.")
    print(f"Extracted {result['chunk_count']:,} chunks on {result['device']}.")
    print(f"Baseline outputs: {result['report_path']}")


if __name__ == "__main__":
    main()
