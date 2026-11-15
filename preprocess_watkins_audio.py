"""Create a resumable log-Mel feature cache for the Watkins audio collection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly, stft


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "watkins_preprocessing_config.json"
METADATA_FIELDS = (
    "segment_id",
    "recording_id",
    "species_code",
    "species_label",
    "display_name",
    "scientific_name",
    "mid",
    "class_id",
    "observation_date",
    "location_names",
    "location_coordinates",
    "sound_metadata_json",
    "signal_metadata_json",
    "metadata_json",
    "parquet_file",
    "parquet_row_index",
    "source_sample_rate_hz",
    "metadata_sample_rate_hz",
    "sample_rate_metadata_mismatch",
    "channel_count",
    "processed_channel_count",
    "recording_duration_seconds",
    "recording_peak_amplitude",
    "recording_rms_amplitude",
    "normalized_recording_peak_amplitude",
    "normalized_recording_rms_amplitude",
    "normalization_method",
    "normalization_gain",
    "segment_index",
    "segment_start_seconds",
    "segment_valid_seconds",
    "segment_padded",
    "feature_partition",
    "feature_index",
)
ERROR_FIELDS = (
    "parquet_file",
    "parquet_row_index",
    "recording_id",
    "species_code",
    "error_type",
    "error_message",
)


@dataclass(frozen=True)
class PreprocessingConfig:
    data_dir: Path
    output_dir: Path
    target_sample_rate_hz: int
    segment_duration_seconds: float
    n_fft: int
    hop_length: int
    n_mels: int
    f_min_hz: float
    f_max_hz: float
    log_power_floor: float
    normalization_method: str
    normalization_target_peak: float
    records_per_partition: int
    feature_dtype: str

    @classmethod
    def load(cls, path: Path) -> "PreprocessingConfig":
        with path.open(encoding="utf-8") as stream:
            values = json.load(stream)
        config = cls(
            data_dir=_resolve_path(values["data_dir"]),
            output_dir=_resolve_path(values["output_dir"]),
            target_sample_rate_hz=int(values["target_sample_rate_hz"]),
            segment_duration_seconds=float(values["segment_duration_seconds"]),
            n_fft=int(values["n_fft"]),
            hop_length=int(values["hop_length"]),
            n_mels=int(values["n_mels"]),
            f_min_hz=float(values["f_min_hz"]),
            f_max_hz=float(values["f_max_hz"]),
            log_power_floor=float(values["log_power_floor"]),
            normalization_method=str(values["normalization_method"]),
            normalization_target_peak=float(
                values["normalization_target_peak"]
            ),
            records_per_partition=int(values["records_per_partition"]),
            feature_dtype=str(values["feature_dtype"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.target_sample_rate_hz <= 0:
            raise ValueError("target_sample_rate_hz must be positive")
        if self.segment_duration_seconds <= 0:
            raise ValueError("segment_duration_seconds must be positive")
        if self.n_fft < 2 or self.hop_length <= 0 or self.hop_length > self.n_fft:
            raise ValueError("Require n_fft >= 2 and 0 < hop_length <= n_fft")
        if self.segment_samples < self.n_fft:
            raise ValueError("The fixed segment must be at least n_fft samples")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")
        if not 0 <= self.f_min_hz < self.f_max_hz:
            raise ValueError("Mel frequency bounds are invalid")
        if self.f_max_hz > self.target_sample_rate_hz / 2:
            raise ValueError("f_max_hz cannot exceed target Nyquist")
        if self.log_power_floor <= 0:
            raise ValueError("log_power_floor must be positive")
        if self.normalization_method != "peak":
            raise ValueError("normalization_method must be 'peak'")
        if not 0 < self.normalization_target_peak <= 1:
            raise ValueError("normalization_target_peak must be in (0, 1]")
        if self.records_per_partition <= 0:
            raise ValueError("records_per_partition must be positive")
        if self.feature_dtype not in ("float16", "float32"):
            raise ValueError("feature_dtype must be float16 or float32")

    @property
    def segment_samples(self) -> int:
        return round(
            self.segment_duration_seconds * self.target_sample_rate_hz
        )

    @property
    def frame_count(self) -> int:
        return 1 + (self.segment_samples - self.n_fft) // self.hop_length

    def as_dict(self) -> dict[str, Any]:
        values = dict(self.__dict__)
        values["data_dir"] = str(self.data_dir)
        values["output_dir"] = str(self.output_dir)
        return values


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def _mel_filterbank(
    sample_rate_hz: int,
    n_fft: int,
    n_mels: int,
    f_min_hz: float,
    f_max_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= f_min_hz < f_max_hz <= sample_rate_hz / 2:
        raise ValueError("Mel filter frequency bounds exceed Nyquist")

    def hz_to_mel(frequency: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + frequency / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(
        hz_to_mel(np.asarray([f_min_hz]))[0],
        hz_to_mel(np.asarray([f_max_hz]))[0],
        n_mels + 2,
    )
    hz_points = mel_to_hz(mel_points)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
    filters = np.zeros((n_mels, len(frequencies)), dtype=np.float32)
    for index in range(n_mels):
        lower, center, upper = hz_points[index : index + 3]
        rising = (frequencies - lower) / (center - lower)
        falling = (upper - frequencies) / (upper - center)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))
    return filters, hz_points[1:-1]


def _resample_channels(
    audio: np.ndarray,
    source_sample_rate_hz: int,
    target_sample_rate_hz: int,
) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    if samples.ndim != 2 or samples.shape[1] == 0:
        raise ValueError("Audio must have shape (frames, channels)")
    if samples.shape[0] == 0 or not np.isfinite(samples).all():
        raise ValueError("Audio is empty or contains non-finite samples")
    if source_sample_rate_hz == target_sample_rate_hz:
        return samples
    ratio = Fraction(target_sample_rate_hz, source_sample_rate_hz)
    return resample_poly(
        samples, ratio.numerator, ratio.denominator, axis=0
    ).astype(np.float32, copy=False)


def _downmix_to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim != 2 or samples.shape[1] == 0:
        raise ValueError("Audio must have shape (frames, channels)")
    if samples.shape[1] == 1:
        return samples[:, 0].copy()
    return samples.mean(axis=1, dtype=np.float32)


def _peak_normalize(
    waveform: np.ndarray,
    target_peak: float,
) -> tuple[np.ndarray, float]:
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("waveform must be a non-empty mono signal")
    if not 0 < target_peak <= 1:
        raise ValueError("target_peak must be in (0, 1]")
    if not np.isfinite(waveform).all():
        raise ValueError("waveform contains non-finite samples")
    peak = float(np.max(np.abs(waveform)))
    gain = target_peak / peak if peak > 0 else 1.0
    normalized = (waveform * gain).astype(np.float32, copy=False)
    if not np.isfinite(normalized).all():
        raise ValueError("Peak normalization produced non-finite samples")
    return normalized, gain


def _segment_audio(
    samples: np.ndarray,
    segment_samples: int,
) -> Iterator[tuple[int, int, np.ndarray]]:
    if segment_samples <= 0:
        raise ValueError("segment_samples must be positive")
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("samples must be a non-empty mono signal")
    for segment_index, start in enumerate(range(0, len(samples), segment_samples)):
        valid = samples[start : start + segment_samples]
        padded = np.zeros(segment_samples, dtype=np.float32)
        padded[: len(valid)] = valid
        yield segment_index, len(valid), padded


def _log_mel_spectrogram(
    segment: np.ndarray,
    config: PreprocessingConfig,
    filters: np.ndarray,
) -> np.ndarray:
    if (
        segment.ndim != 1
        or segment.shape[0] != config.segment_samples
    ):
        raise ValueError("Segment length does not match configured window")
    _, _, spectrum = stft(
        segment,
        fs=config.target_sample_rate_hz,
        window="hann",
        nperseg=config.n_fft,
        noverlap=config.n_fft - config.hop_length,
        nfft=config.n_fft,
        boundary=None,
        padded=False,
    )
    power = np.square(np.abs(spectrum), dtype=np.float32)
    mel_power = filters @ power
    log_power = 10.0 * np.log10(
        np.maximum(mel_power, config.log_power_floor)
    )
    if log_power.shape != (config.n_mels, config.frame_count):
        raise RuntimeError(
            f"Unexpected log-Mel shape {log_power.shape}; expected "
            f"{(config.n_mels, config.frame_count)}"
        )
    if not np.isfinite(log_power).all():
        raise ValueError("Log-Mel features contain non-finite values")
    return log_power.astype(np.dtype(config.feature_dtype), copy=False)


def _species_label(record: dict[str, Any]) -> str:
    species_code = record.get("species_code")
    if isinstance(species_code, str) and species_code.strip():
        return species_code.strip()
    scientific_name = record.get("scientific_name")
    if isinstance(scientific_name, str) and scientific_name.strip():
        return scientific_name.strip()
    display_name = record.get("display_name")
    if isinstance(display_name, str) and display_name.strip():
        return f"[UNLABELED] {display_name.strip()}"
    return "[UNLABELED] missing species label"


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_csv_atomic(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _cache_signature(
    data_path: Path,
    config: PreprocessingConfig,
    config_digest: str,
) -> dict[str, Any]:
    stat = data_path.stat()
    return {
        "config_digest": config_digest,
        "source_file": data_path.name,
        "source_file_size": stat.st_size,
        "source_file_mtime_ns": stat.st_mtime_ns,
    }


def _record_metadata(
    record: dict[str, Any],
    parquet_name: str,
    row_index: int,
    source_rate: int,
    channels: int,
    duration: float,
    peak: float,
    rms: float,
    normalized_peak: float,
    normalized_rms: float,
    normalization_gain: float,
    config: PreprocessingConfig,
) -> dict[str, Any]:
    sound = record.get("sound") or {}
    location = record.get("location") or {}
    signal = record.get("signal") or {}
    sample_rate_metadata = sound.get("sample_rate")
    return {
        "recording_id": str(record.get("record_number") or ""),
        "species_code": str(record.get("species_code") or ""),
        "species_label": _species_label(record),
        "display_name": record.get("display_name") or "",
        "scientific_name": record.get("scientific_name") or "",
        "mid": record.get("mid") or "",
        "class_id": record.get("class_id") or "",
        "observation_date": record.get("observation_date") or "",
        "location_names": " | ".join(location.get("name") or []),
        "location_coordinates": _json_value(location.get("coordinates") or []),
        "sound_metadata_json": _json_value(sound),
        "signal_metadata_json": _json_value(signal),
        "metadata_json": _json_value(
            {
                key: value
                for key, value in record.items()
                if key != "audio"
            }
        ),
        "parquet_file": parquet_name,
        "parquet_row_index": row_index,
        "source_sample_rate_hz": source_rate,
        "metadata_sample_rate_hz": (
            "" if sample_rate_metadata is None else sample_rate_metadata
        ),
        "sample_rate_metadata_mismatch": (
            sample_rate_metadata is not None
            and int(sample_rate_metadata) != source_rate
        ),
        "channel_count": channels,
        "processed_channel_count": 1,
        "recording_duration_seconds": duration,
        "recording_peak_amplitude": peak,
        "recording_rms_amplitude": rms,
        "normalized_recording_peak_amplitude": normalized_peak,
        "normalized_recording_rms_amplitude": normalized_rms,
        "normalization_method": config.normalization_method,
        "normalization_gain": normalization_gain,
    }


def _process_record(
    record: dict[str, Any],
    parquet_name: str,
    row_index: int,
    partition_name: str,
    feature_offset: int,
    config: PreprocessingConfig,
    filters: np.ndarray,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    audio_struct = record.get("audio") or {}
    payload = audio_struct.get("bytes")
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
        raise ValueError("Missing or empty embedded audio payload")

    with sf.SoundFile(BytesIO(payload)) as audio_file:
        source_rate = int(audio_file.samplerate)
        channels = int(audio_file.channels)
        source_frames = int(audio_file.frames)
        samples = audio_file.read(dtype="float32", always_2d=True)
    if source_rate <= 0 or source_frames <= 0:
        raise ValueError("Audio header has invalid sample rate or frame count")
    if not np.isfinite(samples).all():
        raise ValueError("Decoded audio contains non-finite samples")

    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    resampled_channels = _resample_channels(
        samples, source_rate, config.target_sample_rate_hz
    )
    mono = _downmix_to_mono(resampled_channels)
    waveform, normalization_gain = _peak_normalize(
        mono, config.normalization_target_peak
    )
    normalized_peak = float(np.max(np.abs(waveform)))
    normalized_rms = float(
        np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))
    )
    base = _record_metadata(
        record,
        parquet_name,
        row_index,
        source_rate,
        channels,
        source_frames / source_rate,
        peak,
        rms,
        normalized_peak,
        normalized_rms,
        normalization_gain,
        config,
    )
    features: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for segment_index, valid_samples, segment in _segment_audio(
        waveform, config.segment_samples
    ):
        feature_index = feature_offset + len(features)
        start = segment_index * config.segment_duration_seconds
        features.append(_log_mel_spectrogram(segment, config, filters))
        rows.append(
            {
                **base,
                "segment_id": (
                    f"{parquet_name}:{row_index}:{segment_index}"
                ),
                "segment_index": segment_index,
                "segment_start_seconds": start,
                "segment_valid_seconds": (
                    valid_samples / config.target_sample_rate_hz
                ),
                "segment_padded": valid_samples < config.segment_samples,
                "feature_partition": partition_name,
                "feature_index": feature_index,
            }
        )
    return features, rows


def _partition_paths(
    output_dir: Path,
    parquet_name: str,
    row_group: int,
    batch_number: int,
) -> tuple[Path, Path, Path, Path]:
    stem = (
        f"{Path(parquet_name).stem}-rg{row_group:03d}-"
        f"batch{batch_number:06d}"
    )
    return (
        output_dir / "features" / f"{stem}.npz",
        output_dir / "metadata_parts" / f"{stem}.csv",
        output_dir / "error_parts" / f"{stem}.csv",
        output_dir / "cache_state" / f"{stem}.json",
    )


def _partition_is_cached(
    paths: tuple[Path, Path, Path, Path],
    expected_signature: dict[str, Any],
) -> bool:
    features_path, metadata_path, errors_path, marker_path = paths
    if not all(path.is_file() for path in paths):
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(
        marker.get(key) == value
        for key, value in expected_signature.items()
    )


def _cached_shard_parts(
    parquet_path: Path,
    output_dir: Path,
    config: PreprocessingConfig,
    config_digest: str,
) -> tuple[bool, list[Path], list[Path], int]:
    parquet = pq.ParquetFile(parquet_path)
    signature = _cache_signature(parquet_path, config, config_digest)
    row_offset = 0
    metadata_paths: list[Path] = []
    error_paths: list[Path] = []
    partition_count = 0
    for row_group_index in range(parquet.metadata.num_row_groups):
        row_count = parquet.metadata.row_group(row_group_index).num_rows
        batch_count = (
            row_count + config.records_per_partition - 1
        ) // config.records_per_partition
        for batch_number in range(batch_count):
            batch_start = batch_number * config.records_per_partition
            batch_rows = min(
                config.records_per_partition, row_count - batch_start
            )
            expected_signature = {
                **signature,
                "row_start": row_offset + batch_start,
                "row_count": batch_rows,
            }
            paths = _partition_paths(
                output_dir,
                parquet_path.name,
                row_group_index,
                batch_number,
            )
            if not _partition_is_cached(paths, expected_signature):
                return False, [], [], partition_count
            metadata_paths.append(paths[1])
            error_paths.append(paths[2])
            partition_count += 1
        row_offset += row_count
    return True, metadata_paths, error_paths, partition_count


def _write_partition(
    paths: tuple[Path, Path, Path, Path],
    features: list[np.ndarray],
    metadata_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    signature: dict[str, Any],
    config: PreprocessingConfig,
) -> None:
    features_path, metadata_path, errors_path, marker_path = paths
    if features:
        feature_array = np.stack(features).astype(
            np.dtype(config.feature_dtype), copy=False
        )
    else:
        feature_array = np.empty(
            (0, config.n_mels, config.frame_count),
            dtype=np.dtype(config.feature_dtype),
        )
    feature_temp = features_path.with_suffix(".npz.tmp")
    with feature_temp.open("wb") as stream:
        np.savez_compressed(
            stream,
            log_mel=feature_array,
            config_digest=np.asarray(signature["config_digest"]),
        )
    os.replace(feature_temp, features_path)
    _write_csv_atomic(metadata_path, METADATA_FIELDS, metadata_rows)
    _write_csv_atomic(errors_path, ERROR_FIELDS, errors)
    _atomic_json(
        marker_path,
        {
            **signature,
            "segment_count": len(metadata_rows),
            "error_count": len(errors),
        },
    )


def _partitions(
    data_dir: Path,
    parquet_path: Path,
    config: PreprocessingConfig,
    config_digest: str,
) -> Iterator[tuple[int, int, list[dict[str, Any]], dict[str, Any]]]:
    parquet = pq.ParquetFile(parquet_path)
    available = set(parquet.schema_arrow.names)
    required = {"record_number", "species_code", "sound", "audio"}
    if not required.issubset(available):
        raise ValueError(
            f"{parquet_path.name} lacks required columns "
            f"{sorted(required - available)}"
        )
    optional = (
        "display_name",
        "scientific_name",
        "mid",
        "class_id",
        "observation_date",
        "location",
        "signal",
    )
    columns = [
        name for name in ("record_number", "species_code", *optional, "sound", "audio")
        if name in available
    ]
    signature = _cache_signature(parquet_path, config, config_digest)
    row_offset = 0
    for row_group_index in range(parquet.metadata.num_row_groups):
        row_group_rows = parquet.metadata.row_group(row_group_index).num_rows
        batch_number = 0
        group_offset = 0
        for batch in parquet.iter_batches(
            batch_size=config.records_per_partition,
            row_groups=[row_group_index],
            columns=columns,
        ):
            start_index = row_offset + group_offset
            yield (
                row_group_index,
                batch_number,
                batch.to_pylist(),
                {
                    **signature,
                    "row_start": start_index,
                    "row_count": batch.num_rows,
                },
            )
            group_offset += batch.num_rows
            batch_number += 1
        row_offset += row_group_rows


def _write_consolidated_csv(
    output_path: Path,
    fieldnames: tuple[str, ...],
    part_paths: list[Path],
) -> int:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    count = 0
    with temporary.open("w", newline="", encoding="utf-8-sig") as output_stream:
        writer = csv.DictWriter(output_stream, fieldnames=fieldnames)
        writer.writeheader()
        for part_path in part_paths:
            with part_path.open(encoding="utf-8-sig", newline="") as input_stream:
                for row in csv.DictReader(input_stream):
                    writer.writerow(row)
                    count += 1
    os.replace(temporary, output_path)
    return count


def _make_spectrogram_previews(
    manifest_path: Path,
    output_dir: Path,
    config: PreprocessingConfig,
) -> list[dict[str, Any]]:
    selections: dict[str, dict[str, str]] = {}
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            duration = float(row["recording_duration_seconds"])
            category = (
                "short"
                if duration < config.segment_duration_seconds
                else "long"
                if duration >= 30.0
                else "typical"
            )
            if category in selections:
                continue
            selections[category] = row
            if len(selections) == 3:
                break

    preview_dir = output_dir / "spectrogram_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_rows = []
    for category, row in selections.items():
        feature_path = output_dir / "features" / row["feature_partition"]
        with np.load(feature_path) as stored:
            feature = stored["log_mel"][int(row["feature_index"])].astype(
                np.float32
            )
        preview_path = preview_dir / f"{category}_segment.png"
        fig, axis = plt.subplots(figsize=(11, 4))
        image = axis.imshow(
            feature,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(
                0,
                config.segment_duration_seconds,
                0,
                config.n_mels,
            ),
            cmap="magma",
        )
        axis.set_title(
            f"{category.title()} example: {row['recording_id']} "
            f"(species {row['species_label']}, "
            f"source {row['source_sample_rate_hz']} Hz)"
        )
        axis.set_xlabel("Time within fixed segment (seconds)")
        axis.set_ylabel("Mel filter index")
        fig.colorbar(image, ax=axis, label="Log power (dB re 1.0)")
        fig.tight_layout()
        fig.savefig(preview_path, dpi=140)
        plt.close(fig)
        preview_rows.append(
            {
                "example_type": category,
                "recording_id": row["recording_id"],
                "species_label": row["species_label"],
                "source_sample_rate_hz": row["source_sample_rate_hz"],
                "recording_duration_seconds": row[
                    "recording_duration_seconds"
                ],
                "segment_id": row["segment_id"],
                "spectrogram_path": str(preview_path),
            }
        )
    return preview_rows


def run_preprocessing(config: PreprocessingConfig) -> dict[str, Any]:
    config.validate()
    if not config.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {config.data_dir}")
    parquet_files = sorted(config.data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet shards found in {config.data_dir}")

    output_dir = config.output_dir
    for subdirectory in (
        "features",
        "metadata_parts",
        "error_parts",
        "cache_state",
    ):
        (output_dir / subdirectory).mkdir(parents=True, exist_ok=True)
    config_payload = config.as_dict()
    config_digest = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    _atomic_json(
        output_dir / "preprocessing_config_used.json",
        {**config_payload, "config_digest": config_digest},
    )
    filterbank, mel_centers = _mel_filterbank(
        config.target_sample_rate_hz,
        config.n_fft,
        config.n_mels,
        config.f_min_hz,
        config.f_max_hz,
    )

    partition_count = 0
    skipped_partitions = 0
    total_records = 0
    all_metadata_parts: list[Path] = []
    all_error_parts: list[Path] = []
    for parquet_path in parquet_files:
        shard_cached, metadata_parts, error_parts, cached_count = (
            _cached_shard_parts(
                parquet_path,
                output_dir,
                config,
                config_digest,
            )
        )
        if shard_cached:
            all_metadata_parts.extend(metadata_parts)
            all_error_parts.extend(error_parts)
            skipped_partitions += cached_count
            total_records += pq.ParquetFile(parquet_path).metadata.num_rows
            print(f"Reused cached shard: {parquet_path.name}", flush=True)
            continue
        for row_group_index, batch_number, records, signature in _partitions(
            config.data_dir, parquet_path, config, config_digest
        ):
            paths = _partition_paths(
                output_dir,
                parquet_path.name,
                row_group_index,
                batch_number,
            )
            all_metadata_parts.append(paths[1])
            all_error_parts.append(paths[2])
            if _partition_is_cached(paths, signature):
                skipped_partitions += 1
                total_records += len(records)
                continue

            partition_name = paths[0].name
            features: list[np.ndarray] = []
            metadata_rows: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            for record_offset, record in enumerate(records):
                row_index = int(signature["row_start"]) + record_offset
                try:
                    record_features, record_rows = _process_record(
                        record,
                        parquet_path.name,
                        row_index,
                        partition_name,
                        len(features),
                        config,
                        filterbank,
                    )
                    features.extend(record_features)
                    metadata_rows.extend(record_rows)
                except Exception as error:
                    errors.append(
                        {
                            "parquet_file": parquet_path.name,
                            "parquet_row_index": row_index,
                            "recording_id": str(
                                record.get("record_number") or ""
                            ),
                            "species_code": str(
                                record.get("species_code") or ""
                            ),
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }
                    )
            _write_partition(
                paths,
                features,
                metadata_rows,
                errors,
                signature,
                config,
            )
            partition_count += 1
            total_records += len(records)
            if partition_count % 20 == 0:
                print(
                    f"Processed {total_records:,} recordings; "
                    f"new partitions={partition_count}, "
                    f"cached partitions={skipped_partitions}",
                    flush=True,
                )
        print(f"Finished preprocessing shard: {parquet_path.name}", flush=True)

    segment_manifest = output_dir / "segment_metadata.csv"
    error_manifest = output_dir / "processing_errors.csv"
    segment_count = _write_consolidated_csv(
        segment_manifest, METADATA_FIELDS, all_metadata_parts
    )
    error_count = _write_consolidated_csv(
        error_manifest, ERROR_FIELDS, all_error_parts
    )
    preview_rows = _make_spectrogram_previews(
        segment_manifest, output_dir, config
    )
    _write_csv_atomic(
        output_dir / "spectrogram_preview_manifest.csv",
        (
            "example_type",
            "recording_id",
            "species_label",
            "source_sample_rate_hz",
            "recording_duration_seconds",
            "segment_id",
            "spectrogram_path",
        ),
        preview_rows,
    )

    duration_by_record: dict[tuple[str, str], float] = {}
    mismatch_count = 0
    padded_count = 0
    source_rates: Counter[int] = Counter()
    with segment_manifest.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["parquet_file"], row["parquet_row_index"])
            duration_by_record[key] = float(row["recording_duration_seconds"])
            source_rates[int(row["source_sample_rate_hz"])] += (
                1 if row["segment_index"] == "0" else 0
            )
            mismatch_count += (
                row["sample_rate_metadata_mismatch"].casefold() == "true"
                and row["segment_index"] == "0"
            )
            padded_count += row["segment_padded"].casefold() == "true"

    short_count = sum(
        duration < config.segment_duration_seconds
        for duration in duration_by_record.values()
    )
    long_count = sum(
        duration > config.segment_duration_seconds
        for duration in duration_by_record.values()
    )
    report_rows = [
        {
            "metric": "parquet_shards",
            "value": len(parquet_files),
            "details": str(config.data_dir),
        },
        {
            "metric": "recordings_with_processed_segments",
            "value": len(duration_by_record),
            "details": "Unique source recordings represented in segment manifest",
        },
        {
            "metric": "segment_count",
            "value": segment_count,
            "details": f"Fixed window {config.segment_duration_seconds:g} seconds",
        },
        {
            "metric": "recordings_shorter_than_window",
            "value": short_count,
            "details": "Short recordings are zero-padded, not discarded",
        },
        {
            "metric": "recordings_longer_than_window",
            "value": long_count,
            "details": "Long recordings are split into consecutive non-overlapping windows",
        },
        {
            "metric": "padded_segments",
            "value": padded_count,
            "details": "Only the final partial window or short recording is padded",
        },
        {
            "metric": "processing_failures",
            "value": error_count,
            "details": str(error_manifest),
        },
        {
            "metric": "source_sample_rate_metadata_mismatches",
            "value": mismatch_count,
            "details": "Actual decoded file header rate was used",
        },
        {
            "metric": "target_sample_rate_hz",
            "value": config.target_sample_rate_hz,
            "details": "96 kHz Nyquist preserves frequencies through 96 kHz",
        },
        {
            "metric": "segment_samples",
            "value": config.segment_samples,
            "details": "Fixed-length analysis window",
        },
        {
            "metric": "n_fft",
            "value": config.n_fft,
            "details": f"Frequency-bin spacing {config.target_sample_rate_hz / config.n_fft:.2f} Hz",
        },
        {
            "metric": "hop_length",
            "value": config.hop_length,
            "details": f"Frame hop {1000 * config.hop_length / config.target_sample_rate_hz:.2f} ms",
        },
        {
            "metric": "n_mels",
            "value": config.n_mels,
            "details": f"Mel range {config.f_min_hz:g}-{config.f_max_hz:g} Hz",
        },
        {
            "metric": "feature_shape",
            "value": f"{config.n_mels}x{config.frame_count}",
            "details": f"Stored as {config.feature_dtype}",
        },
        {
            "metric": "actual_source_sample_rates_hz",
            "value": len(source_rates),
            "details": _json_value(dict(sorted(source_rates.items()))),
        },
        {
            "metric": "cache_partitions_written_this_run",
            "value": partition_count,
            "details": "Unchanged partitions were skipped using source stat and configuration fingerprints",
        },
        {
            "metric": "cache_partitions_reused",
            "value": skipped_partitions,
            "details": "Partition feature, metadata, error log and cache marker all matched",
        },
    ]
    report_csv = output_dir / "preprocessing_validation.csv"
    _write_csv_atomic(
        report_csv, ("metric", "value", "details"), report_rows
    )
    report_md = output_dir / "preprocessing_validation_report.md"
    report_md.write_text(
        "\n".join(
            [
                "# Watkins Stage 2 preprocessing validation",
                "",
                f"- Source recordings with features: {len(duration_by_record):,}.",
                f"- Segment windows: {segment_count:,}; fixed duration {config.segment_duration_seconds:g} s.",
                f"- Short recordings kept and zero-padded: {short_count:,}.",
                f"- Final partial windows zero-padded: {padded_count:,}.",
                f"- Processing failures: {error_count:,}; see `processing_errors.csv`.",
                f"- Header/metadata sample-rate mismatches found during this pass: {mismatch_count:,}; decoded audio header was authoritative.",
                f"- Feature shape: {config.n_mels} Mel bins × {config.frame_count} frames, {config.feature_dtype}.",
                f"- Cached partitions written: {partition_count:,}; reused without recomputation: {skipped_partitions:,}.",
                "",
                "## Parameter rationale",
                "",
                f"- Target sample rate: {config.target_sample_rate_hz:,} Hz. The existing metadata inventory contains audio through 192 kHz; a 192 kHz target has a 96 kHz Nyquist and avoids the irreversible 24 kHz ceiling imposed by the prior 48 kHz baseline. Resampling does not invent frequencies absent from a source recording.",
                f"- Segment duration: {config.segment_duration_seconds:g} seconds, configurable in `watkins_preprocessing_config.json`; shorter recordings and partial final windows are retained with zero padding and valid-duration metadata.",
                f"- STFT: n_fft={config.n_fft} (~{1000 * config.n_fft / config.target_sample_rate_hz:.1f} ms), hop={config.hop_length} (~{1000 * config.hop_length / config.target_sample_rate_hz:.1f} ms), {config.n_mels} Mel bins from {config.f_min_hz:g} Hz to {config.f_max_hz:g} Hz.",
                "- Resampled channels are arithmetically averaged into one mono waveform before segmentation. The source channel count and processed mono channel count are retained in the manifest. The audited full-dataset cache contains mono source recordings; the downmix path supports future multichannel records.",
                f"- Each mono recording is peak-normalized once to {config.normalization_target_peak:g} before segmentation. The original peak/RMS, normalized RMS, and applied gain are preserved in segment metadata. This equalizes each recording's maximum level but intentionally removes between-recording absolute peak-level differences; use the stored raw levels if amplitude/domain analyses are needed.",
                "- No high-pass filter, denoising, silence removal, loudness gate, or class rebalancing is applied. Low-amplitude and rare-species recordings are not discarded.",
                "",
                "## Outputs",
                "",
                "- `features/`: compressed float16 log-Mel arrays partitioned by source shard, row group, and record batch.",
                "- `segment_metadata.csv`: one row per segment with source recording ID, label, recording metadata, timing, actual sample rate, and feature partition/index.",
                "- `processing_errors.csv`: per-record failures; failed recordings do not stop the rest of the shard.",
                "- `spectrogram_previews/`: representative short, typical, and long-recording examples where available.",
                "- `preprocessing_validation.csv`: machine-readable counts and processing configuration.",
                "- `cache_state/`: source-file/config fingerprints enabling safe incremental reruns.",
                "",
                "Stage 3 must split by the original recording/group before using segments; segment rows from one recording must never cross split boundaries.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Segments: {segment_count:,}; recordings: {len(duration_by_record):,}")
    print(f"Processing failures: {error_count:,}")
    print(f"Validation report: {report_md}")
    print(f"Segment manifest: {segment_manifest}")
    return {
        "record_count": len(duration_by_record),
        "segment_count": segment_count,
        "failure_count": error_count,
        "manifest_path": segment_manifest,
        "error_path": error_manifest,
        "report_path": report_md,
        "preview_rows": preview_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a resumable Watkins log-Mel segment cache."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = PreprocessingConfig.load(args.config)
    if args.data_dir is not None:
        config = PreprocessingConfig(**{**config.__dict__, "data_dir": args.data_dir})
    if args.output_dir is not None:
        config = PreprocessingConfig(
            **{**config.__dict__, "output_dir": args.output_dir}
        )
    run_preprocessing(config)


if __name__ == "__main__":
    main()
