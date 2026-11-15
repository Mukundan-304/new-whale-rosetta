"""Extract resumable frozen CLAP embeddings for the Watkins split manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import psutil
import soundfile as sf
import torch
from scipy.signal import resample_poly

from preprocess_watkins_audio import (
    PreprocessingConfig,
    _downmix_to_mono,
    _peak_normalize,
    _resample_channels,
    _segment_audio,
)


PROJECT_DIR = Path(__file__).resolve().parent
SPLIT_DIR = PROJECT_DIR / "watkins_splits"
PREPROCESSING_CONFIG_PATH = PROJECT_DIR / "watkins_preprocessing_config.json"
STAGE4_CONFIG_PATH = PROJECT_DIR / "watkins_stage4_config.json"
OUTPUT_DIR = PROJECT_DIR / "watkins_stage4"
SPLITS = ("train", "validation", "test")
MODEL_SAMPLE_RATE_HZ = 48_000
LOGGER = logging.getLogger("watkins_stage4")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_manifest(path: Path, expected_split: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required split manifest does not exist: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "segment_id",
            "recording_id",
            "species_label",
            "audio_path",
            "audio_parquet_row_index",
            "parquet_row_index",
            "segment_index",
            "source_sample_rate_hz",
            "channel_count",
            "feature_path",
            "feature_index",
            "split",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} lacks required columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Split manifest is empty: {path}")

    identifiers: set[str] = set()
    segment_keys: set[tuple[str, str]] = set()
    for row_number, row in enumerate(rows, start=2):
        segment_id = row["segment_id"].strip()
        recording_id = row["recording_id"].strip()
        if not segment_id or not recording_id:
            raise ValueError(f"{path}:{row_number} has a missing ID")
        if segment_id in identifiers:
            raise ValueError(f"{path} contains duplicate segment_id {segment_id!r}")
        identifiers.add(segment_id)
        if row["split"].strip() != expected_split:
            raise ValueError(
                f"{path}:{row_number} has split {row['split']!r}, "
                f"expected {expected_split!r}"
            )
        audio_path = Path(row["audio_path"])
        feature_path = Path(row["feature_path"])
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Invalid audio_path for segment {segment_id}: {audio_path}"
            )
        if not feature_path.is_file():
            raise FileNotFoundError(
                f"Invalid feature_path for segment {segment_id}: {feature_path}"
            )
        audio_row = int(row["audio_parquet_row_index"])
        parquet_row = int(row["parquet_row_index"])
        segment_index = int(row["segment_index"])
        int(row["feature_index"])
        if audio_row != parquet_row:
            raise ValueError(
                f"Audio and manifest row indexes disagree for {segment_id}: "
                f"{audio_row} != {parquet_row}"
            )
        key = (str(audio_path.resolve()), str(audio_row))
        segment_key = (key[0], f"{key[1]}:{segment_index}")
        if segment_key in segment_keys:
            raise ValueError(f"Duplicate source segment in {path}: {segment_id}")
        segment_keys.add(segment_key)
    return rows


def _manifest_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ResourceMonitor:
    def __init__(self, device: torch.device, interval_seconds: float = 30.0) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.process.cpu_percent(interval=None)
        self.last_sample = 0.0
        self.cpu_percent: list[float] = []
        self.rss_bytes: list[int] = []
        self.available_ram_bytes: list[int] = []
        self.gpu_utilization_percent: list[int] = []
        self.gpu_memory_used_bytes: list[int] = []

    def sample(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_sample < self.interval_seconds:
            return
        self.last_sample = now
        self.cpu_percent.append(float(self.process.cpu_percent(interval=None)))
        self.rss_bytes.append(int(self.process.memory_info().rss))
        self.available_ram_bytes.append(int(psutil.virtual_memory().available))
        if self.device.type != "cuda":
            return
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as error:
            LOGGER.warning("Could not sample nvidia-smi utilization: %s", error)
            return
        first_line = result.stdout.strip().splitlines()
        if not first_line:
            LOGGER.warning("nvidia-smi returned no GPU utilization data")
            return
        try:
            utilization, memory_mib = (
                int(value.strip()) for value in first_line[0].split(",", maxsplit=1)
            )
        except (ValueError, IndexError) as error:
            LOGGER.warning("Could not parse nvidia-smi output: %s", error)
            return
        self.gpu_utilization_percent.append(utilization)
        self.gpu_memory_used_bytes.append(memory_mib * 1024 * 1024)

    def summary(self) -> dict[str, Any]:
        return {
            "process_cpu_percent_mean": (
                round(float(np.mean(self.cpu_percent)), 2)
                if self.cpu_percent
                else None
            ),
            "process_cpu_percent_max": (
                round(float(np.max(self.cpu_percent)), 2)
                if self.cpu_percent
                else None
            ),
            "process_rss_peak_bytes": max(self.rss_bytes, default=0),
            "system_available_ram_min_bytes": min(
                self.available_ram_bytes, default=0
            ),
            "gpu_utilization_percent_mean": (
                round(float(np.mean(self.gpu_utilization_percent)), 2)
                if self.gpu_utilization_percent
                else None
            ),
            "gpu_utilization_percent_max": (
                max(self.gpu_utilization_percent, default=None)
            ),
            "gpu_memory_used_peak_bytes": max(
                self.gpu_memory_used_bytes, default=0
            ),
            "gpu_utilization_samples": len(self.gpu_utilization_percent),
        }


def _load_encoder(
    model_id: str, device_name: str
) -> tuple[Any, Any, torch.device]:
    try:
        from transformers import ClapModel, ClapProcessor
    except ImportError as error:
        raise RuntimeError(
            "The CLAP encoder requires transformers with ClapModel and ClapProcessor."
        ) from error

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but this PyTorch installation cannot access CUDA."
            )
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    extractor = processor.feature_extractor
    if int(extractor.sampling_rate) != MODEL_SAMPLE_RATE_HZ:
        raise ValueError(
            f"Model expects {extractor.sampling_rate} Hz, "
            f"but the configured pipeline uses {MODEL_SAMPLE_RATE_HZ} Hz."
        )
    return model, processor, device


def _embed_batch(
    model: Any,
    processor: Any,
    device: torch.device,
    waveforms: list[np.ndarray],
    mixed_precision: bool,
) -> np.ndarray:
    inputs = processor(
        audio=waveforms,
        sampling_rate=MODEL_SAMPLE_RATE_HZ,
        return_tensors="pt",
    )
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        if device.type == "cuda" and mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.get_audio_features(**model_inputs)
        else:
            output = model.get_audio_features(**model_inputs)
    embeddings = output.pooler_output
    if embeddings is None:
        raise RuntimeError("CLAP did not return pooled audio embeddings")
    embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=-1)
    vectors = embeddings.cpu().numpy().astype(np.float32, copy=False)
    if vectors.ndim != 2 or vectors.shape[0] != len(waveforms):
        raise RuntimeError(
            f"Unexpected CLAP output shape {vectors.shape} for "
            f"{len(waveforms)} input segments"
        )
    if not np.isfinite(vectors).all():
        raise RuntimeError("CLAP produced a NaN or infinite embedding")
    return vectors


def _iter_waveforms(
    rows: list[dict[str, str]],
    preprocessing: PreprocessingConfig,
    selected_indices: set[int] | None = None,
) -> Iterator[tuple[int, dict[str, str], np.ndarray]]:
    wanted_by_path: dict[str, dict[int, list[tuple[int, dict[str, str]]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for index, row in enumerate(rows):
        if selected_indices is not None and index not in selected_indices:
            continue
        audio_path = str(Path(row["audio_path"]).resolve())
        audio_row = int(row["audio_parquet_row_index"])
        wanted_by_path[audio_path][audio_row].append((index, row))

    model_rate_ratio = Fraction(
        MODEL_SAMPLE_RATE_HZ, preprocessing.target_sample_rate_hz
    )
    for audio_path, wanted_rows in sorted(wanted_by_path.items()):
        parquet = pq.ParquetFile(audio_path)
        row_offset = 0
        found_rows: set[int] = set()
        highest_wanted = max(wanted_rows)
        for batch in parquet.iter_batches(columns=["record_number", "audio"], batch_size=16):
            records = batch.to_pylist()
            for batch_index, record in enumerate(records):
                source_row = row_offset + batch_index
                requested = wanted_rows.get(source_row)
                if requested is None:
                    continue
                found_rows.add(source_row)
                audio = record.get("audio") or {}
                payload = audio.get("bytes")
                if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
                    raise ValueError(
                        f"Empty embedded audio in {Path(audio_path).name} "
                        f"row {source_row}"
                    )
                if str(record.get("record_number", "")).strip() != requested[0][1][
                    "recording_id"
                ].strip():
                    raise ValueError(
                        f"Recording ID mismatch at {Path(audio_path).name} "
                        f"row {source_row}"
                    )
                with sf.SoundFile(BytesIO(payload)) as audio_file:
                    source_rate = int(audio_file.samplerate)
                    channel_count = int(audio_file.channels)
                    samples = audio_file.read(dtype="float32", always_2d=True)
                manifest_rate = int(requested[0][1]["source_sample_rate_hz"])
                manifest_channels = int(requested[0][1]["channel_count"])
                if source_rate != manifest_rate or channel_count != manifest_channels:
                    raise ValueError(
                        f"Audio header mismatch for {requested[0][1]['segment_id']}: "
                        f"header={source_rate} Hz/{channel_count} ch, "
                        f"manifest={manifest_rate} Hz/{manifest_channels} ch"
                    )
                if not np.isfinite(samples).all():
                    raise ValueError(
                        f"Non-finite samples in {Path(audio_path).name} row {source_row}"
                    )

                resampled = _resample_channels(
                    samples, source_rate, preprocessing.target_sample_rate_hz
                )
                waveform = _downmix_to_mono(resampled)
                waveform, _ = _peak_normalize(
                    waveform, preprocessing.normalization_target_peak
                )
                expected_segments = {
                    int(row["segment_index"]): (index, row)
                    for index, row in requested
                }
                produced_segments: set[int] = set()
                for segment_index, valid_samples, segment in _segment_audio(
                    waveform, preprocessing.segment_samples
                ):
                    if segment_index not in expected_segments:
                        continue
                    index, row = expected_segments[segment_index]
                    expected_valid = round(
                        float(row["segment_valid_seconds"])
                        * preprocessing.target_sample_rate_hz
                    )
                    if expected_valid != valid_samples:
                        raise ValueError(
                            f"Segment duration mismatch for {row['segment_id']}: "
                            f"decoded {valid_samples}, manifest {expected_valid} samples"
                        )
                    model_waveform = resample_poly(
                        segment,
                        model_rate_ratio.numerator,
                        model_rate_ratio.denominator,
                    ).astype(np.float32, copy=False)
                    if model_waveform.shape != (
                        round(
                            preprocessing.segment_duration_seconds
                            * MODEL_SAMPLE_RATE_HZ
                        ),
                    ):
                        raise RuntimeError(
                            f"Resampled waveform shape mismatch for {row['segment_id']}: "
                            f"{model_waveform.shape}"
                        )
                    yield index, row, model_waveform
                    produced_segments.add(segment_index)
                missing_segments = set(expected_segments) - produced_segments
                if missing_segments:
                    raise ValueError(
                        f"Missing segment indexes {sorted(missing_segments)} for "
                        f"{Path(audio_path).name} row {source_row}"
                    )
            row_offset += batch.num_rows
            if row_offset > highest_wanted:
                break
        missing_rows = set(wanted_rows) - found_rows
        if missing_rows:
            raise ValueError(
                f"Manifest row indexes not found in {audio_path}: "
                f"{sorted(missing_rows)[:10]}"
            )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.1 * (2**attempt))
    finally:
        if temporary.exists():
            temporary.unlink()


def _select_smoke_indices(
    manifests: dict[str, list[dict[str, str]]], per_split: int, seed: int
) -> dict[str, set[int]]:
    generator = np.random.default_rng(seed)
    result: dict[str, set[int]] = {}
    for split, rows in manifests.items():
        candidates: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            candidates[row["recording_id"]].append(index)
        recording_ids = sorted(candidates)
        count = min(per_split, len(recording_ids))
        selected_recordings = generator.choice(
            recording_ids, size=count, replace=False
        )
        result[split] = {
            candidates[str(recording_id)][0]
            for recording_id in selected_recordings
        }
    return result


def run_smoke_test(
    manifests: dict[str, list[dict[str, str]]],
    preprocessing: PreprocessingConfig,
    stage4_config: dict[str, Any],
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    per_split = 8
    selected = _select_smoke_indices(
        manifests, per_split, int(stage4_config["random_state"])
    )
    model, processor, device = _load_encoder(
        str(stage4_config["model_id"]), device_name
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    resource_monitor = _ResourceMonitor(device)

    all_vectors: list[np.ndarray] = []
    all_rows: list[dict[str, str]] = []
    batch_size = int(stage4_config["batch_size"])
    for split in SPLITS:
        pending_waveforms: list[np.ndarray] = []
        pending_rows: list[dict[str, str]] = []
        for _, row, waveform in _iter_waveforms(
            manifests[split], preprocessing, selected[split]
        ):
            pending_rows.append(row)
            pending_waveforms.append(waveform)
            if len(pending_waveforms) >= batch_size:
                vectors = _embed_batch(
                    model,
                    processor,
                    device,
                    pending_waveforms,
                    bool(stage4_config["mixed_precision_on_cuda"]),
                )
                all_vectors.extend(vectors)
                all_rows.extend(pending_rows)
                pending_waveforms.clear()
                pending_rows.clear()
                resource_monitor.sample()
        if pending_waveforms:
            vectors = _embed_batch(
                model,
                processor,
                device,
                pending_waveforms,
                bool(stage4_config["mixed_precision_on_cuda"]),
            )
            all_vectors.extend(vectors)
            all_rows.extend(pending_rows)
            resource_monitor.sample()

    embeddings = np.stack(all_vectors)
    result = {
        "status": "passed",
        "model_id": stage4_config["model_id"],
        "device": str(device),
        "smoke_segments": len(embeddings),
        "per_split_segments": {
            split: len(selected[split]) for split in SPLITS
        },
        "embedding_shape": list(embeddings.shape),
        "all_finite": bool(np.isfinite(embeddings).all()),
        "all_l2_normalized": bool(
            np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4)
        ),
        "distinct_segment_ids": len(
            {row["segment_id"] for row in all_rows}
        ),
        "distinct_recording_ids": len(
            {row["recording_id"] for row in all_rows}
        ),
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "resources": resource_monitor.summary(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if (
        not result["all_finite"]
        or not result["all_l2_normalized"]
        or result["distinct_segment_ids"] != result["smoke_segments"]
    ):
        raise RuntimeError(f"Smoke test validation failed: {result}")
    smoke_dir = output_dir / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(smoke_dir / "smoke_test_result.json", result)
    LOGGER.info("Smoke test passed: %s", result)
    return result


def _prepare_store(
    split_dir: Path,
    rows: list[dict[str, str]],
    manifest_path: Path,
    embedding_dim: int,
    fingerprint: str,
) -> tuple[np.memmap, np.memmap, Path]:
    split_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = split_dir / "metadata.csv"
    if metadata_path.is_file():
        if _manifest_hash(metadata_path) != _manifest_hash(manifest_path):
            raise ValueError(
                f"Existing metadata does not match the current split manifest: "
                f"{metadata_path}"
            )
    else:
        metadata_path.write_bytes(manifest_path.read_bytes())

    embeddings_path = split_dir / "embeddings.npy"
    completed_path = split_dir / "completed.npy"
    progress_path = split_dir / "progress.json"
    if progress_path.is_file():
        progress = _read_json(progress_path)
        if progress.get("fingerprint") != fingerprint:
            raise ValueError(
                f"Checkpoint config or manifest mismatch in {progress_path}; "
                "refusing to resume incompatible embeddings."
            )
    elif embeddings_path.exists() or completed_path.exists():
        raise ValueError(
            f"Incomplete output store has no checkpoint: {split_dir}. "
            "Move it aside before starting a new run."
        )
    else:
        _atomic_json(
            progress_path,
            {
                "fingerprint": fingerprint,
                "expected_segments": len(rows),
                "embedding_dim": embedding_dim,
                "completed_count": 0,
                "processing_seconds": 0.0,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    expected_embedding_shape = (len(rows), embedding_dim)
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path, mmap_mode="r+")
        if embeddings.shape != expected_embedding_shape or embeddings.dtype != np.float32:
            raise ValueError(f"Invalid embedding store shape or dtype: {embeddings_path}")
    else:
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=expected_embedding_shape,
        )
        embeddings.flush()
    if completed_path.exists():
        completed = np.load(completed_path, mmap_mode="r+")
        if completed.shape != (len(rows),) or completed.dtype != np.bool_:
            raise ValueError(f"Invalid completion store: {completed_path}")
    else:
        completed = np.lib.format.open_memmap(
            completed_path,
            mode="w+",
            dtype=np.bool_,
            shape=(len(rows),),
        )
        completed[:] = False
        completed.flush()
    return embeddings, completed, progress_path


def _extract_split(
    split: str,
    rows: list[dict[str, str]],
    manifest_path: Path,
    preprocessing: PreprocessingConfig,
    stage4_config: dict[str, Any],
    model: Any,
    processor: Any,
    device: torch.device,
    resource_monitor: _ResourceMonitor,
    output_dir: Path,
    fingerprint: str,
) -> dict[str, Any]:
    embeddings_path = output_dir / split / "embeddings.npy"
    embeddings, completed, progress_path = _prepare_store(
        output_dir / split,
        rows,
        manifest_path,
        int(model.config.projection_dim),
        fingerprint,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    previous_progress = _read_json(progress_path)
    previous_processing_seconds = float(
        previous_progress.get("processing_seconds", 0.0)
    )
    batch_size = int(stage4_config["batch_size"])
    pending_indices: list[int] = []
    pending_rows: list[dict[str, str]] = []
    pending_waveforms: list[np.ndarray] = []
    started = time.perf_counter()
    checkpoints = 0

    def flush_batch() -> None:
        nonlocal checkpoints
        if not pending_waveforms:
            return
        vectors = _embed_batch(
            model,
            processor,
            device,
            pending_waveforms,
            bool(stage4_config["mixed_precision_on_cuda"]),
        )
        if vectors.shape[1] != int(model.config.projection_dim):
            raise RuntimeError(
                f"Encoder dimension {vectors.shape[1]} does not match model "
                f"config {model.config.projection_dim}"
            )
        embeddings[pending_indices] = vectors
        embeddings.flush()
        completed[pending_indices] = True
        completed.flush()
        resource_monitor.sample()
        checkpoints += 1
        progress = {
            "fingerprint": fingerprint,
            "expected_segments": len(rows),
            "embedding_dim": int(model.config.projection_dim),
            "completed_count": int(np.count_nonzero(completed)),
            "processing_seconds": round(
                previous_processing_seconds + time.perf_counter() - started, 3
            ),
            "last_segment_id": pending_rows[-1]["segment_id"],
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(progress_path, progress)
        if checkpoints == 1 or checkpoints % 250 == 0:
            LOGGER.info(
                "%s: checkpoint %d, %d/%d embeddings",
                split,
                checkpoints,
                progress["completed_count"],
                len(rows),
            )
        pending_indices.clear()
        pending_rows.clear()
        pending_waveforms.clear()

    remaining = set(np.flatnonzero(~completed).tolist())
    LOGGER.info(
        "%s: resuming with %d/%d segments remaining",
        split,
        len(remaining),
        len(rows),
    )
    for index, row, waveform in _iter_waveforms(
        rows, preprocessing, remaining
    ):
        if completed[index]:
            raise RuntimeError(f"Duplicate work encountered for {row['segment_id']}")
        pending_indices.append(index)
        pending_rows.append(row)
        pending_waveforms.append(waveform)
        if len(pending_waveforms) >= batch_size:
            flush_batch()
    flush_batch()

    missing_indices = np.flatnonzero(~completed)
    if len(missing_indices):
        first_missing = rows[int(missing_indices[0])]["segment_id"]
        raise RuntimeError(
            f"{split} extraction left {len(missing_indices)} embeddings missing; "
            f"first missing segment: {first_missing}"
        )
    matrix = np.asarray(embeddings)
    if matrix.shape != (len(rows), int(model.config.projection_dim)):
        raise RuntimeError(f"{split} embedding matrix has an unexpected shape")
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"{split} embeddings contain NaN or infinite values")

    result = {
        "split": split,
        "segments": len(rows),
        "embedding_dim": int(model.config.projection_dim),
        "completed": int(np.count_nonzero(completed)),
        "processing_seconds": round(
            previous_processing_seconds + time.perf_counter() - started, 3
        ),
        "device": str(device),
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "embedding_file": str(embeddings_path),
        "metadata_file": str(output_dir / split / "metadata.csv"),
    }
    LOGGER.info("%s: complete: %s", split, result)
    return result


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(output_dir / "stage4.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)


def extract_all(
    output_dir: Path = OUTPUT_DIR,
    device_name: str = "auto",
    preprocessing_config_path: Path = PREPROCESSING_CONFIG_PATH,
    stage4_config_path: Path = STAGE4_CONFIG_PATH,
    splits_dir: Path = SPLIT_DIR,
    smoke_test_only: bool = False,
) -> dict[str, Any]:
    _setup_logging(output_dir)
    stage4_config = _read_json(stage4_config_path)
    preprocessing = PreprocessingConfig.load(preprocessing_config_path)
    manifests: dict[str, list[dict[str, str]]] = {}
    manifest_paths: dict[str, Path] = {}
    for split in SPLITS:
        manifest_paths[split] = splits_dir / f"{split}.csv"
        manifests[split] = _read_manifest(manifest_paths[split], split)
    total_ids: set[str] = set()
    for split, rows in manifests.items():
        identifiers = {row["segment_id"] for row in rows}
        overlaps = total_ids & identifiers
        if overlaps:
            raise ValueError(
                f"Segment IDs overlap between split manifests: {sorted(overlaps)[:5]}"
            )
        total_ids.update(identifiers)

    if smoke_test_only:
        return run_smoke_test(
            manifests,
            preprocessing,
            stage4_config,
            output_dir,
            device_name,
        )

    model, processor, device = _load_encoder(
        str(stage4_config["model_id"]), device_name
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    resource_monitor = _ResourceMonitor(device)
    model_config = model.config.to_dict()
    model_commit = getattr(model.config, "_commit_hash", None)
    fingerprint_payload = {
        "model_id": stage4_config["model_id"],
        "model_commit": model_commit,
        "model_config": model_config,
        "preprocessing_config": preprocessing.as_dict(),
        "model_sample_rate_hz": MODEL_SAMPLE_RATE_HZ,
        "manifest_sha256": {
            split: _manifest_hash(manifest_paths[split]) for split in SPLITS
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    encoder_configuration = {
        "encoder_name": stage4_config["encoder_name"],
        "model_id": stage4_config["model_id"],
        "model_commit": model_commit,
        "model_config": model_config,
        "model_sample_rate_hz": MODEL_SAMPLE_RATE_HZ,
        "model_max_frequency_hz": int(
            processor.feature_extractor.frequency_max
        ),
        "model_min_frequency_hz": int(
            processor.feature_extractor.frequency_min
        ),
        "model_mel_bins": int(processor.feature_extractor.feature_size),
        "model_clip_length_seconds": float(
            processor.feature_extractor.max_length_s
        ),
        "model_padding": processor.feature_extractor.padding,
        "preprocessing_config": preprocessing.as_dict(),
        "batch_size": int(stage4_config["batch_size"]),
        "mixed_precision_on_cuda": bool(
            stage4_config["mixed_precision_on_cuda"]
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "split_manifest_sha256": fingerprint_payload["manifest_sha256"],
    }
    _atomic_json(output_dir / "encoder_configuration.json", encoder_configuration)
    started = time.perf_counter()
    split_results: list[dict[str, Any]] = []
    for split in SPLITS:
        split_results.append(
            _extract_split(
                split,
                manifests[split],
                manifest_paths[split],
                preprocessing,
                stage4_config,
                model,
                processor,
                device,
                resource_monitor,
                output_dir,
                fingerprint,
            )
        )
        resource_monitor.sample(force=True)
        _atomic_json(
            output_dir / "generation_summary.json",
            {
                "status": "in_progress",
                "model_id": stage4_config["model_id"],
                "device": str(device),
                "completed_splits": split_results,
                "expected_segment_count": len(total_ids),
                "completed_segment_count": sum(
                    item["completed"] for item in split_results
                ),
                "resources": resource_monitor.summary(),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    final_summary = {
        "status": "complete",
        "model_id": stage4_config["model_id"],
        "model_commit": model_commit,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "expected_segment_count": len(total_ids),
        "completed_segment_count": sum(item["completed"] for item in split_results),
        "embedding_dim": int(model.config.projection_dim),
        "processing_seconds": round(time.perf_counter() - started, 3),
        "splits": split_results,
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "resources": resource_monitor.summary(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output_dir / "generation_summary.json", final_summary)
    LOGGER.info("Stage 4 embedding extraction complete: %s", final_summary)
    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract resumable frozen pretrained audio embeddings."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--splits-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=PREPROCESSING_CONFIG_PATH,
    )
    parser.add_argument(
        "--stage4-config",
        type=Path,
        default=STAGE4_CONFIG_PATH,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Use CUDA when available by default; CPU fallback is supported.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 8 segments per split and validate embeddings without full extraction.",
    )
    args = parser.parse_args()
    result = extract_all(
        output_dir=args.output_dir,
        device_name=args.device,
        preprocessing_config_path=args.preprocessing_config,
        stage4_config_path=args.stage4_config,
        splits_dir=args.splits_dir,
        smoke_test_only=args.smoke_test,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
