"""Streaming Parquet inspection, reporting, and optional balanced sampling."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Sampler


_LABEL_NAMES = (
    "species_code",
    "species",
    "scientific_name",
    "taxon",
    "taxon_name",
    "label",
)
_AUDIO_NAMES = ("audio", "audio_bytes", "audio_data", "waveform", "samples")
_ID_NAMES = ("recording_id", "record_id", "record_number", "clip_id", "file_id")
_CONDITION_FIELDS = (
    ("channel_recorded", ("sound.channel.recorded", "channel.recorded", "channel_recorded")),
    (
        "channel_multiplexed",
        ("sound.channel.multiplexed", "channel.multiplexed", "channel_multiplexed"),
    ),
    ("channel_side", ("sound.channel.side", "channel.side", "channel_side")),
    (
        "channel_source_code",
        ("sound.channel._source_nc", "channel.source_code", "device_id", "hydrophone_id"),
    ),
)


@dataclass(frozen=True)
class DatasetSchema:
    """Column paths inferred from a Parquet shard's Arrow schema."""

    top_level_columns: tuple[str, ...]
    label: str | None
    audio: str | None
    scientific_name: str | None
    display_name: str | None
    record_id: str | None
    date: str | None
    location: str | None
    location_coordinates: str | None
    sample_rate: str | None
    recording_source: str | None
    recording_conditions: tuple[tuple[str, str], ...]


def _flatten_struct_fields(
    schema: pa.Schema, prefix: str = ""
) -> list[tuple[str, pa.DataType]]:
    result: list[tuple[str, pa.DataType]] = []
    for field in schema:
        path = f"{prefix}.{field.name}" if prefix else field.name
        result.append((path, field.type))
        if pa.types.is_struct(field.type):
            result.extend(_flatten_struct_fields(field.type, path))
    return result


def _path_rank(path: str, candidates: Sequence[str]) -> tuple[int, int, str]:
    leaf = path.rsplit(".", 1)[-1].casefold()
    lowered = path.casefold()
    for rank, candidate in enumerate(candidates):
        if lowered == candidate:
            return (rank, 0, lowered)
        if leaf == candidate:
            return (rank, 1, lowered)
    return (len(candidates), 2, lowered)


def _find_path(
    fields: Sequence[tuple[str, pa.DataType]], candidates: Sequence[str]
) -> str | None:
    matches = [
        path
        for path, _ in fields
        if _path_rank(path, candidates)[0] < len(candidates)
    ]
    return min(matches, key=lambda p: _path_rank(p, candidates)) if matches else None


def _has_binary_descendant(dtype: pa.DataType) -> bool:
    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        return True
    if pa.types.is_struct(dtype):
        return any(_has_binary_descendant(field.type) for field in dtype)
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return _has_binary_descendant(dtype.value_type)
    return False


def _infer_audio_field(schema: pa.Schema) -> str | None:
    roots = list(schema)
    named = [field for field in roots if field.name.casefold() in _AUDIO_NAMES]
    named.sort(key=lambda f: _AUDIO_NAMES.index(f.name.casefold()))
    binary_fields = [
        field
        for field in roots
        if _has_binary_descendant(field.type)
    ]
    candidates = named + [field for field in binary_fields if field not in named]
    if candidates:
        return candidates[0].name
    for field in roots:
        if pa.types.is_floating(field.type) or pa.types.is_integer(field.type):
            if field.name.casefold() in _AUDIO_NAMES:
                return field.name
        if (
            (pa.types.is_list(field.type) or pa.types.is_large_list(field.type))
            and (
                pa.types.is_floating(field.type.value_type)
                or pa.types.is_integer(field.type.value_type)
            )
        ):
            if field.name.casefold() in _AUDIO_NAMES:
                return field.name
    return None


def discover_schema(parquet_path: str | Path) -> DatasetSchema:
    """Inspect an Arrow schema and infer likely label, audio, and metadata fields."""
    schema = pq.ParquetFile(parquet_path).schema_arrow
    fields = _flatten_struct_fields(schema)
    label = _find_path(fields, _LABEL_NAMES)
    audio = _infer_audio_field(schema)
    recording_source = _find_path(
        fields, ("recording_source", "recording_session", "session_id")
    )
    if recording_source is None and any(path.casefold() == "source" for path, _ in fields):
        recording_source = "source"
    recording_conditions = tuple(
        (name, path)
        for name, candidates in _CONDITION_FIELDS
        if (path := _find_path(fields, candidates)) is not None
    )
    return DatasetSchema(
        top_level_columns=tuple(schema.names),
        label=label,
        audio=audio,
        scientific_name=_find_path(fields, ("scientific_name", "binomial_name")),
        display_name=_find_path(fields, ("display_name", "common_name")),
        record_id=_find_path(fields, _ID_NAMES),
        date=_find_path(fields, ("observation_date", "recording_date", "date")),
        location=_find_path(
            fields,
            ("location.name", "location_name", "site_name", "site", "location"),
        ),
        location_coordinates=_find_path(
            fields,
            ("location.coordinates", "coordinates", "gps_coordinates"),
        ),
        sample_rate=_find_path(
            fields, ("sample_rate", "sampling_rate", "samplerate")
        ),
        recording_source=recording_source,
        recording_conditions=recording_conditions,
    )


def nested_value(record: Mapping[str, Any], path: str | None) -> Any:
    if not path:
        return None
    value: Any = record
    for component in path.split("."):
        if isinstance(value, Mapping):
            value = value.get(component)
        else:
            return None
    return value


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value) > 0
    if isinstance(value, Mapping):
        return any(is_present(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(is_present(item) for item in value)
    return True


def _audio_payload(value: Any) -> tuple[str, Any]:
    """Find a recognizable payload inside a decoded Arrow audio value."""
    if value is None:
        return "missing", None
    if isinstance(value, Mapping):
        for key in ("bytes", "data", "array", "samples", "waveform"):
            if key in value and value[key] is not None:
                kind, payload = _audio_payload(value[key])
                if kind != "missing":
                    return kind, payload
        for key in ("path", "file", "uri"):
            if is_present(value.get(key)):
                return "path", str(value[key])
        return "missing", None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ("bytes", bytes(value)) if len(value) else ("empty", b"")
    if isinstance(value, str):
        return ("path", value) if value.strip() else ("empty", "")
    if hasattr(value, "shape") or isinstance(value, (list, tuple)):
        return "array", value
    return "unsupported", value


def _resolve_audio_path(path: str, parquet_path: Path) -> Path | None:
    candidate = Path(path)
    candidates = [candidate] if candidate.is_absolute() else [
        parquet_path.parent / candidate,
        parquet_path.parent.parent / candidate,
    ]
    for item in candidates:
        if item.is_file():
            return item
    return None


def _audio_inspection(
    kind: str,
    payload: Any,
    parquet_path: Path,
    sample_rate_hint: Any,
) -> tuple[str, float | None, int | None, int | None, str | None]:
    """Return status, duration, rate, channels, and content fingerprint."""
    if kind in ("missing", "empty"):
        return kind, None, None, None, None

    import soundfile as sf

    fingerprint: str | None = None
    source: Any
    if kind == "bytes":
        fingerprint = hashlib.sha256(payload).hexdigest()
        source = BytesIO(payload)
    elif kind == "path":
        resolved = _resolve_audio_path(payload, parquet_path)
        if resolved is None:
            return "unresolved_path", None, None, None, None
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        fingerprint = digest.hexdigest()
        source = resolved
    elif kind == "array":
        import numpy as np

        array = np.asarray(payload)
        if array.size == 0:
            return "empty", None, None, None, None
        rate = _positive_int(sample_rate_hint)
        if rate is None:
            return "missing_sample_rate", None, None, None, None
        frames = int(array.shape[0])
        channels = int(array.shape[1]) if array.ndim > 1 else 1
        fingerprint = hashlib.sha256(
            str(array.shape).encode("ascii") + array.tobytes()
        ).hexdigest()
        return "valid", frames / rate, rate, channels, fingerprint
    else:
        return "unsupported", None, None, None, None

    try:
        with sf.SoundFile(source) as audio_file:
            frames = int(audio_file.frames)
            rate = int(audio_file.samplerate)
            channels = int(audio_file.channels)
            if frames <= 0 or rate <= 0 or channels <= 0:
                return "corrupt", None, rate, channels, fingerprint
            remaining = frames
            while remaining:
                chunk = audio_file.read(
                    min(65536, remaining), dtype="float32", always_2d=False
                )
                read_frames = int(chunk.shape[0])
                if read_frames <= 0:
                    return "corrupt", None, rate, channels, fingerprint
                remaining -= read_frames
            return "valid", frames / rate, rate, channels, fingerprint
    except (RuntimeError, OSError, ValueError, TypeError):
        return "corrupt", None, None, None, fingerprint


def _positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result > 0 else None


def iter_parquet_records(
    files: Sequence[str | Path],
    batch_size: int = 256,
    columns: Sequence[str] | None = None,
) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield rows from Parquet shards batchwise without materializing the dataset."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    for file in files:
        path = Path(file)
        parquet_file = pq.ParquetFile(path)
        selected = list(columns) if columns is not None else None
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=selected):
            for record in batch.to_pylist():
                yield path, record


class ParquetAudioDataset:
    """Map-style Parquet reader with a bounded row-group cache.

    Rows retain their original metadata; audio payloads are returned in their
    stored representation so callers control decoding and preprocessing.
    """

    def __init__(
        self,
        files: Sequence[str | Path],
        columns: Sequence[str] | None = None,
        cache_row_groups: int = 1,
    ) -> None:
        if cache_row_groups < 1:
            raise ValueError("cache_row_groups must be at least one")
        self.files = [Path(file) for file in files]
        self.columns = list(columns) if columns is not None else None
        self.cache_row_groups = cache_row_groups
        self._locations: list[tuple[int, int, int]] = []
        for file_index, path in enumerate(self.files):
            parquet_file = pq.ParquetFile(path)
            for group_index in range(parquet_file.metadata.num_row_groups):
                group_rows = parquet_file.metadata.row_group(group_index).num_rows
                self._locations.extend(
                    (file_index, group_index, offset) for offset in range(group_rows)
                )
        self._cache: OrderedDict[tuple[int, int], pa.Table] = OrderedDict()

    def __len__(self) -> int:
        return len(self._locations)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        file_index, group_index, offset = self._locations[index]
        cache_key = (file_index, group_index)
        table = self._cache.get(cache_key)
        if table is None:
            table = pq.ParquetFile(self.files[file_index]).read_row_group(
                group_index, columns=self.columns
            )
            self._cache[cache_key] = table
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.cache_row_groups:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(cache_key)
        return table.slice(offset, 1).to_pylist()[0]


class SpeciesBalancedBatchSampler(Sampler[list[int]]):
    """Build equal-species batches without repeating rows within an epoch.

    Missing-label rows are omitted. Each batch contains ``classes_per_batch``
    distinct species and the same number of records from each. Rare-class
    leftovers are retained in the dataset but may not fit into a full balanced
    batch; singleton records are used at most once per epoch.

    To include singleton species, configure ``batch_size`` equal to
    ``classes_per_batch`` (one example per species). Larger per-species batch
    counts make classes with fewer available records ineligible.
    """

    def __init__(
        self,
        species_labels: Sequence[Any],
        batch_size: int,
        classes_per_batch: int = 8,
        seed: int = 0,
    ) -> None:
        if classes_per_batch < 2:
            raise ValueError("classes_per_batch must be at least 2")
        if batch_size <= 0 or batch_size % classes_per_batch:
            raise ValueError(
                "batch_size must be positive and divisible by classes_per_batch"
            )
        self.species_labels = species_labels
        self.batch_size = batch_size
        self.classes_per_batch = classes_per_batch
        self.samples_per_species = batch_size // classes_per_batch
        self.seed = seed
        self.epoch = 0
        self._batches: list[list[int]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._batches = None

    def _build_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(self.species_labels):
            if is_present(label):
                groups[str(label)].append(index)
        for indices in groups.values():
            rng.shuffle(indices)

        queue = deque(
            species
            for species in rng.sample(list(groups), k=len(groups))
            if len(groups[species]) >= self.samples_per_species
        )
        batches: list[list[int]] = []
        while len(queue) >= self.classes_per_batch:
            chosen = [queue.popleft() for _ in range(self.classes_per_batch)]
            batch: list[int] = []
            for species in chosen:
                batch.extend(
                    groups[species][: self.samples_per_species]
                )
                del groups[species][: self.samples_per_species]
                if len(groups[species]) >= self.samples_per_species:
                    queue.append(species)
            rng.shuffle(batch)
            batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        if self._batches is None:
            self._batches = self._build_batches()
        yield from self._batches

    def __len__(self) -> int:
        if self._batches is None:
            self._batches = self._build_batches()
        return len(self._batches)


def grouped_split_indices(
    groups: Sequence[Any],
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Split whole metadata groups; rows with missing group values stay in train."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    train_indices: list[int] = []
    for index, group in enumerate(groups):
        if not is_present(group):
            train_indices.append(index)
        else:
            grouped_indices[str(group)].append(index)
    group_names = list(grouped_indices)
    if len(group_names) < 2:
        raise ValueError("At least two non-missing groups are required for a split")
    random.Random(seed).shuffle(group_names)
    validation_group_count = min(
        len(group_names) - 1,
        max(1, round(len(group_names) * validation_fraction)),
    )
    validation_names = set(group_names[:validation_group_count])
    validation_indices: list[int] = []
    for name, indices in grouped_indices.items():
        (validation_indices if name in validation_names else train_indices).extend(
            indices
        )
    return sorted(train_indices), sorted(validation_indices)


def evaluate_cluster_assignments(
    species_labels: Sequence[Any],
    cluster_labels: Sequence[Any],
    tiers: Mapping[str, str],
    recording_domains: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Summarize post-hoc cluster/species and cluster/domain association by tier.

    These are diagnostic associations, not evidence that clusters are call types.
    """
    if len(species_labels) != len(cluster_labels):
        raise ValueError("species_labels and cluster_labels must have equal length")
    if recording_domains is not None and len(recording_domains) != len(cluster_labels):
        raise ValueError("recording_domains must match cluster_labels length")
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    tier_labels: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(species_labels):
        if is_present(label):
            tier = tiers.get(str(label), "unclassified")
            tier_labels["common" if tier == "dominant" else tier].append(index)

    results: list[dict[str, Any]] = []
    for tier in ("common", "moderate", "rare", "unclassified"):
        indices = tier_labels.get(tier, [])
        if not indices:
            continue
        species = [str(species_labels[i]) for i in indices]
        clusters = [cluster_labels[i] for i in indices]
        row: dict[str, Any] = {
            "tier": tier,
            "records": len(indices),
            "species_count": len(set(species)),
            "cluster_count": len(set(clusters)),
            "noise_fraction": sum(value == -1 for value in clusters) / len(clusters),
            "species_cluster_ari": (
                adjusted_rand_score(species, clusters)
                if len(set(species)) > 1 and len(set(clusters)) > 1
                else None
            ),
        }
        if recording_domains is not None:
            domains = [recording_domains[i] for i in indices]
            row["cluster_domain_nmi"] = (
                normalized_mutual_info_score(clusters, domains)
                if len(set(domains)) > 1 and len(set(clusters)) > 1
                else None
            )
        results.append(row)
    return results


def _as_values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return [item for item in value if is_present(item)]
    return [value] if is_present(value) else []


def _safe_key(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: (
                        f"{row[column]:.6f}"
                        if isinstance(row.get(column), float)
                        and math.isfinite(row[column])
                        else row.get(column, "")
                    )
                    for column in columns
                }
            )


def _make_charts(
    output_dir: Path,
    distribution: Sequence[Mapping[str, Any]],
    durations: Mapping[str, Sequence[float]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labeled = [row for row in distribution if row["imbalance_tier"] != "unlabeled"]
    sorted_rows = sorted(labeled, key=lambda row: row["record_count"])
    height = max(8, min(30, 0.23 * len(sorted_rows)))
    fig, axis = plt.subplots(figsize=(12, height))
    axis.barh(
        [row["species_label"] for row in sorted_rows],
        [row["record_count"] for row in sorted_rows],
        color=[
            {"rare": "#4C78A8", "moderate": "#F2CF5B", "dominant": "#E45756"}.get(
                row["imbalance_tier"], "#999999"
            )
            for row in sorted_rows
        ],
    )
    axis.set_xlabel("Audio records")
    axis.set_ylabel("Species label")
    axis.set_title("Watkins dataset: records per species")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    count_chart = output_dir / "species_distribution.png"
    fig.savefig(count_chart, dpi=160)
    plt.close(fig)

    duration_rows = [
        (label, values)
        for label, values in durations.items()
        if values and not label.startswith("[UNLABELED]")
    ]
    duration_chart = output_dir / "species_duration_distribution.png"
    if duration_rows:
        duration_rows.sort(key=lambda item: sorted(item[1])[len(item[1]) // 2])
        fig, axis = plt.subplots(figsize=(12, height))
        axis.boxplot(
            [values for _, values in duration_rows],
            orientation="horizontal",
            tick_labels=[label for label, _ in duration_rows],
            showfliers=False,
        )
        axis.set_xlabel("Audio duration (seconds)")
        axis.set_ylabel("Species label")
        axis.set_title("Audio duration distribution by species")
        axis.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(duration_chart, dpi=160)
        plt.close(fig)
        return [count_chart.name, duration_chart.name]
    return [count_chart.name]


def analyze_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Audit Parquet shards, preserve source data, and write reports and charts."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet shards found in {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = discover_schema(files[0])
    if not schema.top_level_columns:
        raise ValueError(f"No columns found in {files[0]}")
    expected_columns = set(schema.top_level_columns)
    expected_arrow_schema = pq.ParquetFile(files[0]).schema_arrow
    for file in files[1:]:
        current_arrow_schema = pq.ParquetFile(file).schema_arrow
        current_schema = discover_schema(file)
        if (
            set(current_schema.top_level_columns) != expected_columns
            or not current_arrow_schema.equals(expected_arrow_schema, check_metadata=False)
        ):
            raise ValueError(
                f"Parquet shard schema mismatch in {file}; "
                "all shards must expose the same Arrow schema."
            )

    species: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "record_count": 0,
            "missing_audio_count": 0,
            "empty_audio_count": 0,
            "corrupt_audio_count": 0,
            "unresolved_audio_path_count": 0,
            "duplicate_record_id_count": 0,
            "duplicate_audio_count": 0,
            "duration_count": 0,
            "duration_total_seconds": 0.0,
            "durations": [],
            "dates": 0,
            "locations": 0,
            "location_names": 0,
            "location_coordinates": 0,
            "sample_rates": 0,
            "recording_sources": 0,
            "location_values": set(),
            "condition_counts": Counter(),
            "primary_label_count": 0,
            "fallback_label_count": 0,
            "unlabeled_count": 0,
        }
    )
    quality = Counter()
    quality["shards"] = len(files)
    quality["audio_kind"] = Counter()
    audio_hashes: set[str] = set()
    record_ids: set[str] = set()
    all_durations: list[float] = []
    metadata_values = Counter()

    for file, record in iter_parquet_records(
        files, batch_size=batch_size, columns=schema.top_level_columns
    ):
        quality["record_count"] += 1
        label_value = nested_value(record, schema.label)
        label_status = "labeled"
        if is_present(label_value):
            label = str(label_value).strip()
        else:
            fallback = nested_value(record, schema.scientific_name)
            if is_present(fallback):
                label = str(fallback).strip()
                label_status = "fallback_name"
            else:
                fallback = nested_value(record, schema.display_name)
                label = (
                    f"[UNLABELED] {str(fallback).strip()}"
                    if is_present(fallback)
                    else "[UNLABELED] missing species label"
                )
                label_status = "unlabeled"
        bucket = species[label]
        bucket["record_count"] += 1
        if label_status == "unlabeled":
            bucket["unlabeled_count"] += 1
            quality["missing_label"] += 1
        elif label_status == "fallback_name":
            bucket["fallback_label_count"] += 1
            quality["missing_primary_label"] += 1
        else:
            bucket["primary_label_count"] += 1

        record_id = nested_value(record, schema.record_id)
        if is_present(record_id):
            record_id_key = _safe_key(record_id)
            if record_id_key in record_ids:
                quality["duplicate_record_ids"] += 1
                bucket["duplicate_record_id_count"] += 1
            else:
                record_ids.add(record_id_key)

        for key, path in (
            ("dates", schema.date),
            ("sample_rates", schema.sample_rate),
            ("recording_sources", schema.recording_source),
        ):
            value = nested_value(record, path)
            if is_present(value):
                metadata_values[key] += 1
                bucket[key] += 1
        location_name = nested_value(record, schema.location)
        coordinates = nested_value(record, schema.location_coordinates)
        if is_present(location_name) or is_present(coordinates):
            metadata_values["locations"] += 1
            bucket["locations"] += 1
        if is_present(location_name):
            metadata_values["location_names"] += 1
            bucket["location_names"] += 1
            bucket["location_values"].update(
                _safe_key(value) for value in _as_values(location_name)
            )
        if is_present(coordinates):
            metadata_values["location_coordinates"] += 1
            bucket["location_coordinates"] += 1
        for condition_name, condition_path in schema.recording_conditions:
            if is_present(nested_value(record, condition_path)):
                metadata_values[condition_name] += 1
                bucket["condition_counts"][condition_name] += 1

        audio_value = nested_value(record, schema.audio)
        audio_kind, payload = _audio_payload(audio_value)
        quality["audio_kind"][audio_kind] += 1
        rate_hint = nested_value(record, schema.sample_rate)
        status, duration, decoded_rate, _, fingerprint = _audio_inspection(
            audio_kind, payload, file, rate_hint
        )
        quality[f"audio_{status}"] += 1
        if status in ("missing", "empty"):
            bucket["missing_audio_count"] += 1
            if status == "empty":
                bucket["empty_audio_count"] += 1
        elif status == "corrupt":
            bucket["corrupt_audio_count"] += 1
        elif status == "unresolved_path":
            bucket["unresolved_audio_path_count"] += 1
        if fingerprint:
            if fingerprint in audio_hashes:
                quality["duplicate_audio_payloads"] += 1
                bucket["duplicate_audio_count"] += 1
            else:
                audio_hashes.add(fingerprint)
        if decoded_rate is not None and rate_hint is not None:
            try:
                if int(rate_hint) != decoded_rate:
                    quality["sample_rate_mismatches"] += 1
            except (TypeError, ValueError, OverflowError):
                quality["invalid_sample_rate_metadata"] += 1
        if duration is not None and math.isfinite(duration):
            bucket["duration_count"] += 1
            bucket["duration_total_seconds"] += duration
            bucket["durations"].append(duration)
            all_durations.append(duration)

    labeled_counts = [
        value["record_count"]
        for label, value in species.items()
        if not label.startswith("[UNLABELED]")
    ]
    if labeled_counts:
        low_cutoff, high_cutoff = np.quantile(labeled_counts, [1 / 3, 2 / 3])
    else:
        low_cutoff = high_cutoff = 0

    distribution_rows: list[dict[str, Any]] = []
    tiers: dict[str, str] = {}
    duration_values: dict[str, Sequence[float]] = {}
    for label, values in sorted(
        species.items(), key=lambda item: (-item[1]["record_count"], item[0])
    ):
        if label.startswith("[UNLABELED]"):
            tier = "unlabeled"
        elif values["record_count"] <= low_cutoff:
            tier = "rare"
        elif values["record_count"] <= high_cutoff:
            tier = "moderate"
        else:
            tier = "dominant"
        tiers[label] = tier
        durations = values["durations"]
        duration_values[label] = durations
        label_statuses = sum(
            values[key] > 0
            for key in ("primary_label_count", "fallback_label_count", "unlabeled_count")
        )
        if values["unlabeled_count"] == values["record_count"]:
            label_status = "unlabeled"
        elif label_statuses > 1:
            label_status = "mixed"
        elif values["fallback_label_count"]:
            label_status = "fallback_name"
        else:
            label_status = "labeled"
        distribution_rows.append(
            {
                "species_label": label,
                "label_source": (
                    schema.label
                    if label_status in ("labeled", "mixed")
                    else (schema.scientific_name if label_status == "fallback_name" else "")
                ),
                "label_status": label_status,
                "record_count": values["record_count"],
                "imbalance_tier": tier,
                "missing_audio_count": values["missing_audio_count"],
                "empty_audio_count": values["empty_audio_count"],
                "corrupt_audio_count": values["corrupt_audio_count"],
                "unresolved_audio_path_count": values["unresolved_audio_path_count"],
                "duplicate_record_id_count": values["duplicate_record_id_count"],
                "duplicate_audio_count": values["duplicate_audio_count"],
                "duration_available_count": len(durations),
                "duration_total_seconds": values["duration_total_seconds"],
                "duration_mean_seconds": (
                    values["duration_total_seconds"] / len(durations) if durations else ""
                ),
                "duration_median_seconds": (
                    float(np.median(durations)) if durations else ""
                ),
                "date_metadata_count": values["dates"],
                "location_metadata_count": values["locations"],
                "location_name_metadata_count": values["location_names"],
                "location_coordinate_metadata_count": values["location_coordinates"],
                "distinct_location_values": len(values["location_values"]),
                "sample_rate_metadata_count": values["sample_rates"],
                "recording_source_metadata_count": values["recording_sources"],
                **{
                    f"{name}_metadata_count": values["condition_counts"][name]
                    for name, _ in schema.recording_conditions
                },
            }
        )

    quality_rows: list[dict[str, Any]] = [
        {"metric": "parquet_shards", "value": len(files), "details": str(data_dir)},
        {"metric": "total_records", "value": quality["record_count"], "details": ""},
        {
            "metric": "species_label_groups_including_unlabeled",
            "value": len(species),
            "details": "Includes unlabeled groups split by display name",
        },
        {
            "metric": "distinct_primary_species_labels",
            "value": sum(not label.startswith("[UNLABELED]") for label in species),
            "details": schema.label or "No primary species field detected",
        },
        {
            "metric": "unlabeled_display_name_groups",
            "value": sum(label.startswith("[UNLABELED]") for label in species),
            "details": "Not included in species count-based imbalance tiers",
        },
        {"metric": "primary_label_column", "value": schema.label or "", "details": ""},
        {"metric": "audio_column", "value": schema.audio or "", "details": ""},
        {"metric": "missing_species_labels", "value": quality["missing_label"], "details": "No primary or fallback species label"},
        {"metric": "missing_primary_labels", "value": quality["missing_primary_label"], "details": "Fallback scientific/common name used"},
        {"metric": "duplicate_record_ids", "value": quality["duplicate_record_ids"], "details": f"Repeated values in {schema.record_id or 'no inferred ID column'}"},
        {"metric": "duplicate_audio_payloads", "value": quality["duplicate_audio_payloads"], "details": "Exact SHA-256 matches; source rows retained"},
        {"metric": "sample_rate_mismatches", "value": quality["sample_rate_mismatches"], "details": "Decoded rate differs from metadata"},
    ]
    for status in (
        "valid",
        "missing",
        "empty",
        "corrupt",
        "unresolved_path",
        "unsupported",
        "missing_sample_rate",
    ):
        quality_rows.append(
            {
                "metric": f"audio_{status}",
                "value": quality[f"audio_{status}"],
                "details": "",
            }
        )
    for kind, count in sorted(quality["audio_kind"].items()):
        quality_rows.append(
            {"metric": f"audio_representation_{kind}", "value": count, "details": ""}
        )
    for field_name in (
        "dates",
        "locations",
        "sample_rates",
        "recording_sources",
        "location_names",
        "location_coordinates",
    ):
        quality_rows.append(
            {
                "metric": f"records_with_{field_name}",
                "value": metadata_values[field_name],
                "details": {
                    "dates": schema.date,
                    "locations": (
                        f"{schema.location or 'not detected'} or "
                        f"{schema.location_coordinates or 'not detected'}"
                    ),
                    "sample_rates": schema.sample_rate,
                    "recording_sources": schema.recording_source,
                    "location_names": schema.location or "not detected",
                    "location_coordinates": schema.location_coordinates or "not detected",
                }[field_name],
            }
        )
    for field_name, field_path in schema.recording_conditions:
        quality_rows.append(
            {
                "metric": f"records_with_{field_name}",
                "value": metadata_values[field_name],
                "details": field_path,
            }
        )
    quality_rows.append(
        {
            "metric": "duration_known_records",
            "value": len(all_durations),
            "details": "Decoded from audio payload; audio metadata cut_size was not used",
        }
    )
    quality_rows.append(
        {
            "metric": "total_audio_duration_seconds",
            "value": sum(all_durations),
            "details": "Sum of successfully decoded durations",
        }
    )

    _write_csv(
        output_dir / "species_distribution.csv",
        [
            "species_label",
            "label_source",
            "label_status",
            "record_count",
            "imbalance_tier",
            "missing_audio_count",
            "empty_audio_count",
            "corrupt_audio_count",
            "unresolved_audio_path_count",
            "duplicate_record_id_count",
            "duplicate_audio_count",
            "duration_available_count",
            "duration_total_seconds",
            "duration_mean_seconds",
            "duration_median_seconds",
            "date_metadata_count",
            "location_metadata_count",
            "location_name_metadata_count",
            "location_coordinate_metadata_count",
            "distinct_location_values",
            "sample_rate_metadata_count",
            "recording_source_metadata_count",
            *[
                f"{name}_metadata_count"
                for name, _ in schema.recording_conditions
            ],
        ],
        distribution_rows,
    )
    _write_csv(
        output_dir / "dataset_quality_report.csv",
        ("metric", "value", "details"),
        quality_rows,
    )
    charts = _make_charts(output_dir, distribution_rows, duration_values)

    tier_totals = Counter(row["imbalance_tier"] for row in distribution_rows)
    tier_species = Counter(
        row["imbalance_tier"]
        for row in distribution_rows
        if row["imbalance_tier"] != "unlabeled"
    )
    tier_records = {
        tier: sum(
            row["record_count"]
            for row in distribution_rows
            if row["imbalance_tier"] == tier
        )
        for tier in ("dominant", "moderate", "rare", "unlabeled")
    }
    report_lines = [
        "# Watkins dataset imbalance and quality summary",
        "",
        f"- Parquet shards: {len(files)}",
        f"- Records scanned: {quality['record_count']:,}",
        f"- Inferred species label field: `{schema.label or 'not detected'}`",
        f"- Inferred audio field: `{schema.audio or 'not detected'}`",
        f"- Audio representation types: {dict(quality['audio_kind'])}",
        f"- Records with decoded duration: {len(all_durations):,}",
        f"- Total decoded audio duration: {sum(all_durations) / 3600:.2f} hours",
        f"- Missing labels: {quality['missing_label']:,}; missing primary labels with fallback: {quality['missing_primary_label']:,}",
        f"- Missing audio: {quality['audio_missing']:,}; empty payloads: {quality['audio_empty']:,}; corrupt payloads: {quality['audio_corrupt']:,}",
        f"- Duplicate audio payloads: {quality['duplicate_audio_payloads']:,}; repeated record IDs: {quality['duplicate_record_ids']:,}",
        f"- Audio sample-rate metadata differs from the decoded file header in {quality['sample_rate_mismatches']:,} records.",
        "",
        "## Imbalance tiers",
        "",
        "Labeled species are divided into count-based tertiles: rare at or below the 33rd percentile, moderate above that through the 67th percentile, and dominant above the 67th percentile. Thresholds are dataset-derived, not biological definitions.",
        "",
        f"- Rare: {tier_species['rare']} species, {tier_records['rare']:,} records",
        f"- Moderate: {tier_species['moderate']} species, {tier_records['moderate']:,} records",
        f"- Dominant: {tier_species['dominant']} species, {tier_records['dominant']:,} records",
        f"- Unlabeled display-name groups: {tier_totals['unlabeled']} groups, {tier_records['unlabeled']:,} records",
        "",
        "## Recommended training strategy",
        "",
        "Keep all original records and their natural distribution for the initial self-supervised baseline. Do not use species labels as prediction targets, synthesize audio with SMOTE, or delete rare/common-species records. The optional `SpeciesBalancedBatchSampler` creates batches with equal samples per species, shuffles each species' rows, and never repeats a row within an epoch; singleton/rare rows are therefore not oversampled. Configure one sample per species per batch (`batch_size == classes_per_batch`) to include singleton taxa. If more examples per species are requested, species with fewer records are ineligible for balanced batches. Some leftovers may not fit a complete balanced batch and remain available to the natural-distribution loader.",
        "",
        "For validation, split by a verified recording session, location, or recording source using `grouped_split_indices`. Missing group metadata remains in training rather than being randomly split. Do not treat clip IDs as sessions unless their semantics are verified.",
        "",
        "Evaluate clusters separately by common/moderate/rare tier. Species-cluster association is only a taxonomic diagnostic, not validation of call types. Include recording-domain association where those labels exist; inspect representative audio and report limitations.",
        "",
        "## Generated outputs",
        "",
        "- `species_distribution.csv`: per-label counts, tiers, durations, and quality/metadata coverage.",
        "- `dataset_quality_report.csv`: dataset-level schema, missing/corrupt/duplicate audio, and metadata summary.",
        *[f"- `{chart}`" for chart in charts],
        "",
        "Original Parquet shards were read only and were not modified.",
        "",
    ]
    (output_dir / "imbalance_strategy_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return {
        "schema": schema,
        "record_count": quality["record_count"],
        "species_count": len(species),
        "quality": quality,
        "distribution": distribution_rows,
        "tiers": tiers,
        "output_dir": output_dir,
        "charts": charts,
    }
