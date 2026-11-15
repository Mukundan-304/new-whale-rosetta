"""Create leakage-safe recording- and location-grouped Watkins data splits."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import unicodedata
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SEGMENT_MANIFEST = (
    PROJECT_DIR / "watkins_preprocessed" / "segment_metadata.csv"
)
DEFAULT_SOURCE_MANIFEST = (
    PROJECT_DIR / "watkins_baseline" / "watkins_location_split.csv"
)
DEFAULT_SOURCE_DIR = (
    PROJECT_DIR / "watkins-marine-mammal-full-cuts" / "data"
)
DEFAULT_FEATURE_DIR = PROJECT_DIR / "watkins_preprocessed" / "features"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "watkins_splits"
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
DEFAULT_FOLD_COUNT = 20
MISSING_LABEL = "[MISSING_LABEL]"


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV manifest does not exist: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV manifest has no header: {path}")
        return list(reader), list(reader.fieldnames)


def _row_key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("parquet_file", "").strip(), row.get(
        "parquet_row_index", ""
    ).strip()


def _require_columns(
    fieldnames: Sequence[str], required: set[str], path: Path
) -> None:
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(sorted(missing))}"
        )


def _location_domain(
    row: dict[str, str], recording_id: str
) -> tuple[str, str, str]:
    names = sorted(
        {
            " ".join(
                unicodedata.normalize("NFKC", value).casefold().split()
            )
            for value in row.get("location_names", "").split(" | ")
            if value.strip()
        }
    )
    if names:
        value = " | ".join(names)
        return f"name:{value}", "name", value

    try:
        coordinates = json.loads(row.get("location_coordinates") or "[]")
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid location coordinates for recording {recording_id}"
        ) from error
    points = sorted(
        {
            (round(float(point["lat"]), 4), round(float(point["lon"]), 4))
            for point in coordinates
            if point.get("lat") is not None and point.get("lon") is not None
        }
    )
    if points:
        value = json.dumps(points, separators=(",", ":"))
        return f"coordinates:{value}", "coordinates", value
    return f"missing:{recording_id}", "missing", ""


def _assign_location_groups(recordings: list[dict[str, Any]]) -> None:
    parents = list(range(len(recordings)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    owners: dict[tuple[str, str], int] = {}
    for index, record in enumerate(recordings):
        for entity in (
            ("location", record["location_domain_key"]),
            ("audio", record["source_audio_sha256"]),
        ):
            previous = owners.get(entity)
            if previous is None:
                owners[entity] = index
            else:
                union(index, previous)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(recordings)):
        components[find(index)].append(index)
    ordered_components = sorted(
        components.values(),
        key=lambda indices: min(
            (
                recordings[index]["location_domain_key"],
                recordings[index]["recording_id"],
            )
            for index in indices
        ),
    )
    for group_number, indices in enumerate(ordered_components, start=1):
        group_id = f"location-group-{group_number:04d}"
        for index in indices:
            record = recordings[index]
            record["location_group"] = group_id
            for segment in record["segment_rows"]:
                segment["location_group"] = group_id


def _build_recordings(
    segment_rows: list[dict[str, str]],
    segment_fields: list[str],
    source_rows: list[dict[str, str]],
    source_fields: list[str],
    source_dir: Path,
    feature_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    _require_columns(
        segment_fields,
        {
            "segment_id",
            "recording_id",
            "species_label",
            "species_code",
            "parquet_file",
            "parquet_row_index",
            "feature_partition",
            "feature_index",
            "location_names",
            "location_coordinates",
            "observation_date",
            "mid",
        },
        DEFAULT_SEGMENT_MANIFEST,
    )
    _require_columns(
        source_fields,
        {
            "record_number",
            "parquet_file",
            "parquet_row_index",
            "audio_sha256",
            "location_group",
            "location_metadata_available",
        },
        DEFAULT_SOURCE_MANIFEST,
    )

    source_by_key: dict[tuple[str, str], dict[str, str]] = {}
    source_key_by_recording: dict[str, tuple[str, str]] = {}
    for row in source_rows:
        key = _row_key(row)
        recording_id = row["record_number"].strip()
        if not all(key) or not recording_id:
            raise ValueError("Source manifest contains a missing source key or ID")
        if key in source_by_key:
            raise ValueError(f"Duplicate source row in identity manifest: {key}")
        if recording_id in source_key_by_recording:
            raise ValueError(
                f"Duplicate recording ID in identity manifest: {recording_id}"
            )
        digest = row["audio_sha256"].strip().lower()
        if len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise ValueError(
                f"Invalid or missing SHA-256 for recording {recording_id}"
            )
        source_by_key[key] = row
        source_key_by_recording[recording_id] = key

    recordings: dict[str, dict[str, Any]] = {}
    segment_ids: set[str] = set()
    feature_references: set[tuple[str, str]] = set()
    for row in segment_rows:
        segment_id = row.get("segment_id", "").strip()
        recording_id = row.get("recording_id", "").strip()
        key = _row_key(row)
        if not segment_id:
            raise ValueError("Segment manifest contains a missing segment_id")
        if segment_id in segment_ids:
            raise ValueError(f"Duplicate segment_id: {segment_id}")
        segment_ids.add(segment_id)
        if not recording_id:
            raise ValueError(f"Segment {segment_id} has no recording_id")
        if not all(key):
            raise ValueError(f"Segment {segment_id} has an incomplete source key")
        source = source_by_key.get(key)
        if source is None:
            raise ValueError(
                f"Segment {segment_id} has no matching source identity row: {key}"
            )
        if source["record_number"].strip() != recording_id:
            raise ValueError(
                f"Recording ID mismatch for segment {segment_id}: "
                f"{recording_id} != {source['record_number'].strip()}"
            )

        partition = row["feature_partition"].strip()
        try:
            feature_index = int(row["feature_index"])
        except ValueError as error:
            raise ValueError(
                f"Invalid feature index for segment {segment_id}"
            ) from error
        if not partition or Path(partition).name != partition or feature_index < 0:
            raise ValueError(
                f"Invalid feature reference for segment {segment_id}"
            )
        feature_reference = (partition, str(feature_index))
        if feature_reference in feature_references:
            raise ValueError(
                f"Duplicate feature reference for segment {segment_id}: "
                f"{feature_reference}"
            )
        feature_references.add(feature_reference)

        previous_key = source_key_by_recording.get(recording_id)
        if previous_key != key:
            raise ValueError(
                f"Recording ID {recording_id} maps to multiple source rows"
            )

        entry = recordings.get(recording_id)
        if entry is None:
            domain_key, domain_type, domain_value = _location_domain(
                row, recording_id
            )
            entry = {
                "recording_id": recording_id,
                "species_label": row.get("species_label", "").strip()
                or row.get("species_code", "").strip()
                or MISSING_LABEL,
                "source_audio_sha256": source["audio_sha256"].strip().lower(),
                "baseline_location_component": source[
                    "location_group"
                ].strip(),
                "location_domain_key": domain_key,
                "location_domain_type": domain_type,
                "location_domain": domain_value,
                "location_metadata_available": (
                    source["location_metadata_available"].strip().lower()
                    == "true"
                ),
                "parquet_file": key[0],
                "parquet_row_index": int(key[1]),
                "audio_path": str((source_dir / key[0]).resolve()),
                "audio_storage_type": "audio_embedded_in_parquet",
                "feature_path": str(
                    (feature_dir / partition).resolve()
                ),
                "segments": [],
                "segment_rows": [],
            }
            recordings[recording_id] = entry
        elif entry["species_label"] != (
            row.get("species_label", "").strip()
            or row.get("species_code", "").strip()
            or MISSING_LABEL
        ):
            raise ValueError(
                f"Species label changes between segments of {recording_id}"
            )

        segment_copy = dict(row)
        segment_copy.update(
            {
                "species": entry["species_label"],
                "audio_path": entry["audio_path"],
                "audio_storage_type": entry["audio_storage_type"],
                "audio_parquet_row_index": str(entry["parquet_row_index"]),
                "source_audio_sha256": entry["source_audio_sha256"],
                "feature_path": entry["feature_path"],
                "spectrogram_path": "",
                "location_group": "",
                "baseline_location_component": entry[
                    "baseline_location_component"
                ],
                "location_domain_key": entry["location_domain_key"],
                "location_domain_type": entry["location_domain_type"],
                "location_domain": entry["location_domain"],
                "location_metadata_available": str(
                    entry["location_metadata_available"]
                ),
                "sound_channel_source_code": "",
                "sound_channel_recorded": "",
                "sound_channel_multiplexed": "",
                "sound_channel_side": "",
                "signal_source_code": "",
            }
        )
        try:
            sound = json.loads(row.get("sound_metadata_json") or "{}")
            signal = json.loads(row.get("signal_metadata_json") or "{}")
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid nested metadata JSON for segment {segment_id}"
            ) from error
        channel = sound.get("channel") or {}
        segment_copy.update(
            {
                "sound_channel_source_code": channel.get("_source_nc", ""),
                "sound_channel_recorded": channel.get("recorded", ""),
                "sound_channel_multiplexed": channel.get("multiplexed", ""),
                "sound_channel_side": channel.get("side", ""),
                "signal_source_code": signal.get("_source_sc", ""),
            }
        )
        entry["segments"].append(segment_id)
        entry["segment_rows"].append(segment_copy)

    unmatched_source_rows = set(source_by_key) - {
        _row_key(row) for row in segment_rows
    }
    if unmatched_source_rows:
        raise ValueError(
            f"{len(unmatched_source_rows)} source recordings have no segments"
        )
    recording_list = list(recordings.values())
    _assign_location_groups(recording_list)
    return recording_list, segment_fields


def _candidate_fold_counts(
    ratio: float, fold_count: int, maximum: int
) -> range:
    center = max(1, round(ratio * fold_count))
    low = max(1, center - 1)
    high = min(maximum, center + 1)
    return range(low, high + 1)


def _fold_subset_loss(
    selected: Sequence[int],
    recording_counts: np.ndarray,
    segment_counts: np.ndarray,
    species_recording_counts: np.ndarray,
    species_segment_counts: np.ndarray,
    total_recordings: int,
    total_segments: int,
    species_recording_totals: np.ndarray,
    species_segment_totals: np.ndarray,
    species_independent_totals: np.ndarray,
    target_ratio: float,
) -> float:
    selected_recordings = float(recording_counts[list(selected)].sum())
    selected_segments = float(segment_counts[list(selected)].sum())
    recording_target = total_recordings * target_ratio
    segment_target = total_segments * target_ratio
    global_loss = 0.5 * (
        ((selected_recordings - recording_target) / max(recording_target, 1)) ** 2
        + ((selected_segments - segment_target) / max(segment_target, 1)) ** 2
    )

    selected_species_recordings = species_recording_counts[
        list(selected)
    ].sum(axis=0)
    selected_species_segments = species_segment_counts[list(selected)].sum(
        axis=0
    )
    species_recording_targets = species_recording_totals * target_ratio
    species_segment_targets = species_segment_totals * target_ratio
    can_cover_each_split = species_independent_totals >= len(SPLIT_NAMES)
    species_recording_targets[can_cover_each_split] = np.maximum(
        species_recording_targets[can_cover_each_split], 1.0
    )
    species_segment_targets[can_cover_each_split] = np.maximum(
        species_segment_targets[can_cover_each_split], 1.0
    )
    recording_errors = (
        selected_species_recordings - species_recording_targets
    ) / np.maximum(species_recording_targets, 1.0)
    segment_errors = (
        selected_species_segments - species_segment_targets
    ) / np.maximum(species_segment_targets, 1.0)
    species_loss = float(
        0.5
        * (
            np.mean(np.square(recording_errors))
            + np.mean(np.square(segment_errors))
        )
    )
    coverage_candidates = species_independent_totals >= len(SPLIT_NAMES)
    if np.any(coverage_candidates):
        missing_coverage = np.mean(
            selected_species_recordings[coverage_candidates] == 0
        )
    else:
        missing_coverage = 0.0
    return global_loss + species_loss + float(missing_coverage)


def _select_fold_subset(
    candidates: Sequence[int],
    subset_sizes: range,
    target_ratio: float,
    recording_counts: np.ndarray,
    segment_counts: np.ndarray,
    species_recording_counts: np.ndarray,
    species_segment_counts: np.ndarray,
    total_recordings: int,
    total_segments: int,
    species_recording_totals: np.ndarray,
    species_segment_totals: np.ndarray,
    species_independent_totals: np.ndarray,
) -> tuple[int, ...]:
    best: tuple[float, tuple[int, ...]] | None = None
    for subset_size in subset_sizes:
        if subset_size > len(candidates):
            continue
        for selected in itertools.combinations(candidates, subset_size):
            loss = _fold_subset_loss(
                selected,
                recording_counts,
                segment_counts,
                species_recording_counts,
                species_segment_counts,
                total_recordings,
                total_segments,
                species_recording_totals,
                species_segment_totals,
                species_independent_totals,
                target_ratio,
            )
            candidate = (loss, selected)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("Could not select non-empty grouped split folds")
    return best[1]


def assign_grouped_splits(
    recordings: list[dict[str, Any]],
    group_field: str,
    seed: int = 29,
    fold_count: int = DEFAULT_FOLD_COUNT,
    ratios: dict[str, float] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Assign recordings to approximate target shares without splitting groups."""
    split_ratios = ratios or SPLIT_RATIOS
    if set(split_ratios) != set(SPLIT_NAMES) or not math.isclose(
        sum(split_ratios.values()), 1.0
    ):
        raise ValueError("Split ratios must define train/validation/test and sum to 1")
    if not recordings:
        raise ValueError("Cannot split an empty recording manifest")

    labels = [str(row["species_label"]) for row in recordings]
    groups = [str(row[group_field]) for row in recordings]
    if any(not group for group in groups):
        raise ValueError(f"Missing group values for {group_field}")
    unique_group_count = len(set(groups))
    actual_fold_count = min(fold_count, unique_group_count)
    if actual_fold_count < 3:
        raise ValueError(
            f"At least three independent groups are required; found "
            f"{unique_group_count} for {group_field}"
        )
    segment_counts_by_record = np.asarray(
        [len(row["segments"]) for row in recordings], dtype=np.int64
    )
    species = sorted(set(labels))
    species_index = {label: index for index, label in enumerate(species)}
    warnings_seen: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        folds = list(
            StratifiedGroupKFold(
                n_splits=actual_fold_count,
                shuffle=True,
                random_state=seed,
            ).split(
                np.zeros((len(recordings), 1), dtype=np.float32),
                labels,
                groups,
            )
        )
    warnings_seen = list(dict.fromkeys(str(item.message) for item in caught))

    fold_for_recording = np.full(len(recordings), -1, dtype=np.int32)
    for fold_index, (_, validation_indices) in enumerate(folds):
        fold_for_recording[validation_indices] = fold_index
    if np.any(fold_for_recording < 0):
        raise RuntimeError("Stratified group fold assignment omitted recordings")

    recording_counts = np.bincount(
        fold_for_recording, minlength=actual_fold_count
    ).astype(np.int64)
    segment_counts = np.bincount(
        fold_for_recording,
        weights=segment_counts_by_record,
        minlength=actual_fold_count,
    ).astype(np.int64)
    species_recording_counts = np.zeros(
        (actual_fold_count, len(species)), dtype=np.int64
    )
    species_segment_counts = np.zeros_like(species_recording_counts)
    for record_index, fold_index in enumerate(fold_for_recording):
        label_index = species_index[labels[record_index]]
        species_recording_counts[fold_index, label_index] += 1
        species_segment_counts[fold_index, label_index] += (
            segment_counts_by_record[record_index]
        )
    species_recording_totals = species_recording_counts.sum(axis=0)
    species_segment_totals = species_segment_counts.sum(axis=0)
    species_independent_totals = np.asarray(
        [
            len(
                {
                        group
                        for row, group in zip(recordings, groups)
                    if row["species_label"] == label
                }
            )
            for label in species
        ],
        dtype=np.int64,
    )
    all_folds = list(range(actual_fold_count))

    test_folds = _select_fold_subset(
        all_folds,
        _candidate_fold_counts(
            split_ratios["test"], actual_fold_count, actual_fold_count - 2
        ),
        split_ratios["test"],
        recording_counts,
        segment_counts,
        species_recording_counts,
        species_segment_counts,
        len(recordings),
        int(segment_counts_by_record.sum()),
        species_recording_totals,
        species_segment_totals,
        species_independent_totals,
    )
    remaining_folds = [fold for fold in all_folds if fold not in test_folds]
    validation_folds = _select_fold_subset(
        remaining_folds,
        _candidate_fold_counts(
            split_ratios["validation"],
            actual_fold_count,
            len(remaining_folds) - 1,
        ),
        split_ratios["validation"],
        recording_counts,
        segment_counts,
        species_recording_counts,
        species_segment_counts,
        len(recordings),
        int(segment_counts_by_record.sum()),
        species_recording_totals,
        species_segment_totals,
        species_independent_totals,
    )

    assignments: dict[str, str] = {}
    for record_index, record in enumerate(recordings):
        fold = int(fold_for_recording[record_index])
        if fold in test_folds:
            split = "test"
        elif fold in validation_folds:
            split = "validation"
        else:
            split = "train"
        recording_id = str(record["recording_id"])
        if recording_id in assignments:
            raise ValueError(f"Duplicate recording ID while splitting: {recording_id}")
        assignments[recording_id] = split

    split_by_group: dict[str, str] = {}
    for record in recordings:
        split = assignments[str(record["recording_id"])]
        group = str(record[group_field])
        previous = split_by_group.setdefault(group, split)
        if previous != split:
            raise RuntimeError(
                f"Group leakage detected for {group_field}={group}"
            )
    return assignments, warnings_seen


def _record_splits(
    recordings: list[dict[str, Any]], assignments: dict[str, str]
) -> dict[str, list[dict[str, Any]]]:
    splits = {name: [] for name in SPLIT_NAMES}
    duplicate_counts = Counter(
        item["source_audio_sha256"] for item in recordings
    )
    for record in recordings:
        record["split"] = assignments[record["recording_id"]]
        record["duplicate_audio_payload"] = (
            duplicate_counts[record["source_audio_sha256"]] > 1
        )
        for segment in record["segment_rows"]:
            segment["split"] = record["split"]
            segment["duplicate_audio_payload"] = str(
                record["duplicate_audio_payload"]
            )
            splits[record["split"]].append(segment)
    return splits


def validate_split_rows(
    split_rows: dict[str, list[dict[str, Any]]],
    group_field: str,
    inspect_parquet_rows: bool = False,
) -> None:
    group_splits: dict[str, set[str]] = defaultdict(set)
    recording_splits: dict[str, set[str]] = defaultdict(set)
    segment_splits: dict[str, set[str]] = defaultdict(set)
    source_hash_splits: dict[str, set[str]] = defaultdict(set)
    feature_reference_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    segment_ids: set[str] = set()
    feature_rows: dict[str, set[int]] = defaultdict(set)
    parquet_rows: dict[str, set[int]] = defaultdict(set)

    for split, rows in split_rows.items():
        for row in rows:
            segment_id = row.get("segment_id", "").strip()
            recording_id = row.get("recording_id", "").strip()
            group = row.get(group_field, "").strip()
            digest = row.get("source_audio_sha256", "").strip()
            if not segment_id or not recording_id or not group or not digest:
                raise ValueError(
                    f"Split {split} contains a missing ID, group, or source hash"
                )
            if segment_id in segment_ids:
                raise ValueError(f"Duplicate segment_id across splits: {segment_id}")
            segment_ids.add(segment_id)
            group_splits[group].add(split)
            recording_splits[recording_id].add(split)
            segment_splits[segment_id].add(split)
            source_hash_splits[digest].add(split)
            feature_reference_splits[
                (row.get("feature_path", ""), row.get("feature_index", ""))
            ].add(split)

            for path_field in ("audio_path", "feature_path"):
                path = Path(row.get(path_field, ""))
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing {path_field} for segment {segment_id}: {path}"
                    )
            try:
                parquet_index = int(row["audio_parquet_row_index"])
                feature_index = int(row["feature_index"])
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"Invalid file index for segment {segment_id}"
                ) from error
            if parquet_index < 0 or feature_index < 0:
                raise ValueError(f"Negative file index for segment {segment_id}")
            parquet_rows[row["audio_path"]].add(parquet_index)
            feature_rows[row["feature_path"]].add(feature_index)

    for name, mapping in (
        (group_field, group_splits),
        ("recording_id", recording_splits),
        ("segment_id", segment_splits),
        ("source_audio_sha256", source_hash_splits),
        ("feature reference", feature_reference_splits),
    ):
        leaked = [value for value, split_set in mapping.items() if len(split_set) > 1]
        if leaked:
            raise ValueError(
                f"Leakage detected: {len(leaked)} {name} values occur in "
                "multiple splits"
            )

    if inspect_parquet_rows:
        for path, indices in parquet_rows.items():
            row_count = pq.ParquetFile(path).metadata.num_rows
            invalid = [index for index in indices if index >= row_count]
            if invalid:
                raise ValueError(
                    f"Parquet row index {min(invalid)} is out of range for {path}"
                )

    for path in feature_rows:
        try:
            with np.load(path) as stored:
                if "log_mel" not in stored:
                    raise ValueError(f"Feature file lacks log_mel array: {path}")
                feature_count = int(stored["log_mel"].shape[0])
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(f"Cannot read cached features at {path}: {error}") from error
        invalid = [index for index in feature_rows[path] if index >= feature_count]
        if invalid:
            raise ValueError(
                f"Feature index {min(invalid)} is out of range for {path}"
            )


def _distribution_rows(
    split_rows: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split, rows in split_rows.items():
        species_data: dict[str, dict[str, Any]] = {}
        for row in rows:
            label = row["species"]
            info = species_data.setdefault(
                label,
                {
                    "species_code": row.get("species_code", ""),
                    "segments": 0,
                    "recording_ids": set(),
                },
            )
            info["segments"] += 1
            info["recording_ids"].add(row["recording_id"])
        for label, info in sorted(species_data.items()):
            output.append(
                {
                    "split": split,
                    "species": label,
                    "species_code": info["species_code"],
                    "segments": info["segments"],
                    "recordings": len(info["recording_ids"]),
                }
            )
    return output


def _split_counts(
    recordings: list[dict[str, Any]],
    split_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    total_recordings = len(recordings)
    total_segments = sum(len(row["segments"]) for row in recordings)
    result: dict[str, dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        recording_count = sum(
            1 for row in recordings if row["split"] == split
        )
        segment_count = len(split_rows[split])
        result[split] = {
            "recordings": recording_count,
            "recording_percent": (
                100.0 * recording_count / total_recordings
                if total_recordings
                else 0.0
            ),
            "segments": segment_count,
            "segment_percent": (
                100.0 * segment_count / total_segments if total_segments else 0.0
            ),
            "unique_species_labels": len(
                {row["species"] for row in split_rows[split]}
            ),
            "known_location_recordings": sum(
                1
                for row in recordings
                if row["split"] == split
                and row["location_metadata_available"]
            ),
            "known_location_domains": len(
                {
                    row["location_domain_key"]
                    for row in recordings
                    if row["split"] == split
                    and row["location_domain_type"] != "missing"
                }
            ),
        }
    return result


def _species_coverage(
    recordings: list[dict[str, Any]],
    assignments: dict[str, str],
) -> list[dict[str, Any]]:
    by_species: dict[str, dict[str, Any]] = {}
    for record in recordings:
        label = record["species_label"]
        row = by_species.setdefault(
            label,
            {
                "recording_ids": set(),
                "independent_audio_hashes": set(),
                "splits": set(),
            },
        )
        row["recording_ids"].add(record["recording_id"])
        row["independent_audio_hashes"].add(record["source_audio_sha256"])
        row["splits"].add(assignments[record["recording_id"]])
    return [
        {
            "species": label,
            "recordings": len(info["recording_ids"]),
            "independent_audio_groups": len(info["independent_audio_hashes"]),
            "splits_present": sorted(info["splits"]),
            "splits_absent": [
                split for split in SPLIT_NAMES if split not in info["splits"]
            ],
            "rare_independent_recording_count": (
                len(info["independent_audio_hashes"]) < len(SPLIT_NAMES)
            ),
        }
        for label, info in sorted(by_species.items())
    ]


def _missingness_rows(
    recordings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "species_code",
        "observation_date",
        "location_names",
        "location_coordinates",
        "mid",
        "sound_metadata_json",
        "signal_metadata_json",
    )
    output = []
    for split in SPLIT_NAMES:
        split_records = [row for row in recordings if row["split"] == split]
        for field in fields:
            missing = sum(
                not str(row["segment_rows"][0].get(field, "")).strip()
                for row in split_records
            )
            count = len(split_records)
            output.append(
                {
                    "split": split,
                    "field": field,
                    "recordings": count,
                    "recordings_missing": missing,
                    "missing_percent": 100.0 * missing / count if count else 0.0,
                }
            )
    return output


def _write_csv(path: Path, fieldnames: Sequence[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_split_files(
    split_dir: Path,
    split_rows: dict[str, list[dict[str, Any]]],
    segment_fields: list[str],
) -> None:
    extra_fields = [
        "species",
        "audio_path",
        "audio_storage_type",
        "audio_parquet_row_index",
        "source_audio_sha256",
        "duplicate_audio_payload",
        "feature_path",
        "spectrogram_path",
        "location_group",
        "baseline_location_component",
        "location_domain_key",
        "location_domain_type",
        "location_domain",
        "location_metadata_available",
        "sound_channel_source_code",
        "sound_channel_recorded",
        "sound_channel_multiplexed",
        "sound_channel_side",
        "signal_source_code",
        "split",
    ]
    fields = list(dict.fromkeys([*segment_fields, *extra_fields]))
    for split, rows in split_rows.items():
        _write_csv(split_dir / f"{split}.csv", fields, rows)


def _write_evaluation_outputs(
    output_dir: Path,
    recordings: list[dict[str, Any]],
    split_rows: dict[str, list[dict[str, Any]]],
    assignments: dict[str, str],
    strategy: str,
    group_field: str,
    stratification_warnings: list[str],
    segment_fields: list[str],
    seed: int,
) -> dict[str, Any]:
    counts = _split_counts(recordings, split_rows)
    coverage = _species_coverage(recordings, assignments)
    missingness = _missingness_rows(recordings)
    _write_split_files(output_dir, split_rows, segment_fields)
    _write_csv(
        output_dir / "species_distribution.csv",
        ("split", "species", "species_code", "segments", "recordings"),
        _distribution_rows(split_rows),
    )
    _write_csv(
        output_dir / "metadata_missingness.csv",
        (
            "split",
            "field",
            "recordings",
            "recordings_missing",
            "missing_percent",
        ),
        missingness,
    )
    summary = {
        "strategy": strategy,
        "group_field": group_field,
        "seed": seed,
        "requested_split_percentages": {
            split: ratio * 100 for split, ratio in SPLIT_RATIOS.items()
        },
        "split_counts": counts,
        "species_label_count": len(coverage),
        "species_with_fewer_than_three_independent_recordings": sum(
            row["rare_independent_recording_count"] for row in coverage
        ),
        "species_coverage": coverage,
        "stratification_warnings": stratification_warnings,
    }
    with (output_dir / "split_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return summary


def _write_report(
    output_dir: Path,
    standard_summary: dict[str, Any],
    environment_summary: dict[str, Any],
    recordings: list[dict[str, Any]],
    source_manifest_path: Path,
) -> None:
    known_locations = {
        row["location_domain"]
        for row in recordings
        if row["location_domain_type"] == "name"
    }
    coordinate_domains = {
        row["location_domain"]
        for row in recordings
        if row["location_domain_type"] == "coordinates"
    }
    known_recordings = sum(
        row["location_metadata_available"] for row in recordings
    )
    total_recordings = len(recordings)
    environment_test = environment_summary["split_counts"]["test"]
    environment_validation = environment_summary["split_counts"]["validation"]
    report = [
        "# Watkins dataset split report",
        "",
        "## Standard recording-grouped split",
        "",
        "- All segments from an original recording remain together.",
        "- Exact duplicate source-audio payloads are grouped together using "
        "the verified SHA-256 values in "
        f"`{source_manifest_path}`.",
        "- StratifiedGroupKFold is used to form deterministic folds; whole "
        "folds are selected to best match the requested segment and "
        "recording proportions and species distributions.",
        "- Rare species are not physically oversampled or undersampled. "
        "Species with fewer than three independent audio groups cannot "
        "appear in all three splits.",
        "",
        "| Split | Recordings | Recording % | Segments | Segment % |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLIT_NAMES:
        stats = standard_summary["split_counts"][split]
        report.append(
            f"| {split} | {stats['recordings']:,} | "
            f"{stats['recording_percent']:.2f}% | {stats['segments']:,} | "
            f"{stats['segment_percent']:.2f}% |"
        )
    report.extend(
        [
            "",
            "The exact 70/15/15 ratios are targets. Group sizes, duplicate "
            "audio constraints, and species stratification can cause small "
            "deviations; actual percentages are reported above.",
            "",
            "## Separate location-held-out evaluation split",
            "",
            "- Every exact normalized location-name group stays in one split. "
            "Coordinates are used only when a name is absent; exact duplicate "
            "audio is tied across locations to prevent content leakage.",
            f"- Location metadata is available for {known_recordings:,} of "
            f"{total_recordings:,} recordings "
            f"({100 * known_recordings / total_recordings:.2f}%). "
            f"There are {len(known_locations):,} distinct normalized names "
            f"and {len(coordinate_domains):,} coordinate-only domains.",
            "- This can test generalization to held-out *locations*, but "
            "the dataset has no explicit recording-session or recorder/device "
            "identifier. Location is only an environment proxy; this split "
            "does not establish generalization across devices or sessions.",
            "- For strict location-held-out evaluation, score only examples "
            "with known location metadata in the environment test split. "
            "Rows without location metadata remain available but cannot "
            "substantiate a held-out-location claim.",
            f"- The environment test contains "
            f"{environment_test['known_location_recordings']:,} recordings "
            f"with known location. Its validation and test splits contain "
            f"{environment_validation['unique_species_labels']} and "
            f"{environment_test['unique_species_labels']} of "
            f"{environment_summary['species_label_count']} species labels. "
            "Location groups have highly unequal sizes and are correlated "
            "with species, so these environment-held-out percentages and "
            "species coverage are less balanced than the standard split. "
            "Treat this as a secondary generalization check, not as the main "
            "training/validation partition.",
            "",
            "| Split | Recordings | Recording % | Segments | Segment % |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in SPLIT_NAMES:
        stats = environment_summary["split_counts"][split]
        report.append(
            f"| {split} | {stats['recordings']:,} | "
            f"{stats['recording_percent']:.2f}% | {stats['segments']:,} | "
            f"{stats['segment_percent']:.2f}% |"
        )
    report.extend(
        [
            "",
            "## Paths and metadata",
            "",
            "- `audio_path` points to the source Parquet shard because the "
            "original audio is embedded in that file; "
            "`audio_parquet_row_index` identifies the record. It is not a "
            "standalone WAV path.",
            "- `feature_path` and `feature_index` point to each segment's "
            "cached log-Mel array. `spectrogram_path` is blank because "
            "individual spectrogram image files are not stored for every "
            "segment.",
            "- Species balancing is not applied to validation or test. "
            "Use the existing balanced batch sampler for training only.",
            "- Per-split species counts are in `species_distribution.csv`; "
            "missing metadata is in `metadata_missingness.csv`; full "
            "assignments and rare-species coverage are in each "
            "`split_summary.json`.",
        ]
    )
    path = output_dir / "split_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def create_splits(
    segment_manifest: Path = DEFAULT_SEGMENT_MANIFEST,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 29,
) -> dict[str, Any]:
    segment_rows, segment_fields = _read_csv(segment_manifest)
    source_rows, source_fields = _read_csv(source_manifest)
    _require_columns(
        segment_fields,
        {
            "segment_id",
            "recording_id",
            "species_label",
            "species_code",
            "parquet_file",
            "parquet_row_index",
            "feature_partition",
            "feature_index",
            "location_names",
            "location_coordinates",
            "observation_date",
            "mid",
        },
        segment_manifest,
    )
    _require_columns(
        source_fields,
        {
            "record_number",
            "parquet_file",
            "parquet_row_index",
            "audio_sha256",
            "location_group",
            "location_metadata_available",
        },
        source_manifest,
    )
    recordings, segment_fields = _build_recordings(
        segment_rows,
        segment_fields,
        source_rows,
        source_fields,
        source_dir,
        feature_dir,
    )
    if not recordings:
        raise ValueError("Segment manifest contains no recordings")
    if len(recordings) != len(source_rows):
        raise ValueError(
            f"Recording count mismatch: {len(recordings)} in segment manifest, "
            f"{len(source_rows)} in source manifest"
        )

    standard_assignments, standard_warnings = assign_grouped_splits(
        recordings,
        "source_audio_sha256",
        seed=seed,
    )
    standard_recordings = [
        dict(row, segments=list(row["segments"]), segment_rows=[
            dict(segment) for segment in row["segment_rows"]
        ])
        for row in recordings
    ]
    standard_rows = _record_splits(standard_recordings, standard_assignments)
    validate_split_rows(
        standard_rows,
        "source_audio_sha256",
        inspect_parquet_rows=True,
    )
    standard_summary = _write_evaluation_outputs(
        output_dir,
        standard_recordings,
        standard_rows,
        standard_assignments,
        (
            "StratifiedGroupKFold with exact duplicate audio grouped by "
            "source SHA-256"
        ),
        "source_audio_sha256",
        standard_warnings,
        segment_fields,
        seed,
    )

    environment_assignments, environment_warnings = assign_grouped_splits(
        recordings,
        "location_group",
        seed=seed,
    )
    environment_recordings = [
        dict(row, segments=list(row["segments"]), segment_rows=[
            dict(segment) for segment in row["segment_rows"]
        ])
        for row in recordings
    ]
    environment_dir = output_dir / "environment_holdout"
    environment_rows = _record_splits(
        environment_recordings, environment_assignments
    )
    validate_split_rows(
        environment_rows,
        "location_group",
        inspect_parquet_rows=True,
    )
    environment_summary = _write_evaluation_outputs(
        environment_dir,
        environment_recordings,
        environment_rows,
        environment_assignments,
        (
            "StratifiedGroupKFold with direct location domains and exact "
            "duplicate audio kept together"
        ),
        "location_group",
        environment_warnings,
        segment_fields,
        seed,
    )
    _write_report(
        output_dir,
        standard_summary,
        environment_summary,
        recordings,
        source_manifest,
    )
    return {
        "standard": standard_summary,
        "environment_holdout": environment_summary,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-manifest", type=Path, default=DEFAULT_SEGMENT_MANIFEST)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=29)
    arguments = parser.parse_args()
    result = create_splits(
        segment_manifest=arguments.segment_manifest,
        source_manifest=arguments.source_manifest,
        source_dir=arguments.source_dir,
        feature_dir=arguments.feature_dir,
        output_dir=arguments.output_dir,
        seed=arguments.seed,
    )
    print(f"Split manifests and reports written to {result['output_dir']}")
    for evaluation in ("standard", "environment_holdout"):
        print(f"{evaluation}:")
        for split, stats in result[evaluation]["split_counts"].items():
            print(
                f"  {split}: {stats['recordings']:,} recordings "
                f"({stats['recording_percent']:.2f}%), "
                f"{stats['segments']:,} segments "
                f"({stats['segment_percent']:.2f}%)"
            )


if __name__ == "__main__":
    main()
