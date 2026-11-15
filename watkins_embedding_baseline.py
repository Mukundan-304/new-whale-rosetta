"""Build a leakage-aware frozen-audio baseline for the Watkins dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import unicodedata
from collections import Counter, defaultdict
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from scipy.signal import resample_poly, welch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "watkins-marine-mammal-full-cuts" / "data"
AUDIT_PATH = PROJECT_DIR / "species_distribution.csv"
OUTPUT_DIR = PROJECT_DIR / "watkins_baseline"
MODEL_ID = "laion/clap-htsat-unfused"
TARGET_SAMPLE_RATE = 48_000
CLIP_SECONDS = 10
FULL_BAND_FEATURE_COUNT = 64
VALIDATION_FRACTION = 0.20
MAX_CLUSTERS = 10
RANDOM_STATE = 29
CLUSTER_NAMES = ("clap", "full_band")
_UNLABELED_PREFIX = "[UNLABELED]"


def _read_species_tiers(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Species distribution report not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not {"species_label", "imbalance_tier"}.issubset(rows.fieldnames or ()):
            raise ValueError(f"{path} lacks species_label or imbalance_tier")
        return {row["species_label"]: row["imbalance_tier"] for row in rows}


def _species_label(record: dict[str, Any]) -> str:
    species_code = record.get("species_code")
    if isinstance(species_code, str) and species_code.strip():
        return species_code.strip()
    scientific_name = record.get("scientific_name")
    if isinstance(scientific_name, str) and scientific_name.strip():
        return scientific_name.strip()
    display_name = record.get("display_name")
    if isinstance(display_name, str) and display_name.strip():
        return f"{_UNLABELED_PREFIX} {display_name.strip()}"
    return f"{_UNLABELED_PREFIX} missing species label"


def _location_entities(
    location: dict[str, Any] | None,
) -> tuple[set[tuple[str, Any]], list[str], list[tuple[float, float]]]:
    location = location or {}
    names = sorted(
        {
            " ".join(
                unicodedata.normalize("NFKC", str(value)).casefold().split()
            )
            for value in (location.get("name") or [])
            if str(value).strip()
        }
    )
    coordinates = sorted(
        {
            (round(float(value["lat"]), 4), round(float(value["lon"]), 4))
            for value in (location.get("coordinates") or [])
            if value.get("lat") is not None and value.get("lon") is not None
        }
    )
    entities = {("name", name) for name in names}
    entities.update(("coordinates", coordinate) for coordinate in coordinates)
    return entities, names, coordinates


def _assign_location_groups(
    records: list[dict[str, Any]],
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[dict[str, list[int]], set[str]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    owners: dict[tuple[str, Any], int] = {}
    for index, record in enumerate(records):
        for entity in record["_entities"]:
            previous = owners.get(entity)
            if previous is None:
                owners[entity] = index
            else:
                union(index, previous)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[find(index)].append(index)
    component_order = sorted(
        components.values(),
        key=lambda indices: min(
            (
                repr(entity)
                for index in indices
                for entity in records[index]["_entities"]
            ),
            default=(
                f"unlocated-{records[min(indices)]['parquet_file']}-"
                f"{records[min(indices)]['parquet_row_index']:012d}"
            ),
        ),
    )
    groups = {
        f"location-group-{position:04d}": indices
        for position, indices in enumerate(component_order, start=1)
    }

    eligible_groups = [
        group
        for group, indices in groups.items()
        if any(
            kind in ("name", "coordinates")
            for index in indices
            for kind, _ in records[index]["_entities"]
        )
    ]
    target_size = round(len(records) * validation_fraction)
    best_for_sum: dict[int, tuple[str, ...]] = {0: ()}
    for group in eligible_groups:
        group_size = len(groups[group])
        additions: dict[int, tuple[str, ...]] = {}
        for current_size, chosen in best_for_sum.items():
            new_size = current_size + group_size
            if new_size >= len(records):
                continue
            candidate = chosen + (group,)
            existing = best_for_sum.get(new_size) or additions.get(new_size)
            if existing is None or (-len(candidate), candidate) < (
                -len(existing),
                existing,
            ):
                additions[new_size] = candidate
        best_for_sum.update(additions)
    attainable_sizes = [size for size in best_for_sum if size]
    if not attainable_sizes:
        raise ValueError(
            "Cannot create a non-empty location-held-out validation split"
        )
    selected_size = min(
        attainable_sizes,
        key=lambda size: (
            abs(size - target_size),
            -len(best_for_sum[size]),
            best_for_sum[size],
        ),
    )
    validation_groups = set(best_for_sum[selected_size])
    species_labels = sorted(
        {
            str(record["species_label"])
            for record in records
            if record.get("species_label")
            and not str(record["species_label"]).startswith(_UNLABELED_PREFIX)
        }
    )
    tiers = sorted(
        {str(record["imbalance_tier"]) for record in records if record.get("imbalance_tier")}
    )
    if species_labels or tiers:
        validation_groups = _balance_validation_groups(
            records,
            groups,
            eligible_groups,
            selected_size,
            validation_fraction,
            species_labels,
            tiers,
        )
    return groups, validation_groups


def _balance_validation_groups(
    records: list[dict[str, Any]],
    groups: dict[str, list[int]],
    eligible_groups: list[str],
    selected_size: int,
    validation_fraction: float,
    species_labels: list[str],
    tiers: list[str],
) -> set[str]:
    group_count = len(eligible_groups)
    species_count = len(species_labels)
    tier_count = len(tiers)
    strata_count = species_count + tier_count
    variable_count = group_count + strata_count
    group_sizes = np.asarray(
        [len(groups[group]) for group in eligible_groups], dtype=np.float64
    )
    species_counts = np.zeros((group_count, species_count), dtype=np.float64)
    tier_counts = np.zeros((group_count, tier_count), dtype=np.float64)
    species_indices = {label: index for index, label in enumerate(species_labels)}
    tier_indices = {tier: index for index, tier in enumerate(tiers)}
    for group_index, group in enumerate(eligible_groups):
        for record_index in groups[group]:
            record = records[record_index]
            label = record.get("species_label")
            tier = record.get("imbalance_tier")
            if label in species_indices:
                species_counts[group_index, species_indices[label]] += 1
            if tier in tier_indices:
                tier_counts[group_index, tier_indices[tier]] += 1

    constraints = lil_matrix((1 + 2 * strata_count, variable_count))
    lower_bounds = np.full(1 + 2 * strata_count, -np.inf)
    upper_bounds = np.full(1 + 2 * strata_count, np.inf)
    constraints[0, :group_count] = group_sizes
    lower_bounds[0] = upper_bounds[0] = selected_size
    objective = np.zeros(variable_count)
    objective[:group_count] = -1e-6
    objective[group_count:] = 1 / np.maximum(
        np.concatenate((species_counts.sum(axis=0), tier_counts.sum(axis=0))),
        1,
    )
    row_index = 1
    for counts, offset in (
        (species_counts, group_count),
        (tier_counts, group_count + species_count),
    ):
        totals = counts.sum(axis=0)
        for stratum_index, total in enumerate(totals):
            constraints[row_index, :group_count] = counts[:, stratum_index]
            constraints[row_index, offset + stratum_index] = -1
            upper_bounds[row_index] = validation_fraction * total
            row_index += 1
            constraints[row_index, :group_count] = -counts[:, stratum_index]
            constraints[row_index, offset + stratum_index] = -1
            upper_bounds[row_index] = -validation_fraction * total
            row_index += 1

    result = milp(
        objective,
        integrality=np.concatenate(
            (np.ones(group_count), np.zeros(strata_count))
        ),
        bounds=Bounds(
            np.zeros(variable_count),
            np.concatenate((np.ones(group_count), np.full(strata_count, np.inf))),
        ),
        constraints=LinearConstraint(
            constraints.tocsr(), lower_bounds, upper_bounds
        ),
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"Could not balance location-held-out groups: {result.message}"
        )
    selected_groups = {
        group
        for index, group in enumerate(eligible_groups)
        if result.x[index] >= 0.5
    }
    if not selected_groups:
        raise RuntimeError("Location-group optimizer selected no validation groups")
    return selected_groups


def build_location_manifest(
    data_dir: Path,
    output_dir: Path,
    species_tiers: dict[str, str],
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet shards found in {data_dir}")

    records: list[dict[str, Any]] = []
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        required = {
            "record_number",
            "species_code",
            "location",
            "observation_date",
            "sound",
            "audio",
            "signal",
        }
        if not required.issubset(parquet_file.schema_arrow.names):
            raise ValueError(
                f"{parquet_path.name} is missing required fields "
                f"{sorted(required - set(parquet_file.schema_arrow.names))}"
            )
        row_offset = 0
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            row_group_offset = 0
            for batch in parquet_file.iter_batches(
                batch_size=64,
                row_groups=[row_group_index],
                columns=[
                    column
                    for column in (
                        "record_number",
                        "species_code",
                        "display_name",
                        "scientific_name",
                        "mid",
                        "class_id",
                        "observation_date",
                        "location",
                        "sound",
                        "audio",
                        "signal",
                    )
                    if column in parquet_file.schema_arrow.names
                ],
            ):
                for row_index, record in enumerate(batch.to_pylist()):
                    source_row_index = row_offset + row_group_offset + row_index
                    payload = ((record.get("audio") or {}).get("bytes"))
                    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
                        raise ValueError(
                            f"Missing audio payload in {parquet_path.name} row "
                            f"{source_row_index}"
                        )
                    digest = hashlib.sha256(payload).hexdigest()
                    location_entities, location_names, coordinates = (
                        _location_entities(record.get("location"))
                    )
                    location_entities.add(("audio", digest))
                    label = _species_label(record)
                    sound = record.get("sound") or {}
                    position = (
                        (record.get("signal") or {}).get("position")
                    )
                    records.append(
                        {
                            "record_number": str(record.get("record_number") or ""),
                            "species_code": str(record.get("species_code") or ""),
                            "species_label": label,
                            "imbalance_tier": species_tiers.get(
                                label,
                                "unlabeled"
                                if label.startswith(_UNLABELED_PREFIX)
                                else "unclassified",
                            ),
                            "display_name": record.get("display_name") or "",
                            "scientific_name": record.get("scientific_name") or "",
                            "mid": record.get("mid") or "",
                            "class_id": record.get("class_id") or "",
                            "observation_date": record.get("observation_date") or "",
                            "location_names": " | ".join(location_names),
                            "location_coordinates": " | ".join(
                                f"{latitude:g},{longitude:g}"
                                for latitude, longitude in coordinates
                            ),
                            "metadata_sample_rate_hz": sound.get("sample_rate") or "",
                            "signal_position": repr(position) if position else "",
                            "parquet_file": parquet_path.name,
                            "parquet_row_index": source_row_index,
                            "audio_sha256": digest,
                            "_entities": location_entities,
                            "_row_group_index": row_group_index,
                            "_row_index_in_group": row_group_offset + row_index,
                        }
                    )
                row_group_offset += batch.num_rows
            row_offset += parquet_file.metadata.row_group(row_group_index).num_rows
        print(f"Scanned metadata and location groups: {parquet_path.name}", flush=True)

    groups, validation_groups = _assign_location_groups(
        records, validation_fraction
    )
    payload_counts = Counter(record["audio_sha256"] for record in records)
    group_for_row = {
        index: group for group, indices in groups.items() for index in indices
    }
    manifest_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        group = group_for_row[index]
        has_location = any(
            kind in ("name", "coordinates")
            for kind, _ in record["_entities"]
        )
        manifest_rows.append(
            {
                key: value
                for key, value in record.items()
                if key not in {
                    "_entities",
                    "_row_group_index",
                    "_row_index_in_group",
                }
            }
            | {
                "split": "validation" if group in validation_groups else "train",
                "location_group": group,
                "location_metadata_available": has_location,
                "duplicate_audio_payload": (
                    payload_counts[record["audio_sha256"]] > 1
                ),
                "_row_group_index": record["_row_group_index"],
                "_row_index_in_group": record["_row_index_in_group"],
            }
        )
    manifest_path = output_dir / "watkins_location_split.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in manifest_rows
    ]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)
    return manifest_rows, {
        (row["parquet_file"], row["parquet_row_index"]): row
        for row in manifest_rows
    }


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


def _audio_chunks(audio: np.ndarray) -> list[tuple[int, float, float, np.ndarray]]:
    chunk_frames = TARGET_SAMPLE_RATE * CLIP_SECONDS
    chunks = []
    for chunk_index, start in enumerate(range(0, len(audio), chunk_frames)):
        end = min(start + chunk_frames, len(audio))
        chunks.append(
            (
                chunk_index,
                start / TARGET_SAMPLE_RATE,
                end / TARGET_SAMPLE_RATE,
                audio[start:end],
            )
        )
    return chunks


def _full_band_features(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        raise ValueError("Cannot extract spectral features from empty audio")
    frequencies, power = welch(
        audio,
        fs=TARGET_SAMPLE_RATE,
        nperseg=min(8192, len(audio)),
    )
    edges = np.geomspace(
        50, TARGET_SAMPLE_RATE / 2, FULL_BAND_FEATURE_COUNT + 1
    )
    bin_power = np.zeros(FULL_BAND_FEATURE_COUNT, dtype=np.float64)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (frequencies >= lower) & (frequencies < upper)
        if index == FULL_BAND_FEATURE_COUNT - 1:
            mask |= frequencies == TARGET_SAMPLE_RATE / 2
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
            "`requirements-be3c-baseline.txt`."
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
        outputs = model.get_audio_features(**model_inputs)
    embeddings = outputs.pooler_output
    if embeddings is None:
        raise RuntimeError("CLAP did not return pooled audio embeddings")
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return embeddings.cpu().numpy().astype(np.float32, copy=False)


def extract_features(
    data_dir: Path,
    output_dir: Path,
    audit_path: Path,
    batch_size: int = 8,
    validation_fraction: float = VALIDATION_FRACTION,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    tiers = _read_species_tiers(audit_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows, manifest = build_location_manifest(
        data_dir, output_dir, tiers, validation_fraction
    )
    model, processor, device = _load_clap()

    metadata_rows: list[dict[str, Any]] = []
    full_band_rows: list[np.ndarray] = []
    clap_rows: list[np.ndarray] = []
    batch_waveforms: list[np.ndarray] = []
    processed: set[tuple[str, int]] = set()

    def flush_batch() -> None:
        if not batch_waveforms:
            return
        batch_embeddings = _embed_batch(
            model, processor, device, batch_waveforms
        )
        expected = len(metadata_rows) - len(clap_rows)
        if len(batch_embeddings) != expected:
            raise RuntimeError(
                f"CLAP produced {len(batch_embeddings)} features for "
                f"{expected} pending chunks"
            )
        clap_rows.extend(batch_embeddings)
        batch_waveforms.clear()

    for parquet_path in sorted(data_dir.glob("*.parquet")):
        parquet_file = pq.ParquetFile(parquet_path)
        row_offset = 0
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            group_offset = 0
            for batch in parquet_file.iter_batches(
                batch_size=64,
                row_groups=[row_group_index],
                columns=["record_number", "audio"],
            ):
                for row_index, record in enumerate(batch.to_pylist()):
                    source_row_index = row_offset + group_offset + row_index
                    source_key = (parquet_path.name, source_row_index)
                    manifest_row = manifest.get(source_key)
                    if manifest_row is None:
                        raise ValueError(
                            f"Source row missing from location manifest: {source_key}"
                        )
                    payload = ((record.get("audio") or {}).get("bytes"))
                    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
                        raise ValueError(f"Missing audio payload at {source_key}")
                    with sf.SoundFile(BytesIO(payload)) as audio_file:
                        source_rate = audio_file.samplerate
                        source_audio = audio_file.read(
                            dtype="float32", always_2d=True
                        ).mean(axis=1)
                    resampled_audio = _resample_audio(source_audio, source_rate)
                    chunks = _audio_chunks(resampled_audio)
                    if not chunks:
                        raise ValueError(f"Audio contains no samples at {source_key}")
                    for chunk_index, start, end, chunk in chunks:
                        metadata_rows.append(
                            {
                                key: value
                                for key, value in manifest_row.items()
                                if not key.startswith("_")
                            }
                            | {
                                "source_sample_rate_hz": source_rate,
                                "source_nyquist_hz": source_rate / 2,
                                "clap_sample_rate_hz": TARGET_SAMPLE_RATE,
                                "chunk_index": chunk_index,
                                "start_seconds": start,
                                "end_seconds": end,
                                "chunk_seconds": end - start,
                            }
                        )
                        full_band_rows.append(_full_band_features(chunk))
                        batch_waveforms.append(chunk)
                        if len(batch_waveforms) >= batch_size:
                            flush_batch()
                    processed.add(source_key)
                group_offset += batch.num_rows
            row_offset += parquet_file.metadata.row_group(row_group_index).num_rows
        print(f"Embedded source shard: {parquet_path.name}", flush=True)
    flush_batch()

    missing_rows = set(manifest) - processed
    if missing_rows:
        raise ValueError(
            f"{len(missing_rows)} location-manifest rows were not embedded; "
            f"first: {min(missing_rows)}"
        )
    if len(clap_rows) != len(metadata_rows):
        raise RuntimeError(
            f"CLAP embeddings {len(clap_rows)} do not match chunks "
            f"{len(metadata_rows)}"
        )

    metadata_path = output_dir / "watkins_embedding_chunks.csv"
    with metadata_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metadata_rows[0]))
        writer.writeheader()
        writer.writerows(metadata_rows)
    feature_path = output_dir / "watkins_embedding_features.npz"
    np.savez_compressed(
        feature_path,
        clap_embeddings=np.stack(clap_rows),
        full_band_features=np.stack(full_band_rows),
        model_id=np.asarray(MODEL_ID),
        clap_sample_rate=np.asarray(TARGET_SAMPLE_RATE),
        full_band_frequency_range_hz=np.asarray(
            [50, TARGET_SAMPLE_RATE / 2]
        ),
    )
    return {
        "record_count": len(manifest_rows),
        "chunk_count": len(metadata_rows),
        "train_count": sum(row["split"] == "train" for row in manifest_rows),
        "validation_count": sum(
            row["split"] == "validation" for row in manifest_rows
        ),
        "location_groups": len({row["location_group"] for row in manifest_rows}),
        "clap_embeddings": feature_path,
        "chunk_metadata": metadata_path,
        "device": str(device),
    }


def _read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _aggregate_recordings(
    metadata: list[dict[str, str]],
    clap_chunks: np.ndarray,
    full_band_chunks: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    if len(metadata) != len(clap_chunks) or len(metadata) != len(full_band_chunks):
        raise ValueError("Chunk metadata and feature counts must match")
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[(row["parquet_file"], row["parquet_row_index"])].append(index)
    rows: list[dict[str, Any]] = []
    clap_records: list[np.ndarray] = []
    full_band_records: list[np.ndarray] = []
    for source_key, indices in groups.items():
        chunks = [metadata[index] for index in indices]
        for key in ("record_number", "split", "location_group", "species_label"):
            if len({row[key] for row in chunks}) != 1:
                raise ValueError(f"Chunks for {source_key} disagree on {key}")
        vector = np.mean(clap_chunks[indices], axis=0)
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError(f"Zero CLAP mean for source row {source_key}")
        clap_records.append((vector / norm).astype(np.float32))
        full_band_records.append(
            np.mean(full_band_chunks[indices], axis=0).astype(np.float32)
        )
        first = chunks[0]
        rows.append(
            {
                key: first[key]
                for key in (
                    "record_number",
                    "species_code",
                    "species_label",
                    "imbalance_tier",
                    "display_name",
                    "scientific_name",
                    "mid",
                    "class_id",
                    "observation_date",
                    "location_names",
                    "location_coordinates",
                    "metadata_sample_rate_hz",
                    "signal_position",
                    "parquet_file",
                    "parquet_row_index",
                    "split",
                    "location_group",
                    "location_metadata_available",
                    "duplicate_audio_payload",
                    "source_sample_rate_hz",
                    "source_nyquist_hz",
                )
            }
            | {"chunk_count": len(indices)}
        )
    return rows, np.stack(clap_records), np.stack(full_band_records)


def _cluster_split(
    representation: str,
    features: np.ndarray,
    rows: list[dict[str, Any]],
    max_clusters: int = MAX_CLUSTERS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_clusters < 2:
        raise ValueError("max_clusters must be at least 2")
    train_indices = [
        index for index, row in enumerate(rows) if row["split"] == "train"
    ]
    validation_indices = [
        index for index, row in enumerate(rows) if row["split"] == "validation"
    ]
    if len(train_indices) < 3 or not validation_indices:
        raise ValueError("Need at least three train and one validation record")
    if representation == "full_band":
        scaler = StandardScaler()
        train_features = scaler.fit_transform(features[train_indices])
        validation_features = scaler.transform(features[validation_indices])
    elif representation == "clap":
        train_features = features[train_indices]
        validation_features = features[validation_indices]
    else:
        raise ValueError(f"Unsupported representation: {representation}")

    max_clusters = min(max_clusters, len(train_indices) - 1)
    scores: list[dict[str, Any]] = []
    candidates: list[tuple[float, int, MiniBatchKMeans, np.ndarray]] = []
    random_generator = np.random.default_rng(RANDOM_STATE)
    silhouette_indices = (
        random_generator.choice(
            len(train_indices), size=min(3000, len(train_indices)), replace=False
        )
        if len(train_indices) > 3000
        else np.arange(len(train_indices))
    )
    for cluster_count in range(2, max_clusters + 1):
        model = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=RANDOM_STATE,
            n_init=3,
            batch_size=min(2048, len(train_indices)),
            max_iter=100,
            reassignment_ratio=0.01,
        )
        train_labels = model.fit_predict(train_features)
        silhouette = _sampled_silhouette(
            train_features[silhouette_indices],
            train_labels[silhouette_indices],
        )
        scores.append(
            {
                "representation": representation,
                "cluster_count": cluster_count,
                "train_silhouette_sample": (
                    "" if silhouette is None else silhouette
                ),
            }
        )
        if silhouette is None:
            continue
        candidates.append((silhouette, -cluster_count, model, train_labels))

    if not candidates:
        raise ValueError(
            f"No valid silhouette score could be computed for {representation}"
        )
    _, negative_count, model, train_labels = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    selected_count = -negative_count
    validation_labels = model.predict(validation_features)
    assignments_by_index: dict[int, dict[str, Any]] = {}
    for indices, labels in (
        (train_indices, train_labels),
        (validation_indices, validation_labels),
    ):
        for index, cluster in zip(indices, labels):
            assignments_by_index[index] = {
                **rows[index],
                "representation": representation,
                "selected_cluster_count": selected_count,
                "cluster": int(cluster),
            }
    assignments = [
        assignments_by_index[index] for index in range(len(rows))
    ]
    if len(assignments) != len(rows):
        raise RuntimeError("Some source recordings were not assigned a cluster")
    return scores, assignments


def _sampled_silhouette(
    features: np.ndarray, labels: np.ndarray
) -> float | None:
    cluster_count = len(np.unique(labels))
    if cluster_count < 2 or cluster_count >= len(labels):
        return None
    return float(silhouette_score(features, labels))


def _association_metrics(
    representation: str,
    assignments: list[dict[str, Any]],
    features: np.ndarray,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        split_indices = [
            index
            for index, row in enumerate(assignments)
            if row["split"] == split
        ]
        for tier in ("dominant", "moderate", "rare", "unlabeled", "unclassified"):
            indices = [
                index
                for index in split_indices
                if assignments[index]["imbalance_tier"] == tier
            ]
            if not indices:
                continue
            subset = [assignments[index] for index in indices]
            species = [row["species_label"] for row in subset]
            clusters = [row["cluster"] for row in subset]
            locations = [row["location_group"] for row in subset]
            sample_rates = [row["source_sample_rate_hz"] for row in subset]
            label_count = len(set(species))
            cluster_count = len(set(clusters))
            silhouette: float | str = ""
            if len(indices) >= 3 and 1 < cluster_count < len(indices):
                sample_size = min(3000, len(indices))
                sampled_indices = (
                    np.random.default_rng(RANDOM_STATE).choice(
                        len(indices), size=sample_size, replace=False
                    )
                    if len(indices) > sample_size
                    else np.arange(len(indices))
                )
                sampled_features = features[indices][sampled_indices]
                sampled_labels = np.asarray(clusters)[sampled_indices]
                sampled_silhouette = _sampled_silhouette(
                    sampled_features, sampled_labels
                )
                if sampled_silhouette is not None:
                    silhouette = sampled_silhouette
            metrics.append(
                {
                    "representation": representation,
                    "split": split,
                    "imbalance_tier": tier,
                    "record_count": len(indices),
                    "species_count": label_count,
                    "cluster_count_present": cluster_count,
                    "species_cluster_nmi": (
                        normalized_mutual_info_score(species, clusters)
                        if tier not in ("unlabeled", "unclassified")
                        and label_count > 1
                        else ""
                    ),
                    "location_cluster_nmi": (
                        normalized_mutual_info_score(locations, clusters)
                        if len(set(locations)) > 1
                        else ""
                    ),
                    "sample_rate_cluster_nmi": (
                        normalized_mutual_info_score(sample_rates, clusters)
                        if len(set(sample_rates)) > 1
                        else ""
                    ),
                    "cluster_silhouette": silhouette,
                }
            )
    return metrics


def evaluate_and_cluster(
    output_dir: Path,
    max_clusters: int = MAX_CLUSTERS,
) -> dict[str, Any]:
    metadata_path = output_dir / "watkins_embedding_chunks.csv"
    features_path = output_dir / "watkins_embedding_features.npz"
    if not metadata_path.is_file() or not features_path.is_file():
        raise FileNotFoundError("Run feature extraction before clustering")
    with np.load(features_path) as stored:
        clap_chunks = stored["clap_embeddings"]
        full_band_chunks = stored["full_band_features"]
    record_rows, clap_features, full_band_features = _aggregate_recordings(
        _read_metadata(metadata_path), clap_chunks, full_band_chunks
    )
    all_scores: list[dict[str, Any]] = []
    all_assignments: list[dict[str, Any]] = []
    tier_metrics: list[dict[str, Any]] = []
    for representation, features in (
        ("clap", clap_features),
        ("full_band", full_band_features),
    ):
        scores, assignments = _cluster_split(
            representation, features, record_rows, max_clusters=max_clusters
        )
        all_scores.extend(scores)
        all_assignments.extend(assignments)
        selected_features = (
            features
            if representation == "clap"
            else StandardScaler().fit(
                features[
                    [
                        index
                        for index, row in enumerate(record_rows)
                        if row["split"] == "train"
                    ]
                ]
            ).transform(features)
        )
        tier_metrics.extend(
            _association_metrics(representation, assignments, selected_features)
        )

    _write_csv(output_dir / "watkins_cluster_scores.csv", all_scores)
    _write_csv(output_dir / "watkins_cluster_assignments.csv", all_assignments)
    _write_csv(output_dir / "watkins_tier_evaluation.csv", tier_metrics)
    return {
        "record_count": len(record_rows),
        "scores": all_scores,
        "assignments": all_assignments,
        "tier_metrics": tier_metrics,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_baseline_report(
    output_dir: Path, result: dict[str, Any], chunk_count: int
) -> Path:
    manifest = _read_metadata(output_dir / "watkins_location_split.csv")
    tiers = Counter(row["imbalance_tier"] for row in manifest)
    split_tiers = {
        split: Counter(
            row["imbalance_tier"] for row in manifest if row["split"] == split
        )
        for split in ("train", "validation")
    }
    split_group_counts = {
        split: len(
            {
                row["location_group"]
                for row in manifest
                if row["split"] == split
            }
        )
        for split in ("train", "validation")
    }
    rows = result["record_count"]
    report_lines = [
        "# Watkins full-dataset frozen representation baseline",
        "",
        f"- Source recordings: {rows:,}; location/date/sample-rate metadata and "
        "audio were read without modifying Parquet shards.",
        f"- Location-connected groups: {result['location_groups']:,}; split: "
        f"{result['train_count']:,} train / {result['validation_count']:,} "
        f"validation ({result['validation_count'] / rows:.1%}).",
        f"- Groups represented: {split_group_counts['train']:,} train / "
        f"{split_group_counts['validation']:,} validation. The split targets "
        "the requested validation size, then favors more held-out groups when "
        "multiple group combinations are equally close.",
        "- Rows without usable location metadata stay in training. Exact duplicate "
        "audio payloads are linked into one group so they cannot cross the split.",
        f"- Species tiers in original records: {dict(tiers)}.",
        f"- Validation species tiers: {dict(split_tiers['validation'])}.",
        f"- Non-overlapping chunks: {chunk_count:,}; CLAP embeddings "
        "were extracted from a frozen pretrained encoder; no model weights were "
        "trained.",
        "- A parallel 64-bin log-power representation spans nominally 50 Hz–24 kHz. "
        "It is an acoustic baseline, not a learned embedding.",
        "- CLAP operates at 48 kHz and focuses on approximately 50 Hz–14 kHz; "
        "up-sampling lower-rate recordings cannot restore missing frequencies.",
        "- K-means is fit only on training-record features; held-out recordings "
        "are assigned to the fitted centroids. All chunks from a recording stay "
        "together and are averaged before record-level clustering.",
        "- Species labels are not model targets. Tier metrics summarize "
        "species/cluster and recording-condition associations.",
        "- Treat clusters as candidate acoustic groups, not confirmed call types. "
        "Location is coarse metadata, not verified session identity.",
        "",
        "See `watkins_tier_evaluation.csv`, `watkins_cluster_scores.csv`, "
        "`watkins_cluster_assignments.csv`, and `watkins_location_split.csv`.",
        "",
    ]
    report_path = output_dir / "watkins_baseline_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def run_baseline(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    audit_path: Path = AUDIT_PATH,
    batch_size: int = 8,
    max_clusters: int = MAX_CLUSTERS,
) -> dict[str, Any]:
    result = extract_features(data_dir, output_dir, audit_path, batch_size)
    evaluated = evaluate_and_cluster(output_dir, max_clusters=max_clusters)
    report_path = write_baseline_report(
        output_dir, result, result["chunk_count"]
    )
    return {
        **result,
        "evaluation": evaluated,
        "report_path": report_path,
        "tiers": dict(
            Counter(
                row["imbalance_tier"]
                for row in _read_metadata(
                    output_dir / "watkins_location_split.csv"
                )
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frozen embeddings and evaluate the full Watkins dataset."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--audit-report", type=Path, default=AUDIT_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-clusters", type=int, default=MAX_CLUSTERS)
    args = parser.parse_args()
    result = run_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        audit_path=args.audit_report,
        batch_size=args.batch_size,
        max_clusters=args.max_clusters,
    )
    print(
        f"Completed frozen embeddings for {result['record_count']:,} "
        f"recordings ({result['chunk_count']:,} chunks) on "
        f"{result['device']}."
    )
    print(
        f"Location-held-out split: {result['train_count']:,} train, "
        f"{result['validation_count']:,} validation."
    )
    print(f"Report: {result['report_path']}")


if __name__ == "__main__":
    main()
