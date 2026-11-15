"""Audit BE3C clips and create a reproducible location-held-out split."""

from __future__ import annotations

import csv
import unicodedata
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import spectrogram


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "watkins-marine-mammal-full-cuts" / "data"
OUTPUT_DIR = PROJECT_DIR / "be3c_feasibility"
SPECIES_CODE = "BE3C"
VALIDATION_FRACTION = 0.20
def _normalize_location(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _entities(record: dict[str, Any]) -> tuple[list[tuple[str, Any]], list[str], list[tuple[float, float]]]:
    location = record.get("location") or {}
    names = [
        str(name).strip()
        for name in (
            record.get("location_names") or location.get("name") or []
        )
        if str(name).strip()
    ]
    coordinates = sorted(
        {
            (round(float(item["lat"]), 4), round(float(item["lon"]), 4))
            for item in (
                record.get("location_coordinates")
                or location.get("coordinates")
                or []
            )
            if item.get("lat") is not None and item.get("lon") is not None
        }
    )
    entities: list[tuple[str, Any]] = [
        ("name", _normalize_location(name)) for name in names
    ]
    entities.extend(("coordinates", coordinate) for coordinate in coordinates)
    return sorted(set(entities), key=repr), names, coordinates


def _assign_connected_location_groups(
    records: list[dict[str, Any]],
) -> tuple[dict[str, list[int]], int]:
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
    unlocated = 0
    for index, record in enumerate(records):
        entities, _, _ = _entities(record)
        if not entities:
            unlocated += 1
        for entity in entities:
            if entity in owners:
                union(index, owners[entity])
            else:
                owners[entity] = index

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[find(index)].append(index)

    ordered_components = sorted(
        components.values(),
        key=lambda indices: min(
            repr(entity)
            for index in indices
            for entity in _entities(records[index])[0]
        )
        if any(_entities(records[index])[0] for index in indices)
        else f"unlocated-{min(indices):012d}",
    )
    result = {
        f"location-group-{group_index:03d}": indices
        for group_index, indices in enumerate(ordered_components, start=1)
    }
    return result, unlocated


def _select_validation_groups(
    groups: dict[str, list[int]], target_fraction: float
) -> set[str]:
    eligible = sorted(groups)
    total_records = sum(len(indices) for indices in groups.values())
    target_records = round(total_records * target_fraction)
    best_for_sum: dict[int, tuple[str, ...]] = {0: ()}
    for group in eligible:
        group_size = len(groups[group])
        additions: dict[int, tuple[str, ...]] = {}
        for record_sum, selected in best_for_sum.items():
            new_sum = record_sum + group_size
            if new_sum >= total_records:
                continue
            candidate = selected + (group,)
            current = best_for_sum.get(new_sum) or additions.get(new_sum)
            if current is None or (len(candidate), candidate) < (
                len(current),
                current,
            ):
                additions[new_sum] = candidate
        best_for_sum.update(additions)

    candidates = [
        (abs(record_sum - target_records), len(selected), selected)
        for record_sum, selected in best_for_sum.items()
        if selected
    ]
    if not candidates:
        raise ValueError("At least two independent location groups are required")
    return set(min(candidates)[2])


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_split(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    species_code: str = SPECIES_CODE,
    validation_fraction: float = VALIDATION_FRACTION,
) -> dict[str, Any]:
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet shards found in {data_dir}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    quality = Counter()
    channels = Counter()
    source_ids = Counter()
    field_coverage = Counter()
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        row_offset = 0
        columns = [
            column
            for column in (
                "species_code",
                "record_number",
                "mid",
                "display_name",
                "scientific_name",
                "observation_date",
                "location",
                "sound",
                "signal",
            )
            if column in parquet_file.schema_arrow.names
        ]
        species_column_index = next(
            index
            for index in range(parquet_file.metadata.num_columns)
            if parquet_file.metadata.schema.column(index).path == "species_code"
        )
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            row_group = parquet_file.metadata.row_group(row_group_index)
            stats = row_group.column(species_column_index).statistics
            if (
                stats is not None
                and stats.has_min_max
                and not (stats.min <= species_code <= stats.max)
            ):
                row_offset += row_group.num_rows
                continue
            table = parquet_file.read_row_group(row_group_index, columns=columns)
            for row_in_group, row in enumerate(table.to_pylist()):
                if row.get("species_code") != species_code:
                    continue
                source_row = row_offset + row_in_group
                location = row.get("location") or {}
                names = location.get("name") or []
                coordinates = location.get("coordinates") or []
                sound = row.get("sound") or {}
                channel = sound.get("channel") or {}
                record = {
                    "parquet_file": parquet_path.name,
                    "parquet_row_index": source_row,
                    "_row_group_index": row_group_index,
                    "_row_index_in_group": row_in_group,
                    "record_number": row.get("record_number"),
                    "mid": row.get("mid"),
                    "display_name": row.get("display_name"),
                    "scientific_name": row.get("scientific_name"),
                    "observation_date": row.get("observation_date"),
                    "location_names": names,
                    "location_coordinates": coordinates,
                    "sample_rate_metadata": sound.get("sample_rate"),
                    "signal_position": (row.get("signal") or {}).get("position"),
                }
                records.append(record)
                if not row.get("observation_date"):
                    quality["missing_date"] += 1
                if names:
                    field_coverage["location_name"] += 1
                if coordinates:
                    field_coverage["location_coordinates"] += 1
                if not names and not coordinates:
                    quality["missing_location"] += 1
                if sound.get("sample_rate") is not None:
                    field_coverage["sample_rate"] += 1
                if channel.get("recorded") is not None:
                    field_coverage["channel_recorded"] += 1
                if channel.get("_source_nc") is not None:
                    field_coverage["channel_source_code"] += 1
                channels[
                    (
                        channel.get("_source_nc"),
                        channel.get("recorded"),
                        channel.get("multiplexed"),
                        channel.get("side"),
                    )
                ] += 1
                source_ids[row.get("mid")] += 1
            row_offset += row_group.num_rows
        print(f"Scanned metadata: {parquet_path.name}", flush=True)

    if not records:
        raise ValueError(f"No rows found for species_code={species_code}")

    groups, unlocated_count = _assign_connected_location_groups(records)
    print(f"Found {len(groups)} connected location groups", flush=True)
    print(f"Grouping {len(records)} records by location metadata", flush=True)
    validation_groups = _select_validation_groups(groups, validation_fraction)
    print(
        f"Found {len(groups)} connected location groups; "
        f"holding out {len(validation_groups)}",
        flush=True,
    )
    validation_indices = {
        index for group in validation_groups for index in groups[group]
    }
    row_split = {
        index: "validation" if index in validation_indices else "train"
        for index in range(len(records))
    }
    validation_rows = [
        index for index, split in row_split.items() if split == "validation"
    ]
    train_rows = [index for index, split in row_split.items() if split == "train"]
    if not train_rows or not validation_rows:
        raise RuntimeError("Location split produced an empty train or validation set")

    group_for_row = {
        index: group
        for group, indices in groups.items()
        for index in indices
    }
    group_summary: list[dict[str, Any]] = []
    for group, indices in groups.items():
        group_names = sorted(
            {
                name
                for index in indices
                for name in records[index]["location_names"]
            }
        )
        group_coordinates = sorted(
            {
                (float(coordinate["lat"]), float(coordinate["lon"]))
                for index in indices
                for coordinate in records[index]["location_coordinates"]
                if coordinate.get("lat") is not None
                and coordinate.get("lon") is not None
            }
        )
        group_summary.append(
            {
                "location_group": group,
                "split": "validation" if group in validation_groups else "train",
                "record_count": len(indices),
                "location_names": " | ".join(group_names),
                "coordinates": " | ".join(
                    f"{lat:g},{lon:g}" for lat, lon in group_coordinates
                ),
                "start_date": min(
                    (
                        records[index]["observation_date"]
                        for index in indices
                        if records[index]["observation_date"]
                    ),
                    default="",
                ),
                "end_date": max(
                    (
                        records[index]["observation_date"]
                        for index in indices
                        if records[index]["observation_date"]
                    ),
                    default="",
                ),
            }
        )

    manifest_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        manifest_rows.append(
            {
                "species_code": species_code,
                "split": row_split[index],
                "location_group": group_for_row[index],
                "parquet_file": record["parquet_file"],
                "parquet_row_index": record["parquet_row_index"],
                "record_number": record["record_number"],
                "mid": record["mid"],
                "observation_date": record["observation_date"],
                "location_names": " | ".join(record["location_names"]),
                "location_coordinates": " | ".join(
                    f"{coordinate['lat']:g},{coordinate['lon']:g}"
                    for coordinate in record["location_coordinates"]
                    if coordinate.get("lat") is not None
                    and coordinate.get("lon") is not None
                ),
                "sample_rate_metadata": record["sample_rate_metadata"],
            }
        )
    _write_csv(
        output_dir / "be3c_location_split.csv",
        list(manifest_rows[0]),
        manifest_rows,
    )
    _write_csv(
        output_dir / "be3c_location_groups.csv",
        list(group_summary[0]),
        sorted(group_summary, key=lambda group: group["location_group"]),
    )

    examples: list[tuple[str, int]] = []
    for split in ("train", "validation"):
        candidates = sorted(
            train_rows if split == "train" else validation_rows,
            key=lambda index: records[index]["record_number"] or "",
        )
        example_index = candidates[len(candidates) // 2]
        examples.append((split, example_index))

    sample_dir = output_dir / "be3c_examples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    example_records: list[dict[str, Any]] = []
    for split, example_index in examples:
        selected = records[example_index]
        parquet_path = data_dir / selected["parquet_file"]
        row_group_index = selected["_row_group_index"]
        row_in_group = selected["_row_index_in_group"]
        audio_row = pq.ParquetFile(parquet_path).read_row_group(
            row_group_index, columns=["audio"]
        ).column("audio")[row_in_group].as_py()
        payload = audio_row["bytes"]
        with sf.SoundFile(BytesIO(payload)) as audio_file:
            recording_duration = audio_file.frames / audio_file.samplerate
            preview_frames = min(audio_file.frames, int(audio_file.samplerate * 10))
            waveform = audio_file.read(
                frames=preview_frames, dtype="float32", always_2d=True
            )
            sample_rate = audio_file.samplerate
        preview_duration = len(waveform) / sample_rate
        sample_name = f"{split}_{selected['record_number']}.wav"
        sf.write(sample_dir / sample_name, waveform, sample_rate, subtype="PCM_16")
        example_records.append(
            {
                "split": split,
                "record_number": selected["record_number"],
                "parquet_file": selected["parquet_file"],
                "parquet_row_index": selected["parquet_row_index"],
                "location_group": group_for_row[example_index],
                "location_names": " | ".join(selected["location_names"]),
                "location_coordinates": " | ".join(
                    f"{coordinate['lat']:g},{coordinate['lon']:g}"
                    for coordinate in selected["location_coordinates"]
                    if coordinate.get("lat") is not None
                    and coordinate.get("lon") is not None
                ),
                "recording_duration_seconds": recording_duration,
                "preview_duration_seconds": preview_duration,
                "sample_rate": sample_rate,
                "wav_file": sample_name,
            }
        )

    _write_csv(
        output_dir / "be3c_example_clips.csv",
        list(example_records[0]),
        example_records,
    )
    _write_spectrograms(example_records, sample_dir, sample_dir.parent)

    split_description = "; ".join(
        f"{group['location_names'] or '[no location name]'} "
        f"({group['coordinates'] or 'coordinates missing'}): "
        f"{group['record_count']} records"
        for group in group_summary
        if group["location_group"] in validation_groups
    )
    validation_fraction_actual = len(validation_rows) / len(records)
    report_lines = [
        "# BE3C location-held-out feasibility audit",
        "",
        f"- Species code: `{species_code}` (long-finned pilot whale).",
        f"- Records: {len(records):,}; location/date/sample-rate/channel metadata was audited. The two selected WAV previews were decoded successfully.",
        f"- Location name coverage: {field_coverage['location_name']:,}/{len(records):,}.",
        f"- Coordinate coverage: {field_coverage['location_coordinates']:,}/{len(records):,}.",
        f"- Date coverage: {len(records) - quality['missing_date']:,}/{len(records):,}.",
        f"- Sample-rate metadata coverage: {field_coverage['sample_rate']:,}/{len(records):,}.",
        f"- Sample-rate metadata values observed: {len({record['sample_rate_metadata'] for record in records if record['sample_rate_metadata'] is not None})}; channel configurations: {len(channels)}.",
        f"- `mid` values observed: {dict(source_ids)}. The README documents `mid` as a species identifier, not recording-session metadata.",
        f"- Records with neither location names nor coordinates: {quality['missing_location']}; unlocated rows without a location grouping entity: {unlocated_count}. These remain in training.",
        "",
        "## Split",
        "",
        f"- Train: {len(train_rows):,} records.",
        f"- Validation: {len(validation_rows):,} records ({validation_fraction_actual:.1%}).",
        f"- Location-connected groups: {len(groups)}; validation groups: {len(validation_groups)}.",
        f"- Held-out location groups: {split_description}.",
        "- Group construction links rows sharing any normalized location name or rounded recorded coordinate.",
        "- The split is deterministic and selects whole connected location groups nearest the requested 20% validation size; all source records remain represented exactly once in the CSV manifest.",
        "- Dataset documentation describes `record_number` as the original Watkins recording identifier and `mid` as a species identifier. `mid` is not used as a session split key.",
        "",
        "## Caveats",
        "",
        "- Location coordinates are coarse (often rounded to whole degrees), and labels range from named places to broad regions. This is a preliminary geographic holdout, not proof of independent hydrophone/site conditions.",
        "- No explicit recording-session field was found. Recording notes and `signal.position` may help investigate session relationships, but do not assume they define sessions without validating their semantics.",
        "- Two WAV previews (up to 10 seconds each) and their spectrograms are provided for manual listening/visual review. The previous dataset audit fully decoded all audio and found 13 exact duplicates in the entire dataset; a BE3C-specific hash check found no duplicate payloads. This script decodes the selected previews but does not claim biological call identification.",
        "",
        "## Files",
        "",
        "- `be3c_location_split.csv`: one row per source record, assigned to train/validation.",
        "- `be3c_location_groups.csv`: location groups and split assignment.",
        "- `be3c_example_clips.csv`: metadata for representative clips.",
        "- `be3c_examples/`: extracted train and validation WAV samples.",
        "- `be3c_spectrograms.png`: representative log-frequency spectrograms.",
        "",
        "Original Parquet shards were read only and were not modified.",
        "",
    ]
    (output_dir / "be3c_feasibility_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return {
        "records": len(records),
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "validation_fraction": validation_fraction_actual,
        "location_groups": len(groups),
        "validation_groups": validation_groups,
        "group_summary": group_summary,
        "output_dir": output_dir,
    }


def _write_spectrograms(
    examples: list[dict[str, Any]],
    sample_dir: Path,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(examples), 1, figsize=(12, 8), squeeze=False)
    for axis, example in zip(axes[:, 0], examples):
        wav_path = sample_dir / example["wav_file"]
        audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        frequencies, times, power = spectrogram(
            audio.mean(axis=1),
            fs=sample_rate,
            nperseg=min(1024, len(audio)),
            noverlap=min(768, max(0, min(1024, len(audio)) - 1)),
        )
        valid = frequencies > 0
        axis.pcolormesh(
            times,
            frequencies[valid] / 1000,
            10 * np.log10(np.maximum(power[valid], 1e-15)),
            shading="auto",
            cmap="magma",
        )
        axis.set_yscale("log")
        axis.set_ylabel("Frequency (kHz)")
        axis.set_title(
            f"{example['split'].title()} | record {example['record_number']} | "
            f"{example['preview_duration_seconds']:.2f} s preview | "
            f"{example['location_names'] or example['location_coordinates'] or 'location unavailable'}"
        )
    axes[-1, 0].set_xlabel("Time (seconds)")
    figure.suptitle("BE3C representative audio spectrograms", y=1.01)
    figure.tight_layout()
    figure.savefig(output_dir / "be3c_spectrograms.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    result = prepare_split()
    print(f"Species: {SPECIES_CODE}")
    print(f"Records: {result['records']}")
    print(f"Train: {result['train_count']}")
    print(
        f"Validation: {result['validation_count']} "
        f"({result['validation_fraction']:.1%})"
    )
    print(
        f"Location groups: {result['location_groups']}; "
        f"held out: {', '.join(sorted(result['validation_groups']))}"
    )
    print(f"Outputs: {result['output_dir']}")
