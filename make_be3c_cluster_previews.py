"""Export medoid-like WAV previews from each selected BE3C cluster."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any
from io import BytesIO

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from sklearn.preprocessing import StandardScaler

from cluster_be3c_baseline import _aggregate_recordings


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "watkins-marine-mammal-full-cuts" / "data"
OUTPUT_DIR = PROJECT_DIR / "be3c_feasibility"
MAX_PREVIEW_SECONDS = 10


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _choose_cluster_representatives(
    representation: str,
    record_rows: list[dict[str, Any]],
    features: np.ndarray,
    assignments: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if representation == "full_band":
        scaler = StandardScaler()
        train_indices = [
            index
            for index, row in enumerate(record_rows)
            if row["split"] == "train"
        ]
        features = features.copy()
        features[train_indices] = scaler.fit_transform(features[train_indices])
        validation_indices = [
            index
            for index, row in enumerate(record_rows)
            if row["split"] == "validation"
        ]
        features[validation_indices] = scaler.transform(
            features[validation_indices]
        )

    assignment_lookup = {
        row["record_number"]: int(row["cluster"])
        for row in assignments
        if row["representation"] == representation
    }
    if len(assignment_lookup) != len(record_rows):
        raise ValueError(
            f"Expected one {representation} cluster assignment per recording"
        )

    representatives: list[dict[str, Any]] = []
    clusters = sorted(set(assignment_lookup.values()))
    for cluster in clusters:
        train_indices = [
            index
            for index, row in enumerate(record_rows)
            if row["split"] == "train"
            and assignment_lookup[row["record_number"]] == cluster
        ]
        if not train_indices:
            raise ValueError(
                f"Cluster {cluster} has no training recordings for {representation}"
            )
        centroid = np.mean(features[train_indices], axis=0)
        for split in ("train", "validation"):
            candidates = [
                index
                for index, row in enumerate(record_rows)
                if row["split"] == split
                and assignment_lookup[row["record_number"]] == cluster
            ]
            if not candidates:
                continue
            candidate_groups = (
                {
                    group: [
                        index
                        for index in candidates
                        if record_rows[index]["location_group"] == group
                    ]
                    for group in sorted(
                        {record_rows[index]["location_group"] for index in candidates}
                    )
                }
                if split == "validation"
                else {"": candidates}
            )
            for group_candidates in candidate_groups.values():
                selected = min(
                    group_candidates,
                    key=lambda index: (
                        float(np.linalg.norm(features[index] - centroid)),
                        record_rows[index]["record_number"],
                    ),
                )
                representatives.append(
                    {
                        "representation": representation,
                        "split": split,
                        "cluster": cluster,
                        **record_rows[selected],
                        "distance_to_train_centroid": float(
                            np.linalg.norm(features[selected] - centroid)
                        ),
                    }
                )
    return representatives


def _read_audio_payload(data_dir: Path, parquet_file: str, row_index: int) -> bytes:
    parquet_path = data_dir / parquet_file
    parquet_file_reader = pq.ParquetFile(parquet_path)
    row_offset = 0
    for row_group_index in range(parquet_file_reader.metadata.num_row_groups):
        row_group = parquet_file_reader.metadata.row_group(row_group_index)
        group_end = row_offset + row_group.num_rows
        if row_offset <= row_index < group_end:
            row_in_group = row_index - row_offset
            batch_offset = 0
            for batch in parquet_file_reader.iter_batches(
                batch_size=64,
                row_groups=[row_group_index],
                columns=["audio"],
            ):
                if batch_offset <= row_in_group < batch_offset + batch.num_rows:
                    position = row_in_group - batch_offset
                    audio = batch.column(0)[position].as_py()
                    payload = (audio or {}).get("bytes")
                    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
                        raise ValueError(
                            f"Missing audio bytes in {parquet_file} row {row_index}"
                        )
                    return bytes(payload)
                batch_offset += batch.num_rows
        row_offset = group_end
    raise IndexError(f"Parquet row index not found: {parquet_file}:{row_index}")


def export_cluster_previews(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    metadata_rows = _read_csv(output_dir / "be3c_embedding_chunks.csv")
    assignments = _read_csv(output_dir / "be3c_cluster_assignments.csv")
    with np.load(output_dir / "be3c_embedding_features.npz") as stored:
        clap_chunks = stored["clap_embeddings"]
        full_band_chunks = stored["full_band_features"]
    record_rows, clap_records, full_band_records = _aggregate_recordings(
        metadata_rows, clap_chunks, full_band_chunks
    )

    representatives: list[dict[str, Any]] = []
    representatives.extend(
        _choose_cluster_representatives(
            "clap", record_rows, clap_records, assignments
        )
    )
    representatives.extend(
        _choose_cluster_representatives(
            "full_band", record_rows, full_band_records, assignments
        )
    )
    sample_dir = output_dir / "be3c_cluster_examples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    preview_rows: list[dict[str, Any]] = []
    written: set[tuple[str, str]] = set()
    for representative in representatives:
        key = (
            representative["parquet_file"],
            representative["parquet_row_index"],
        )
        filename = f"record_{representative['record_number']}.wav"
        if key not in written:
            payload = _read_audio_payload(
                data_dir,
                representative["parquet_file"],
                int(representative["parquet_row_index"]),
            )
            with sf.SoundFile(BytesIO(payload)) as audio_file:
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
            sf.write(
                sample_dir / filename,
                waveform,
                sample_rate,
                subtype="PCM_16",
            )
            written.add(key)
        preview_rows.append(
            {
                "representation": representative["representation"],
                "split": representative["split"],
                "cluster": representative["cluster"],
                "record_number": representative["record_number"],
                "location_group": representative["location_group"],
                "source_sample_rate_hz": representative["source_sample_rate_hz"],
                "distance_to_train_centroid": (
                    representative["distance_to_train_centroid"]
                ),
                "wav_file": filename,
            }
        )

    manifest_path = output_dir / "be3c_cluster_examples.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(preview_rows[0]))
        writer.writeheader()
        writer.writerows(preview_rows)
    return {
        "representative_count": len(preview_rows),
        "unique_wav_count": len(written),
        "manifest_path": manifest_path,
        "sample_dir": sample_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export train and validation WAV representatives per cluster."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = export_cluster_previews(args.data_dir, args.output_dir)
    print(
        f"Wrote {result['representative_count']} representative entries "
        f"({result['unique_wav_count']} unique WAVs)."
    )
    print(f"Manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
