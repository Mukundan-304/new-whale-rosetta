"""Fit train-only clustering diagnostics for BE3C baseline features."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "be3c_feasibility"
MAX_CLUSTERS = 10
RANDOM_STATE = 17


def _read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _aggregate_recordings(
    metadata_rows: list[dict[str, str]],
    clap_chunks: np.ndarray,
    full_band_chunks: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    if len(metadata_rows) != len(clap_chunks) or len(metadata_rows) != len(
        full_band_chunks
    ):
        raise ValueError("Chunk metadata and feature counts must match")
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(metadata_rows):
        key = (row["parquet_file"], row["parquet_row_index"])
        grouped[key].append(index)

    record_rows: list[dict[str, Any]] = []
    clap_records: list[np.ndarray] = []
    full_band_records: list[np.ndarray] = []
    for key, indices in grouped.items():
        chunks = [metadata_rows[index] for index in indices]
        invariant_fields = ("record_number", "split", "location_group")
        for field in invariant_fields:
            if len({row[field] for row in chunks}) != 1:
                raise ValueError(f"Chunks for {key} disagree on {field}")
        clap_mean = np.mean(clap_chunks[indices], axis=0)
        norm = float(np.linalg.norm(clap_mean))
        if norm == 0:
            raise ValueError(f"CLAP feature mean is zero for source row {key}")
        clap_records.append((clap_mean / norm).astype(np.float32))
        full_band_records.append(
            np.mean(full_band_chunks[indices], axis=0).astype(np.float32)
        )
        first = chunks[0]
        record_rows.append(
            {
                "species_code": first["species_code"],
                "record_number": first["record_number"],
                "split": first["split"],
                "location_group": first["location_group"],
                "parquet_file": first["parquet_file"],
                "parquet_row_index": first["parquet_row_index"],
                "source_sample_rate_hz": first["source_sample_rate_hz"],
                "chunk_count": len(indices),
                "spectral_power_fraction_above_14khz": first.get(
                    "spectral_power_fraction_above_14khz", ""
                ),
            }
        )
    return (
        record_rows,
        np.stack(clap_records),
        np.stack(full_band_records),
    )


def _cluster_representation(
    representation: str,
    features: np.ndarray,
    record_rows: list[dict[str, Any]],
    max_clusters: int = MAX_CLUSTERS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(features) != len(record_rows):
        raise ValueError("Record metadata and feature counts must match")
    train_indices = [
        index for index, row in enumerate(record_rows) if row["split"] == "train"
    ]
    validation_indices = [
        index for index, row in enumerate(record_rows) if row["split"] == "validation"
    ]
    if len(train_indices) < 3 or not validation_indices:
        raise ValueError("Need at least three train rows and one validation row")

    if representation == "full_band":
        scaler = StandardScaler()
        train_features = scaler.fit_transform(features[train_indices])
        validation_features = scaler.transform(features[validation_indices])
    else:
        train_features = features[train_indices]
        validation_features = features[validation_indices]

    upper_cluster_count = min(max_clusters, len(train_indices) - 1)
    scores: list[dict[str, Any]] = []
    candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    train_groups = [record_rows[index]["location_group"] for index in train_indices]
    train_rates = [
        record_rows[index]["source_sample_rate_hz"] for index in train_indices
    ]
    validation_groups = [
        record_rows[index]["location_group"] for index in validation_indices
    ]
    validation_rates = [
        record_rows[index]["source_sample_rate_hz"] for index in validation_indices
    ]
    for cluster_count in range(2, upper_cluster_count + 1):
        model = KMeans(
            n_clusters=cluster_count,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        train_labels = model.fit_predict(train_features)
        validation_labels = model.predict(validation_features)
        score = float(silhouette_score(train_features, train_labels))
        scores.append(
            {
                "representation": representation,
                "cluster_count": cluster_count,
                "train_silhouette": score,
                "train_location_nmi": normalized_mutual_info_score(
                    train_groups, train_labels
                ),
                "train_sample_rate_nmi": normalized_mutual_info_score(
                    train_rates, train_labels
                ),
                "validation_location_nmi": normalized_mutual_info_score(
                    validation_groups, validation_labels
                ),
                "validation_sample_rate_nmi": normalized_mutual_info_score(
                    validation_rates, validation_labels
                ),
            }
        )
        candidates.append(
            (
                score,
                -cluster_count,
                train_labels,
                validation_labels,
            )
        )

    _, negative_count, train_labels, validation_labels = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    selected_count = -negative_count
    assignments: list[dict[str, Any]] = []
    for indices, labels in (
        (train_indices, train_labels),
        (validation_indices, validation_labels),
    ):
        for record_index, cluster in zip(indices, labels):
            assignments.append(
                {
                    "representation": representation,
                    "selected_cluster_count": selected_count,
                    "record_number": record_rows[record_index]["record_number"],
                    "split": record_rows[record_index]["split"],
                    "location_group": record_rows[record_index]["location_group"],
                    "source_sample_rate_hz": record_rows[record_index][
                        "source_sample_rate_hz"
                    ],
                    "chunk_count": record_rows[record_index]["chunk_count"],
                    "spectral_power_fraction_above_14khz": record_rows[
                        record_index
                    ]["spectral_power_fraction_above_14khz"],
                    "cluster": int(cluster),
                }
            )
    if len(assignments) != len(record_rows):
        raise RuntimeError("Not every source recording received a cluster")
    return scores, assignments


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty report: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cluster_be3c_baseline(
    output_dir: Path = OUTPUT_DIR,
    max_clusters: int = MAX_CLUSTERS,
) -> dict[str, Any]:
    if max_clusters < 2:
        raise ValueError("max_clusters must be at least 2")
    metadata_path = output_dir / "be3c_embedding_chunks.csv"
    features_path = output_dir / "be3c_embedding_features.npz"
    if not metadata_path.is_file() or not features_path.is_file():
        raise FileNotFoundError(
            "Run be3c_embedding_baseline.py before clustering its features"
        )
    metadata_rows = _read_metadata(metadata_path)
    with np.load(features_path) as stored:
        clap_chunks = stored["clap_embeddings"]
        full_band_chunks = stored["full_band_features"]
    record_rows, clap_records, full_band_records = _aggregate_recordings(
        metadata_rows, clap_chunks, full_band_chunks
    )

    all_scores: list[dict[str, Any]] = []
    all_assignments: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    for representation, features in (
        ("clap", clap_records),
        ("full_band", full_band_records),
    ):
        scores, assignments = _cluster_representation(
            representation,
            features,
            record_rows,
            max_clusters=max_clusters,
        )
        all_scores.extend(scores)
        all_assignments.extend(assignments)
        selected_counts[representation] = next(
            row["selected_cluster_count"]
            for row in assignments
            if row["split"] == "train"
        )

    score_path = output_dir / "be3c_cluster_scores.csv"
    assignment_path = output_dir / "be3c_cluster_assignments.csv"
    _write_csv(score_path, all_scores)
    _write_csv(assignment_path, all_assignments)
    selected_scores = {
        representation: next(
            row
            for row in all_scores
            if row["representation"] == representation
            and row["cluster_count"] == selected_counts[representation]
        )
        for representation in selected_counts
    }
    cluster_sizes = {
        (representation, split): Counter(
            row["cluster"]
            for row in all_assignments
            if row["representation"] == representation and row["split"] == split
        )
        for representation in selected_counts
        for split in ("train", "validation")
    }
    report_lines = [
        "# BE3C unsupervised clustering baseline",
        "",
        f"- Record-level features: {len(record_rows):,} source recordings; chunk "
        "features were averaged per original recording to avoid giving longer "
        "recordings extra clustering weight.",
        "- K-means was fitted using training records only; validation records were "
        "assigned to the nearest training centroid.",
        "- The number of clusters was selected separately per representation by "
        "maximum training-only silhouette score among the tested values.",
        f"- Selected cluster counts: CLAP {selected_counts['clap']}; "
        f"50 Hz–24 kHz acoustic features {selected_counts['full_band']}.",
        f"- Training silhouette scores at the selected counts: CLAP "
        f"{selected_scores['clap']['train_silhouette']:.3f}; full-band "
        f"{selected_scores['full_band']['train_silhouette']:.3f}.",
        f"- Train location/sample-rate NMI: CLAP "
        f"{selected_scores['clap']['train_location_nmi']:.3f}/"
        f"{selected_scores['clap']['train_sample_rate_nmi']:.3f}; full-band "
        f"{selected_scores['full_band']['train_location_nmi']:.3f}/"
        f"{selected_scores['full_band']['train_sample_rate_nmi']:.3f}.",
        f"- Validation location/sample-rate NMI: CLAP "
        f"{selected_scores['clap']['validation_location_nmi']:.3f}/"
        f"{selected_scores['clap']['validation_sample_rate_nmi']:.3f}; full-band "
        f"{selected_scores['full_band']['validation_location_nmi']:.3f}/"
        f"{selected_scores['full_band']['validation_sample_rate_nmi']:.3f}.",
        f"- Train cluster sizes (cluster 0/1): CLAP "
        f"{cluster_sizes[('clap', 'train')].get(0, 0)}/"
        f"{cluster_sizes[('clap', 'train')].get(1, 0)}; full-band "
        f"{cluster_sizes[('full_band', 'train')].get(0, 0)}/"
        f"{cluster_sizes[('full_band', 'train')].get(1, 0)}.",
        f"- Validation cluster sizes (cluster 0/1): CLAP "
        f"{cluster_sizes[('clap', 'validation')].get(0, 0)}/"
        f"{cluster_sizes[('clap', 'validation')].get(1, 0)}; full-band "
        f"{cluster_sizes[('full_band', 'validation')].get(0, 0)}/"
        f"{cluster_sizes[('full_band', 'validation')].get(1, 0)}.",
        "- Location and sample-rate normalized mutual information values are "
        "diagnostics for recording-condition association, not model targets.",
        "- This focused set contains one species, so it cannot assess common, "
        "moderate, and rare species performance. Treat clusters as candidate "
        "acoustic patterns, not confirmed call types.",
        "",
        "See `be3c_cluster_scores.csv` for every tested cluster count and "
        "`be3c_cluster_assignments.csv` for the selected train/validation clusters. "
        "Run `make_be3c_cluster_previews.py` to export representative WAVs listed "
        "in `be3c_cluster_examples.csv`.",
        "",
    ]
    report_path = output_dir / "be3c_clustering_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "record_count": len(record_rows),
        "selected_cluster_counts": selected_counts,
        "scores_path": score_path,
        "assignments_path": assignment_path,
        "report_path": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run train-only K-means diagnostics on BE3C features."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-clusters", type=int, default=MAX_CLUSTERS)
    args = parser.parse_args()
    result = cluster_be3c_baseline(
        output_dir=args.output_dir,
        max_clusters=args.max_clusters,
    )
    print(f"Clustered {result['record_count']:,} source recordings.")
    print(f"Selected cluster counts: {result['selected_cluster_counts']}")
    print(f"Report: {result['report_path']}")


if __name__ == "__main__":
    main()
