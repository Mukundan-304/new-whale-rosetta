from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import umap
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent
STAGE4_DIR = PROJECT_ROOT / "watkins_stage4"
SPLIT_DIR = PROJECT_ROOT / "watkins_splits"
SPLITS = ("train", "validation", "test")
LOGGER = logging.getLogger("watkins_stage6")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "stage6.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("fit_split") != "train":
        raise ValueError("Stage 6 clustering is restricted to the training split.")
    return config


def _load_and_validate_split(
    split: str, expected_dim: int
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    split_dir = STAGE4_DIR / split
    embedding_path = split_dir / "embeddings.npy"
    metadata_path = split_dir / "metadata.csv"
    completed_path = split_dir / "completed.npy"
    manifest_path = SPLIT_DIR / f"{split}.csv"

    embeddings = np.load(embedding_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path)
    if "segment_id" not in metadata or "recording_id" not in metadata:
        raise ValueError(f"{split} metadata must include segment_id and recording_id.")
    if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"{split} embedding shape {embeddings.shape} does not match "
            f"the expected dimension {expected_dim}."
        )
    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"{split} embedding/metadata row count mismatch: "
            f"{embeddings.shape[0]} vs {len(metadata)}."
        )

    ids = metadata["segment_id"].astype(str).to_numpy()
    if pd.isna(metadata["segment_id"]).any() or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{split} metadata has missing or duplicate segment IDs.")
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path, usecols=["segment_id"])
        manifest_ids = manifest["segment_id"].astype(str).to_numpy()
        if not np.array_equal(ids, manifest_ids):
            raise ValueError(f"{split} Stage 4 IDs do not match the split manifest.")

    if not np.isfinite(embeddings).all():
        raise ValueError(f"{split} embeddings contain NaN or infinite values.")
    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{split} embeddings contain zero-length vectors.")
    normalized = bool(np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4))
    if not normalized:
        vectors = vectors / norms[:, None]

    completed_count = len(vectors)
    if completed_path.exists():
        completed = np.load(completed_path, mmap_mode="r")
        if completed.shape != (len(vectors),) or not np.asarray(completed).all():
            raise ValueError(f"{split} Stage 4 completion mask is incomplete.")

    details = {
        "segments": int(len(vectors)),
        "embedding_dim": int(vectors.shape[1]),
        "all_finite": True,
        "input_l2_normalized": normalized,
        "normalized_for_analysis": not normalized,
        "manifest_ids_match": bool(manifest_path.exists()),
        "stage4_completion_count": completed_count,
        "embedding_path": str(embedding_path),
        "metadata_path": str(metadata_path),
    }
    return vectors, metadata, details


def _cluster_statistics(labels: np.ndarray) -> dict[str, Any]:
    counts = Counter(int(label) for label in labels)
    total = int(len(labels))
    noise_count = int(counts.get(-1, 0))
    cluster_ids = sorted(label for label in counts if label != -1)
    sizes = [
        {
            "cluster": int(cluster_id),
            "segments": int(counts[cluster_id]),
            "percentage": float(100.0 * counts[cluster_id] / total),
        }
        for cluster_id in cluster_ids
    ]
    return {
        "samples": total,
        "cluster_count": len(cluster_ids),
        "noise_count": noise_count,
        "noise_percentage": float(100.0 * noise_count / total) if total else 0.0,
        "assigned_count": total - noise_count,
        "assigned_percentage": (
            float(100.0 * (total - noise_count) / total) if total else 0.0
        ),
        "cluster_sizes": sizes,
        "noise_included_size_distribution": [
            {"cluster": int(label), "segments": int(counts[label])}
            for label in sorted(counts)
        ],
    }


def _silhouette(
    features: np.ndarray, labels: np.ndarray, seed: int, sample_size: int
) -> dict[str, Any]:
    eligible = np.flatnonzero(labels >= 0)
    unique_clusters = np.unique(labels[eligible])
    if len(unique_clusters) < 2 or len(eligible) <= len(unique_clusters):
        return {
            "score": None,
            "sampled_non_noise_points": 0,
            "note": "Silhouette is undefined with fewer than two populated clusters.",
        }

    rng = np.random.default_rng(seed)
    sample_count = min(sample_size, len(eligible))
    indices = np.sort(rng.choice(eligible, size=sample_count, replace=False))
    sampled_labels = labels[indices]
    if len(np.unique(sampled_labels)) < 2 or len(np.unique(sampled_labels)) >= len(
        sampled_labels
    ):
        return {
            "score": None,
            "sampled_non_noise_points": int(sample_count),
            "note": "The deterministic sample did not contain a valid label set.",
        }
    return {
        "score": float(
            silhouette_score(features[indices], sampled_labels, metric="euclidean")
        ),
        "sampled_non_noise_points": int(sample_count),
        "note": "Sampled non-noise points in the 15-D UMAP clustering space.",
    }


def _post_cluster_species_metrics(
    metadata: pd.DataFrame, labels: np.ndarray
) -> dict[str, Any]:
    if "species_label" not in metadata:
        return {"status": "unavailable", "reason": "species_label column is absent."}
    species = metadata["species_label"].fillna("").astype(str).to_numpy()
    eligible = (labels >= 0) & (species != "")
    if not eligible.any():
        return {"status": "unavailable", "reason": "No labeled clustered samples."}
    return {
        "status": "exploratory_post_clustering_only",
        "samples": int(eligible.sum()),
        "adjusted_rand_index": float(adjusted_rand_score(species[eligible], labels[eligible])),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(species[eligible], labels[eligible])
        ),
        "note": "Species labels were not passed to UMAP or HDBSCAN.",
    }


def _write_representatives(
    output_dir: Path,
    metadata: pd.DataFrame,
    cluster_space: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    per_cluster: int,
) -> int:
    requested_columns = [
        "segment_id",
        "recording_id",
        "species_label",
        "display_name",
        "scientific_name",
        "observation_date",
        "location_names",
        "location_domain",
        "audio_path",
        "audio_storage_type",
        "audio_parquet_row_index",
        "parquet_file",
        "parquet_row_index",
        "segment_start_seconds",
        "segment_valid_seconds",
    ]
    columns = [column for column in requested_columns if column in metadata]
    rows: list[dict[str, Any]] = []
    for cluster_id in sorted(int(label) for label in np.unique(labels) if label >= 0):
        indices = np.flatnonzero(labels == cluster_id)
        center = cluster_space[indices].mean(axis=0)
        distances = np.linalg.norm(cluster_space[indices] - center, axis=1)
        order = np.argsort(distances, kind="stable")[:per_cluster]
        for rank, local_index in enumerate(order, start=1):
            sample_index = int(indices[local_index])
            row = {column: metadata.iloc[sample_index][column] for column in columns}
            row.update(
                {
                    "cluster": cluster_id,
                    "cluster_size": int(len(indices)),
                    "representative_rank": rank,
                    "distance_to_cluster_centroid_15d": float(distances[local_index]),
                    "membership_probability": float(probabilities[sample_index]),
                }
            )
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "representative_samples.csv", index=False)
    return len(rows)


def _write_plots(
    output_dir: Path, coordinates: np.ndarray, labels: np.ndarray
) -> None:
    cluster_ids = sorted(int(label) for label in np.unique(labels) if label >= 0)
    palette = plt.get_cmap("turbo", max(1, len(cluster_ids)))
    colors = ["#777777"] + [palette(index) for index in range(len(cluster_ids))]
    colormap = ListedColormap(colors)
    cluster_position = {cluster: index + 1 for index, cluster in enumerate(cluster_ids)}
    plot_labels = np.asarray(
        [0 if label == -1 else cluster_position[int(label)] for label in labels]
    )
    boundaries = np.arange(-0.5, len(colors) + 0.5, 1)
    norm = BoundaryNorm(boundaries, colormap.N)

    figure, axis = plt.subplots(figsize=(11, 8), constrained_layout=True)
    points = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=plot_labels,
        cmap=colormap,
        norm=norm,
        s=5,
        alpha=0.7,
        linewidths=0,
        rasterized=True,
    )
    tick_positions = [0] + [
        index + 1 for index in range(0, len(cluster_ids), 10)
    ]
    if cluster_ids and tick_positions[-1] != len(cluster_ids):
        tick_positions.append(len(cluster_ids))
    tick_labels = ["noise"] + [
        str(cluster_ids[position - 1]) for position in tick_positions[1:]
    ]
    colorbar = figure.colorbar(points, ax=axis, ticks=tick_positions)
    colorbar.set_ticklabels(tick_labels)
    colorbar.set_label("HDBSCAN cluster ID")
    axis.set(title="Training embeddings coloured by HDBSCAN cluster", xlabel="UMAP 1", ylabel="UMAP 2")
    figure.savefig(output_dir / "umap_clusters.png", dpi=180)
    plt.close(figure)

    noise = labels == -1
    figure, axis = plt.subplots(figsize=(11, 8), constrained_layout=True)
    axis.scatter(
        coordinates[~noise, 0],
        coordinates[~noise, 1],
        c="#c7c7c7",
        s=5,
        alpha=0.45,
        linewidths=0,
        label="Assigned to cluster",
        rasterized=True,
    )
    axis.scatter(
        coordinates[noise, 0],
        coordinates[noise, 1],
        c="#d62728",
        s=8,
        alpha=0.8,
        linewidths=0,
        label="HDBSCAN noise",
        rasterized=True,
    )
    axis.set(title="HDBSCAN noise points highlighted", xlabel="UMAP 1", ylabel="UMAP 2")
    axis.legend(markerscale=2)
    figure.savefig(output_dir / "umap_noise.png", dpi=180)
    plt.close(figure)

    statistics = _cluster_statistics(labels)
    ordered = sorted(statistics["cluster_sizes"], key=lambda item: item["segments"], reverse=True)
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    x_values = np.arange(len(ordered) + int(statistics["noise_count"] > 0))
    counts = [item["segments"] for item in ordered]
    names = [str(item["cluster"]) for item in ordered]
    if statistics["noise_count"] > 0:
        counts.append(statistics["noise_count"])
        names.append("noise")
    bar_colors = ["#2878b5"] * len(ordered)
    if statistics["noise_count"] > 0:
        bar_colors.append("#d62728")
    axis.bar(x_values, counts, color=bar_colors)
    tick_positions = list(range(0, len(names), 10))
    if names and tick_positions[-1] != len(names) - 1:
        tick_positions.append(len(names) - 1)
    axis.set_xticks(
        x_values[tick_positions],
        [names[index] for index in tick_positions],
        rotation=70,
        ha="right",
    )
    axis.set(title="HDBSCAN cluster-size distribution", xlabel="Cluster ID", ylabel="Segments")
    figure.savefig(output_dir / "cluster_size_distribution.png", dpi=180)
    plt.close(figure)


def _write_report(
    output_dir: Path,
    started_at: str,
    completed_at: str,
    elapsed_seconds: float,
    split_details: dict[str, dict[str, Any]],
    config: dict[str, Any],
    statistics: dict[str, Any],
    silhouette: dict[str, Any],
    species_metrics: dict[str, Any],
    representatives: int,
) -> None:
    distribution_rows = "\n".join(
        f"| {item['cluster']} | {item['segments']:,} | {item['percentage']:.2f}% |"
        for item in sorted(statistics["cluster_sizes"], key=lambda item: item["cluster"])
    )
    if not distribution_rows:
        distribution_rows = "| (none) | 0 | 0.00% |"
    report = f"""# Stage 6 — Unsupervised acoustic pattern discovery

- Status: **complete**.
- Primary representation: Stage 4 `laion/clap-htsat-unfused` embeddings.
- Discovery fit split: **training only**; validation and test embeddings were validated but not clustered.
- Samples clustered: {statistics['samples']:,} training segments ({split_details['train']['embedding_dim']}-D CLAP embeddings).
- Execution duration: {elapsed_seconds:.1f} seconds (from {started_at} to {completed_at}).
- Random seed: {config['random_state']}.
- HDBSCAN clusters (noise excluded): {statistics['cluster_count']}.
- Noise: {statistics['noise_count']:,} segments ({statistics['noise_percentage']:.2f}%).
- Assigned to a cluster: {statistics['assigned_count']:,} segments ({statistics['assigned_percentage']:.2f}%).

## Input validation

| Split | Segments | Dimensions | Finite | L2-normalized on input | Manifest IDs aligned |
|---|---:|---:|---|---|---|
| train | {split_details['train']['segments']:,} | {split_details['train']['embedding_dim']} | yes | {split_details['train']['input_l2_normalized']} | {split_details['train']['manifest_ids_match']} |
| validation | {split_details['validation']['segments']:,} | {split_details['validation']['embedding_dim']} | yes | {split_details['validation']['input_l2_normalized']} | {split_details['validation']['manifest_ids_match']} |
| test | {split_details['test']['segments']:,} | {split_details['test']['embedding_dim']} | yes | {split_details['test']['input_l2_normalized']} | {split_details['test']['manifest_ids_match']} |

Non-unit vectors, if any, were L2-normalized in memory for analysis; no Stage 4 or Stage 5 files were modified.

## UMAP and HDBSCAN settings

- Clustering space: UMAP to {config['umap_clustering']['n_components']} dimensions, cosine metric, `n_neighbors={config['umap_clustering']['n_neighbors']}`, `min_dist={config['umap_clustering']['min_dist']}`.
- Visualization: separate 2-D UMAP, cosine metric, `n_neighbors={config['umap_visualization']['n_neighbors']}`, `min_dist={config['umap_visualization']['min_dist']}`.
- HDBSCAN: `min_cluster_size={config['hdbscan']['min_cluster_size']}`, `min_samples={config['hdbscan']['min_samples']}`, metric `{config['hdbscan']['metric']}`, selection `{config['hdbscan']['cluster_selection_method']}`.
- UMAP/HDBSCAN saw only the unlabeled training embedding matrix; validation and test were not used for clustering or parameter selection.

## Cluster sizes

| Cluster ID | Segments | Share of training split |
|---:|---:|---:|
{distribution_rows}

Noise is reported separately from the clusters. The cluster assignment file contains a row for every training segment.

## Evaluation

- Sampled non-noise silhouette in the 15-D UMAP space: {silhouette['score'] if silhouette['score'] is not None else 'not available'} ({silhouette['sampled_non_noise_points']:,} points). {silhouette['note']}
- Post-clustering species ARI/NMI: {species_metrics.get('adjusted_rand_index', 'not available')} / {species_metrics.get('normalized_mutual_info', 'not available')} on {species_metrics.get('samples', 0):,} clustered labeled training samples. This is exploratory external evaluation; species labels were not inputs to clustering.

## Representative samples

`representative_samples.csv` contains up to {config['representatives_per_cluster']} training examples per cluster, chosen as nearest members to each cluster centroid in the 15-D UMAP space. Rows include segment and recording IDs plus location/date and audio pointers where available. In this dataset, audio is embedded in Parquet; the listed Parquet path/row and segment offset identify the source but are not standalone WAV files.

## Outputs

- `cluster_assignments.csv` — train IDs, cluster IDs, HDBSCAN membership probabilities, and 2-D coordinates.
- `umap_coordinates.csv` and `umap_clustering_coordinates.npy` — visualization and 15-D clustering coordinates.
- `cluster_sizes.csv`, `representative_samples.csv`, and `run_summary.json`.
- `umap_clusters.png`, `umap_noise.png`, and `cluster_size_distribution.png`.
- `stage6_config.json` and `stage6.log`.
- Representative rows written: {representatives:,}.

## Interpretation and limitations

These are **candidate acoustic patterns**, not confirmed biological categories, species, communication signals, or meanings. UMAP changes neighborhood geometry and HDBSCAN results depend on its parameters; clusters/noise should be treated as exploratory. The embeddings are CLAP representations (512 dimensions, with the Stage 4 frontend's frequency limitations), and clustering was performed only on the training split. No Stage 4/5 retraining, embedding regeneration, preprocessing, or split modification was performed.

## Reproducibility

Start time: `{started_at}`. Completion time: `{completed_at}`. Total execution: `{elapsed_seconds:.1f}` seconds. Python `{platform.python_version()}`, scikit-learn `{sklearn.__version__}`, UMAP `{umap.__version__}`. Parameters and seed are recorded in `stage6_config.json` and `run_summary.json`.
"""
    (output_dir / "stage6_report.md").write_text(report, encoding="utf-8")


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.resolve() == STAGE4_DIR.resolve() or output_dir.resolve().is_relative_to(
        STAGE4_DIR.resolve()
    ):
        raise ValueError("Stage 6 output must not be inside the Stage 4 directory.")
    stage5_dir = PROJECT_ROOT / "watkins_stage5"
    if output_dir.resolve() == stage5_dir.resolve() or output_dir.resolve().is_relative_to(
        stage5_dir.resolve()
    ):
        raise ValueError("Stage 6 output must not be inside the Stage 5 directory.")

    _configure_logging(output_dir)
    config = _load_config(config_path)
    seed = int(config["random_state"])
    np.random.seed(seed)
    started_at = _utc_now()
    timer = time.perf_counter()
    split_details: dict[str, dict[str, Any]] = {}
    matrices: dict[str, np.ndarray] = {}
    metadata_by_split: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        matrix, metadata, details = _load_and_validate_split(
            split, int(config["expected_embedding_dim"])
        )
        matrices[split] = matrix
        metadata_by_split[split] = metadata
        split_details[split] = details
        LOGGER.info(
            "Validated split=%s samples=%d dimensions=%d",
            split,
            details["segments"],
            details["embedding_dim"],
        )

    train = matrices["train"]
    train_metadata = metadata_by_split["train"]

    clustering_config = config["umap_clustering"]
    LOGGER.info("Fitting %d-D UMAP on %d training embeddings", clustering_config["n_components"], len(train))
    clustering_reducer = umap.UMAP(
        n_components=int(clustering_config["n_components"]),
        n_neighbors=int(clustering_config["n_neighbors"]),
        min_dist=float(clustering_config["min_dist"]),
        metric=str(clustering_config["metric"]),
        random_state=seed,
        n_jobs=1,
        low_memory=True,
        compatibility_layout=True,
    )
    cluster_space = np.asarray(clustering_reducer.fit_transform(train), dtype=np.float32)
    if not np.isfinite(cluster_space).all():
        raise ValueError("15-D UMAP output contains NaN or infinite values.")
    np.save(output_dir / "umap_clustering_coordinates.npy", cluster_space)
    LOGGER.info("Fitting HDBSCAN in 15-D UMAP space")
    hdbscan_config = config["hdbscan"]
    clusterer = HDBSCAN(
        min_cluster_size=int(hdbscan_config["min_cluster_size"]),
        min_samples=int(hdbscan_config["min_samples"]),
        metric=str(hdbscan_config["metric"]),
        cluster_selection_method=str(hdbscan_config["cluster_selection_method"]),
        n_jobs=1,
        copy=False,
    )
    labels = clusterer.fit_predict(cluster_space).astype(np.int32, copy=False)
    probabilities = np.asarray(clusterer.probabilities_, dtype=np.float32)
    if labels.shape != (len(train),) or probabilities.shape != labels.shape:
        raise ValueError("HDBSCAN returned assignment arrays with an unexpected shape.")
    if not np.isfinite(probabilities).all():
        raise ValueError("HDBSCAN returned non-finite membership probabilities.")
    statistics = _cluster_statistics(labels)
    LOGGER.info(
        "HDBSCAN complete clusters=%d noise=%d (%.2f%%) assigned=%d",
        statistics["cluster_count"],
        statistics["noise_count"],
        statistics["noise_percentage"],
        statistics["assigned_count"],
    )

    LOGGER.info("Fitting 2-D UMAP for visualization")
    visualization_config = config["umap_visualization"]
    visualization_reducer = umap.UMAP(
        n_components=int(visualization_config["n_components"]),
        n_neighbors=int(visualization_config["n_neighbors"]),
        min_dist=float(visualization_config["min_dist"]),
        metric=str(visualization_config["metric"]),
        random_state=seed,
        n_jobs=1,
        low_memory=True,
        compatibility_layout=True,
    )
    coordinates = np.asarray(visualization_reducer.fit_transform(train), dtype=np.float32)
    if not np.isfinite(coordinates).all():
        raise ValueError("2-D UMAP output contains NaN or infinite values.")

    assignments = pd.DataFrame(
        {
            "segment_id": train_metadata["segment_id"].astype(str),
            "recording_id": train_metadata["recording_id"].astype(str),
            "cluster": labels,
            "membership_probability": probabilities,
            "umap_1": coordinates[:, 0],
            "umap_2": coordinates[:, 1],
        }
    )
    assignments.to_csv(output_dir / "cluster_assignments.csv", index=False)
    pd.DataFrame(
        {
            "segment_id": train_metadata["segment_id"].astype(str),
            "umap_1": coordinates[:, 0],
            "umap_2": coordinates[:, 1],
            "cluster": labels,
        }
    ).to_csv(output_dir / "umap_coordinates.csv", index=False)

    cluster_sizes = statistics["noise_included_size_distribution"]
    pd.DataFrame(cluster_sizes).to_csv(output_dir / "cluster_sizes.csv", index=False)
    representatives = _write_representatives(
        output_dir,
        train_metadata,
        cluster_space,
        labels,
        probabilities,
        int(config["representatives_per_cluster"]),
    )
    _write_plots(output_dir, coordinates, labels)
    silhouette = _silhouette(
        cluster_space,
        labels,
        seed,
        int(config["silhouette_sample_size"]),
    )
    species_metrics = _post_cluster_species_metrics(train_metadata, labels)
    completed_at = _utc_now()
    elapsed_seconds = time.perf_counter() - timer

    summary = {
        "status": "complete",
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "execution_seconds": float(elapsed_seconds),
        "fit_split": "train",
        "validated_splits": split_details,
        "total_validated_segments": int(
            sum(details["segments"] for details in split_details.values())
        ),
        "embedding_model": "laion/clap-htsat-unfused",
        "embedding_dimensions": int(train.shape[1]),
        "config": config,
        "hdbscan_summary": statistics,
        "silhouette": silhouette,
        "post_cluster_species_metrics": species_metrics,
        "representative_samples_written": representatives,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "umap_learn": umap.__version__,
        },
        "outputs": [
            "cluster_assignments.csv",
            "cluster_sizes.csv",
            "representative_samples.csv",
            "umap_coordinates.csv",
            "umap_clustering_coordinates.npy",
            "umap_clusters.png",
            "umap_noise.png",
            "cluster_size_distribution.png",
            "stage6_report.md",
            "run_summary.json",
        ],
        "species_label_use": "post-clustering exploratory metrics only",
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    _write_report(
        output_dir,
        started_at,
        completed_at,
        elapsed_seconds,
        split_details,
        config,
        statistics,
        silhouette,
        species_metrics,
        representatives,
    )
    LOGGER.info(
        "Stage 6 complete elapsed=%.1fs output=%s",
        elapsed_seconds,
        output_dir,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 6 UMAP + HDBSCAN discovery.")
    parser.add_argument(
        "--config",
        type=Path,
        default=OUTPUT_DIR / "stage6_config.json",
        help="Path to Stage 6 JSON configuration.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    run(arguments.config, arguments.output_dir)


if __name__ == "__main__":
    main()
