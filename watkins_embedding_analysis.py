"""Inspect frozen Watkins embeddings and create train-only exploratory analyses."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from watkins_pretrained_embeddings import _atomic_json, _read_json


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "watkins_stage4"
LOGGER = logging.getLogger("watkins_stage4.analysis")
SPLITS = ("train", "validation", "test")
RANDOM_STATE = 29


def _read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, Any]]
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_embeddings(
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, str]]], dict[str, Any]]:
    matrices: dict[str, np.ndarray] = {}
    metadata: dict[str, list[dict[str, str]]] = {}
    seen_ids: set[str] = set()
    per_split: dict[str, Any] = {}
    dimensions: set[int] = set()
    for split in SPLITS:
        matrix = np.load(output_dir / split / "embeddings.npy", mmap_mode="r")
        rows = _read_metadata(output_dir / split / "metadata.csv")
        if matrix.ndim != 2 or len(matrix) != len(rows):
            raise ValueError(
                f"{split} matrix/metadata mismatch: {matrix.shape} vs {len(rows)}"
            )
        if matrix.dtype != np.float32:
            raise ValueError(f"{split} embeddings must be float32, got {matrix.dtype}")
        ids = [row["segment_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{split} metadata has duplicate segment IDs")
        overlaps = seen_ids & set(ids)
        if overlaps:
            raise ValueError(
                f"Segment IDs cross split boundaries: {sorted(overlaps)[:5]}"
            )
        seen_ids.update(ids)
        dimensions.add(int(matrix.shape[1]))
        finite = np.isfinite(matrix).all(axis=1)
        norms = np.linalg.norm(matrix, axis=1)
        valid = finite & np.isfinite(norms) & (norms > 0)
        if not valid.all():
            raise ValueError(
                f"{split} has {int((~valid).sum())} invalid embeddings"
            )
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError(f"{split} embeddings are not L2-normalized")
        for row in rows:
            if row["split"] != split:
                raise ValueError(
                    f"{split} metadata contains row from split {row['split']!r}"
                )
            if not Path(row["audio_path"]).is_file():
                raise FileNotFoundError(
                    f"Audio path no longer exists for {row['segment_id']}: "
                    f"{row['audio_path']}"
                )
        matrices[split] = matrix
        metadata[split] = rows
        per_split[split] = {
            "segments": len(rows),
            "recordings": len({row["recording_id"] for row in rows}),
            "species": len({row["species_label"] for row in rows}),
            "embedding_shape": list(matrix.shape),
            "invalid_vectors": int((~valid).sum()),
            "duplicate_segment_ids": 0,
            "cross_split_segment_id_overlap": 0,
            "metadata_matches_embeddings": len(rows) == len(matrix),
            "all_audio_paths_valid": True,
        }
    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimensions differ between splits: {dimensions}")
    return matrices, metadata, {
        "status": "passed",
        "total_segments": len(seen_ids),
        "embedding_dimensions": dimensions.pop(),
        "unique_segment_ids": len(seen_ids),
        "split_id_overlap": False,
        "splits": per_split,
    }


def _similarity_analysis(
    embeddings: np.ndarray,
    rows: list[dict[str, str]],
    pairs_per_group: int,
    seed: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    by_species: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(rows):
        by_species[row["species_label"]][row["recording_id"]].append(index)
    eligible = {
        species: recordings
        for species, recordings in by_species.items()
        if len(recordings) >= 2
    }
    if len(eligible) < 2:
        raise ValueError("Need at least two species with multiple train recordings")

    generator = np.random.default_rng(seed)
    same_pairs: list[tuple[int, int, str, str]] = []
    species_names = sorted(eligible)
    for _ in range(pairs_per_group):
        species = species_names[int(generator.integers(len(species_names)))]
        recording_ids = sorted(eligible[species])
        selected = generator.choice(recording_ids, size=2, replace=False)
        first_recording, second_recording = str(selected[0]), str(selected[1])
        first = eligible[species][first_recording][
            int(generator.integers(len(eligible[species][first_recording])))
        ]
        second = eligible[species][second_recording][
            int(generator.integers(len(eligible[species][second_recording])))
        ]
        same_pairs.append((first, second, species, species))

    different_pairs: list[tuple[int, int, str, str]] = []
    for _ in range(pairs_per_group):
        selected_species = generator.choice(species_names, size=2, replace=False)
        first_species, second_species = str(selected_species[0]), str(selected_species[1])
        first_recordings = sorted(eligible[first_species])
        second_recordings = sorted(eligible[second_species])
        first_recording = str(
            generator.choice(first_recordings)
        )
        second_recording = str(
            generator.choice(second_recordings)
        )
        first = eligible[first_species][first_recording][
            int(generator.integers(len(eligible[first_species][first_recording])))
        ]
        second = eligible[second_species][second_recording][
            int(generator.integers(len(eligible[second_species][second_recording])))
        ]
        different_pairs.append(
            (first, second, first_species, second_species)
        )

    output_rows: list[dict[str, Any]] = []
    for same_species, pairs in ((True, same_pairs), (False, different_pairs)):
        for first, second, first_species, second_species in pairs:
            output_rows.append(
                {
                    "same_species": same_species,
                    "segment_id_a": rows[first]["segment_id"],
                    "segment_id_b": rows[second]["segment_id"],
                    "recording_id_a": rows[first]["recording_id"],
                    "recording_id_b": rows[second]["recording_id"],
                    "species_a": first_species,
                    "species_b": second_species,
                    "cosine_similarity": float(
                        np.dot(embeddings[first], embeddings[second])
                    ),
                }
            )
    _write_csv(
        output_dir / "cosine_similarity_pairs.csv",
        [
            "same_species",
            "segment_id_a",
            "segment_id_b",
            "recording_id_a",
            "recording_id_b",
            "species_a",
            "species_b",
            "cosine_similarity",
        ],
        output_rows,
    )
    summaries = []
    for same_species, label in ((True, "same_species"), (False, "different_species")):
        values = np.asarray(
            [
                row["cosine_similarity"]
                for row in output_rows
                if row["same_species"] is same_species
            ],
            dtype=np.float32,
        )
        summaries.append(
            {
                "comparison": label,
                "pair_count": len(values),
                "mean_cosine_similarity": float(values.mean()),
                "median_cosine_similarity": float(np.median(values)),
                "p05_cosine_similarity": float(np.quantile(values, 0.05)),
                "p95_cosine_similarity": float(np.quantile(values, 0.95)),
            }
        )
    _write_csv(
        output_dir / "cosine_similarity_summary.csv",
        list(summaries[0]),
        summaries,
    )
    return summaries


def _pca_analysis(
    matrices: dict[str, np.ndarray],
    metadata: dict[str, list[dict[str, str]]],
    seed: int,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    train = np.asarray(matrices["train"], dtype=np.float32)
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(train)
    coordinates = {
        split: pca.transform(np.asarray(matrices[split], dtype=np.float32))
        for split in SPLITS
    }
    coordinate_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for row, point in zip(metadata[split], coordinates[split]):
            coordinate_rows.append(
                {
                    "segment_id": row["segment_id"],
                    "recording_id": row["recording_id"],
                    "species_label": row["species_label"],
                    "split": split,
                    "pca_1": float(point[0]),
                    "pca_2": float(point[1]),
                }
            )
    _write_csv(
        output_dir / "pca_coordinates.csv",
        [
            "segment_id",
            "recording_id",
            "species_label",
            "split",
            "pca_1",
            "pca_2",
        ],
        coordinate_rows,
    )

    all_train_labels = [row["species_label"] for row in metadata["train"]]
    top_species = [
        species for species, _ in Counter(all_train_labels).most_common(12)
    ]
    plot_indices = np.arange(len(all_train_labels))
    if len(plot_indices) > 8000:
        plot_indices = np.random.default_rng(seed).choice(
            plot_indices, size=8000, replace=False
        )
    train_labels = np.asarray(all_train_labels, dtype=object)[plot_indices]
    train_color_labels = np.asarray(
        [label if label in top_species else "Other" for label in train_labels],
        dtype=object,
    )
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for label in sorted(set(train_color_labels)):
        mask = train_color_labels == label
        axes[0].scatter(
            coordinates["train"][plot_indices[mask], 0],
            coordinates["train"][plot_indices[mask], 1],
            s=5,
            alpha=0.55,
            label=label,
        )
    axes[0].set_title("PCA of training embeddings (top 12 species)")
    axes[0].legend(markerscale=2, fontsize=7, ncol=2)
    for split, color in (
        ("train", "#4C78A8"),
        ("validation", "#F58518"),
        ("test", "#54A24B"),
    ):
        split_points = coordinates[split]
        if split == "train" and len(split_points) > 8000:
            selected = np.random.default_rng(seed + 1).choice(
                len(split_points), size=8000, replace=False
            )
            split_points = split_points[selected]
        axes[1].scatter(
            split_points[:, 0],
            split_points[:, 1],
            s=4,
            alpha=0.4,
            color=color,
            label=split,
        )
    axes[1].set_title("PCA projection by split (fit on train only)")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("PC 1")
        axis.set_ylabel("PC 2")
    figure.savefig(output_dir / "pca_embeddings.png", dpi=180)
    plt.close(figure)
    return {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "coordinates": coordinates,
    }


def _umap_analysis(
    matrices: dict[str, np.ndarray],
    metadata: dict[str, list[dict[str, str]]],
    seed: int,
    sample_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        import umap
    except ImportError as error:
        LOGGER.warning("UMAP unavailable; skipping UMAP: %s", error)
        return {
            "status": "skipped",
            "reason": "Install umap-learn to enable the optional UMAP projection.",
        }

    train = np.asarray(matrices["train"], dtype=np.float32)
    if len(train) < 4:
        return {"status": "skipped", "reason": "Fewer than four train embeddings."}
    generator = np.random.default_rng(seed)
    train_indices = np.arange(len(train))
    if len(train_indices) > sample_size:
        train_indices = np.sort(
            generator.choice(train_indices, size=sample_size, replace=False)
        )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
        n_jobs=1,
        low_memory=True,
    )
    train_sample = train[train_indices]
    train_coordinates = reducer.fit_transform(train_sample)
    labels = [metadata["train"][int(index)]["species_label"] for index in train_indices]
    top_species = [
        species for species, _ in Counter(labels).most_common(12)
    ]
    colors = np.asarray(
        [label if label in top_species else "Other" for label in labels],
        dtype=object,
    )
    figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for label in sorted(set(colors)):
        mask = colors == label
        axis.scatter(
            train_coordinates[mask, 0],
            train_coordinates[mask, 1],
            s=6,
            alpha=0.55,
            label=label,
        )
    axis.set_title(f"UMAP of {len(train_indices):,} training embeddings")
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.legend(markerscale=2, fontsize=7, ncol=2)
    figure.savefig(output_dir / "umap_train_embeddings.png", dpi=180)
    plt.close(figure)
    coordinate_rows = []
    for index, point in zip(train_indices, train_coordinates):
        row = metadata["train"][int(index)]
        coordinate_rows.append(
            {
                "segment_id": row["segment_id"],
                "recording_id": row["recording_id"],
                "species_label": row["species_label"],
                "umap_1": float(point[0]),
                "umap_2": float(point[1]),
            }
        )
    _write_csv(
        output_dir / "umap_train_coordinates.csv",
        [
            "segment_id",
            "recording_id",
            "species_label",
            "umap_1",
            "umap_2",
        ],
        coordinate_rows,
    )
    return {
        "status": "complete",
        "fit_split": "train",
        "sample_size": len(train_indices),
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "cosine",
    }


def _cluster_analysis(
    train: np.ndarray,
    metadata: list[dict[str, str]],
    cluster_count: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    if len(train) <= cluster_count:
        raise ValueError("Training set is too small for configured K-Means clusters")
    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=seed,
        n_init=5,
        batch_size=min(2048, len(train)),
        max_iter=100,
        reassignment_ratio=0.01,
    )
    labels = model.fit_predict(np.asarray(train, dtype=np.float32))
    assignment_rows = [
        {
            "segment_id": row["segment_id"],
            "recording_id": row["recording_id"],
            "species_label": row["species_label"],
            "cluster": int(cluster),
        }
        for row, cluster in zip(metadata, labels)
    ]
    _write_csv(
        output_dir / "train_cluster_assignments.csv",
        ["segment_id", "recording_id", "species_label", "cluster"],
        assignment_rows,
    )
    species = [row["species_label"] for row in metadata]
    cluster_rows: list[dict[str, Any]] = []
    for cluster in range(cluster_count):
        members = [
            row for row in assignment_rows if row["cluster"] == cluster
        ]
        counts = Counter(row["species_label"] for row in members)
        cluster_rows.append(
            {
                "cluster": cluster,
                "segment_count": len(members),
                "recording_count": len({row["recording_id"] for row in members}),
                "top_species": " | ".join(
                    f"{label}:{count}" for label, count in counts.most_common(5)
                ),
                "top_species_fraction": (
                    float(counts.most_common(1)[0][1] / len(members))
                    if members
                    else 0.0
                ),
            }
        )
    _write_csv(
        output_dir / "train_cluster_species_composition.csv",
        [
            "cluster",
            "segment_count",
            "recording_count",
            "top_species",
            "top_species_fraction",
        ],
        cluster_rows,
    )
    sample_size = min(3000, len(train))
    sample_indices = np.random.default_rng(seed).choice(
        len(train), size=sample_size, replace=False
    )
    silhouette = silhouette_score(
        np.asarray(train[sample_indices], dtype=np.float32),
        labels[sample_indices],
        metric="euclidean",
    )
    return {
        "method": "MiniBatchKMeans",
        "fit_split": "train",
        "cluster_count": cluster_count,
        "sampled_train_silhouette": float(silhouette),
        "train_species_nmi": float(normalized_mutual_info_score(species, labels)),
        "train_species_ari": float(adjusted_rand_score(species, labels)),
        "clusters": cluster_rows,
        "interpretation_note": (
            "Exploratory acoustic grouping only; clusters do not demonstrate "
            "communication patterns or animal language."
        ),
    }


def _write_report(
    output_dir: Path,
    generation: dict[str, Any],
    validation: dict[str, Any],
    similarity: list[dict[str, Any]],
    pca_result: dict[str, Any],
    umap_result: dict[str, Any],
    cluster_result: dict[str, Any],
    analysis_seconds: float,
) -> None:
    split_lines = [
        f"| {split} | {details['segments']:,} | "
        f"{details['embedding_shape'][1]} | "
        f"{details['recordings']:,} | {details['species']:,} |"
        for split, details in validation["splits"].items()
    ]
    sim_lines = [
        f"| {row['comparison']} | {row['pair_count']:,} | "
        f"{row['mean_cosine_similarity']:.4f} | "
        f"{row['median_cosine_similarity']:.4f} | "
        f"{row['p05_cosine_similarity']:.4f} | "
        f"{row['p95_cosine_similarity']:.4f} |"
        for row in similarity
    ]
    pca_variance = pca_result["explained_variance_ratio"]
    cluster_lines = [
        f"| {row['cluster']} | {row['segment_count']:,} | "
        f"{row['top_species']} | {row['top_species_fraction']:.3f} |"
        for row in cluster_result["clusters"]
    ]
    split_seconds = {
        item["split"]: item["processing_seconds"]
        for item in generation["splits"]
    }
    gpu_memory = generation["cuda_peak_memory_bytes"] / (1024**3)
    resources = generation["resources"]
    report = [
        "# Stage 4 — Pretrained Audio Embedding Baseline",
        "",
        f"- Status: **{generation['status']}**.",
        f"- Encoder: `{generation['model_id']}` "
        f"(checkpoint `{generation.get('model_commit') or 'model revision not reported'}`).",
        f"- Device: `{generation['device']}`; peak allocated CUDA memory "
        f"{gpu_memory:.2f} GiB.",
        f"- Embedding size: **{generation['embedding_dim']}** dimensions; "
        "float32 L2-normalized embeddings.",
        f"- Extraction elapsed: **{generation['processing_seconds']:.1f} s** "
        f"({sum(split_seconds.values()) / 60:.1f} min across split timers); "
        f"analysis elapsed: **{analysis_seconds:.1f} s**.",
        f"- Process CPU utilization: mean "
        f"{resources['process_cpu_percent_mean']}%, peak "
        f"{resources['process_cpu_percent_max']}%; peak process RAM "
        f"{resources['process_rss_peak_bytes'] / (1024**3):.2f} GiB; "
        f"minimum observed system-available RAM "
        f"{resources['system_available_ram_min_bytes'] / (1024**3):.2f} GiB.",
        f"- Sampled GPU utilization: mean "
        f"{resources['gpu_utilization_percent_mean']}%, peak "
        f"{resources['gpu_utilization_percent_max']}% across "
        f"{resources['gpu_utilization_samples']} `nvidia-smi` samples; "
        f"peak driver-reported GPU memory "
        f"{resources['gpu_memory_used_peak_bytes'] / (1024**3):.2f} GiB.",
        "",
        "## Encoder selection and input handling",
        "",
        "CLAP (`laion/clap-htsat-unfused`) was selected for this executable baseline "
        "because its pretrained weights and processor are already available in the "
        "repository's Transformers environment, its released frontend accepts 48 kHz "
        "waveforms, and the same encoder is already used by the project's Watkins "
        "and BE3C baselines. Its released checkpoint is trained on LAION-Audio-630K "
        "audio-text pairs, not specifically on underwater recordings. AVES2 BEATs "
        "is a more bioacoustics-focused alternative, but its AVEX loader currently "
        "requires Python 3.11–3.13 while this Windows environment is Python 3.14; "
        "the PANN/AudioSet-style options are also not marine-specific.",
        "",
        "The CLAP frontend expects 48 kHz mono input, 10 s windows, 64 Mel bins, "
        "and uses frequencies from 50 Hz to 14 kHz. Each segment was reconstructed "
        "from the existing Parquet audio using the Stage 2 192 kHz resampling, "
        "recording-level peak normalization, and 5 s windowing/padding, then "
        "resampled to 48 kHz. CLAP's configured padding extends the 5 s segment to "
        "its 10 s input length. Original files and Stage 1–3 outputs were not changed. "
        "The model discards frequencies above 14 kHz, so these embeddings do not "
        "represent the full bandwidth retained in Stage 2 and require in-domain "
        "validation before ecological interpretation.",
        "",
        "## Embedding counts and validation",
        "",
        "| Split | Segments | Dimensions | Recordings | Species labels |",
        "|---|---:|---:|---:|---:|",
        *split_lines,
        "",
        f"- Total: {validation['total_segments']:,} embeddings for "
        f"{validation['unique_segment_ids']:,} unique segment IDs.",
        "- Invalid vectors: 0; embedding dimensions are consistent; every output "
        "metadata row corresponds to one embedding; all source paths remain valid.",
        "- Duplicate segment IDs and train/validation/test overlap: none.",
        f"- Peak CUDA allocator usage: {gpu_memory:.2f} GiB. "
        "The separate driver-reported GPU memory metric includes display/runtime use.",
        "",
        "## Similarity analysis (training split only)",
        "",
        "| Pair type | Pairs | Mean cosine | Median cosine | 5th percentile | 95th percentile |",
        "|---|---:|---:|---:|---:|---:|",
        *sim_lines,
        "",
        "Pairs use distinct original recordings; species are sampled uniformly "
        "from eligible labels. These descriptive similarities are not a classifier "
        "score or a test-set evaluation.",
        "",
        "## PCA and clustering",
        "",
        f"- PCA was fit using training embeddings only. The first two components "
        f"explain {pca_variance[0]:.4f} and {pca_variance[1]:.4f} of training variance.",
        "- PCA plot: `pca_embeddings.png`; coordinates: `pca_coordinates.csv`.",
        f"- UMAP status: **{umap_result['status']}**"
        + (
            f" on {umap_result['sample_size']:,} training embeddings."
            if umap_result["status"] == "complete"
            else f" ({umap_result.get('reason', 'not run')})"
        ),
        "- K-Means was fit on training embeddings only; validation and test were "
        "not used for clustering or model selection.",
        f"- {cluster_result['method']} with {cluster_result['cluster_count']} "
        f"clusters; sampled train silhouette "
        f"{cluster_result['sampled_train_silhouette']:.4f}; train species NMI "
        f"{cluster_result['train_species_nmi']:.4f}; train species ARI "
        f"{cluster_result['train_species_ari']:.4f}.",
        "",
        "| Cluster | Segments | Five most common labels (count) | Leading-label fraction |",
        "|---:|---:|---|---:|",
        *cluster_lines,
        "",
        "These clusters are candidate acoustic groupings only. They do not prove "
        "communication patterns, animal language, or biological categories.",
        "",
        "## Generated outputs",
        "",
        "- `train/`, `validation/`, and `test/`: `embeddings.npy`, "
        "`metadata.csv`, `completed.npy`, and resumable `progress.json`.",
        "- `encoder_configuration.json`, `generation_summary.json`, `stage4.log`.",
        "- `quality_checks.json`, `cosine_similarity_pairs.csv`, "
        "`cosine_similarity_summary.csv`.",
        "- `pca_embeddings.png`, `pca_coordinates.csv`, "
        "`train_cluster_assignments.csv`, `train_cluster_species_composition.csv`.",
        (
            "- `umap_train_embeddings.png` and `umap_train_coordinates.csv`."
            if umap_result["status"] == "complete"
            else "- UMAP outputs were skipped because the optional dependency was unavailable."
        ),
        "",
        "## Limitations",
        "",
        "- CLAP was not trained specifically on Watkins underwater mammal audio; "
        "the similarity and clustering results are exploratory.",
        "- Frequencies above 14 kHz are not represented in CLAP embeddings.",
        "- PCA/UMAP plots are projections and do not establish separability or "
        "environment invariance. Held-out validation/test data are preserved for "
        "later evaluation.",
        "- No CNN–Transformer, self-supervised training, or Stage 5 work was run.",
        "",
        "## Processing times",
        "",
        *[
            f"- {split}: {seconds:.1f} s."
            for split, seconds in split_seconds.items()
        ],
        f"- Total extraction: {generation['processing_seconds']:.1f} s.",
        f"- Baseline analysis: {analysis_seconds:.1f} s.",
        "",
    ]
    (output_dir / "stage4_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def analyze(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config_path: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    generation = _read_json(output_dir / "generation_summary.json")
    if generation.get("status") != "complete":
        raise ValueError("Embedding extraction summary is not complete.")
    split_results = generation.get("splits", [])
    if not split_results:
        raise ValueError("Generation summary does not contain split timings.")
    generation["processing_seconds"] = round(
        sum(float(split["processing_seconds"]) for split in split_results), 3
    )
    _atomic_json(output_dir / "generation_summary.json", generation)
    if config_path is None:
        config_path = PROJECT_DIR / "watkins_stage4_config.json"
    config = _read_json(config_path)
    matrices, metadata, validation = _validate_embeddings(output_dir)
    _atomic_json(analysis_dir / "quality_checks.json", validation)

    similarity = _similarity_analysis(
        np.asarray(matrices["train"], dtype=np.float32),
        metadata["train"],
        int(config["similarity_pairs_per_group"]),
        int(config["random_state"]),
        analysis_dir,
    )
    pca_result = _pca_analysis(
        matrices,
        metadata,
        int(config["random_state"]),
        analysis_dir,
    )
    umap_result = _umap_analysis(
        matrices,
        metadata,
        int(config["random_state"]),
        int(config["umap_train_sample_size"]),
        analysis_dir,
    )
    cluster_result = _cluster_analysis(
        np.asarray(matrices["train"], dtype=np.float32),
        metadata["train"],
        int(config["cluster_count"]),
        int(config["random_state"]),
        analysis_dir,
    )
    analysis_seconds = time.perf_counter() - started
    _atomic_json(
        analysis_dir / "baseline_analysis.json",
        {
            "similarity": similarity,
            "pca": {
                "fit_split": "train",
                "explained_variance_ratio": pca_result[
                    "explained_variance_ratio"
                ].tolist(),
            },
            "umap": umap_result,
            "clustering": cluster_result,
            "analysis_seconds": round(analysis_seconds, 3),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_report(
        output_dir,
        generation,
        validation,
        similarity,
        pca_result,
        umap_result,
        cluster_result,
        analysis_seconds,
    )
    LOGGER.info("Stage 4 analysis complete in %.1f seconds", analysis_seconds)
    return {
        "quality": validation,
        "similarity": similarity,
        "pca_explained_variance_ratio": pca_result[
            "explained_variance_ratio"
        ].tolist(),
        "umap": umap_result,
        "clustering": {
            key: value
            for key, value in cluster_result.items()
            if key != "clusters"
        },
        "analysis_seconds": round(analysis_seconds, 3),
        "report": str(output_dir / "stage4_report.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and analyze pretrained Watkins embeddings."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.output_dir, args.config), indent=2))


if __name__ == "__main__":
    main()
