"""Post-training comparison of Stage 4 and Stage 5 acoustic embeddings."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)


SPLITS = ("train", "validation", "test")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sample_pairs(
    metadata: list[dict[str, str]], seed: int, pairs_per_type: int = 2000
) -> dict[str, list[tuple[int, int]]]:
    rng = random.Random(seed)
    by_species: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        species = row.get("species_label", "").strip()
        if species:
            by_species[species].append(index)

    species_names = sorted(by_species)
    outputs = {
        "same_species": [],
        "different_species": [],
        "same_species_same_location": [],
        "same_species_different_location": [],
    }
    seen = {key: set() for key in outputs}
    attempts = 0
    max_attempts = pairs_per_type * 100
    while attempts < max_attempts and any(
        len(values) < pairs_per_type for values in outputs.values()
    ):
        attempts += 1
        if len(species_names) < 2:
            break
        same = rng.random() < 0.5
        if same:
            eligible = [name for name in species_names if len(by_species[name]) >= 2]
            if not eligible:
                continue
            species = rng.choice(eligible)
            left, right = rng.sample(by_species[species], 2)
            if metadata[left].get("recording_id") == metadata[right].get(
                "recording_id"
            ):
                continue
            pair = tuple(sorted((left, right)))
            key = "same_species"
            left_location = metadata[left].get("location_domain", "").strip()
            right_location = metadata[right].get("location_domain", "").strip()
            if left_location and right_location:
                env_key = (
                    "same_species_same_location"
                    if left_location == right_location
                    else "same_species_different_location"
                )
                if (
                    len(outputs[env_key]) < pairs_per_type
                    and pair not in seen[env_key]
                ):
                    outputs[env_key].append(pair)
                    seen[env_key].add(pair)
        else:
            left_species, right_species = rng.sample(species_names, 2)
            left = rng.choice(by_species[left_species])
            right = rng.choice(by_species[right_species])
            key = "different_species"
            pair = tuple(sorted((left, right)))

        if metadata[left].get("recording_id") == metadata[right].get("recording_id"):
            continue
        if len(outputs[key]) < pairs_per_type and pair not in seen[key]:
            outputs[key].append(pair)
            seen[key].add(pair)
    return outputs


def _cosine_rows(
    pairs: dict[str, list[tuple[int, int]]],
    baseline: np.ndarray,
    learned: np.ndarray,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for pair_type, indices in pairs.items():
        if not indices:
            continue
        left = np.asarray([pair[0] for pair in indices])
        right = np.asarray([pair[1] for pair in indices])
        baseline_values = np.sum(baseline[left] * baseline[right], axis=1)
        learned_values = np.sum(learned[left] * learned[right], axis=1)
        for representation, values in (
            ("stage4_clap", baseline_values),
            ("stage5_ssl", learned_values),
        ):
            result.append(
                {
                    "pair_type": pair_type,
                    "representation": representation,
                    "pairs": len(values),
                    "mean_cosine": float(np.mean(values)),
                    "median_cosine": float(np.median(values)),
                    "p05_cosine": float(np.quantile(values, 0.05)),
                    "p95_cosine": float(np.quantile(values, 0.95)),
                }
            )
    return result


def _linear_probe(
    train: np.ndarray,
    train_labels: np.ndarray,
    validation: np.ndarray,
    validation_labels: np.ndarray,
    test: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    classifier.fit(train, train_labels)
    known_classes = classifier.classes_
    results: dict[str, Any] = {}
    for split, vectors, labels in (
        ("validation", validation, validation_labels),
        ("test", test, test_labels),
    ):
        prediction = classifier.predict(vectors)
        results[split] = {
            "accuracy": float(accuracy_score(labels, prediction)),
            "macro_f1_train_classes": float(
                f1_score(
                    labels,
                    prediction,
                    labels=known_classes,
                    average="macro",
                    zero_division=0,
                )
            ),
            "samples": int(len(labels)),
            "unseen_species_samples": int(
                sum(label not in known_classes for label in labels)
            ),
        }
    return results


def _cluster_metrics(
    features: np.ndarray, labels: np.ndarray, seed: int
) -> dict[str, Any]:
    clusters = MiniBatchKMeans(
        n_clusters=20,
        random_state=seed,
        batch_size=1024,
        n_init=5,
    ).fit_predict(features)
    rng = np.random.default_rng(seed)
    silhouette_indices = rng.choice(
        len(features), size=min(2000, len(features)), replace=False
    )
    return {
        "clusters": 20,
        "sampled_train_silhouette": float(
            silhouette_score(
                features[silhouette_indices],
                clusters[silhouette_indices],
                metric="euclidean",
            )
        ),
        "train_species_nmi": float(normalized_mutual_info_score(labels, clusters)),
        "train_species_ari": float(adjusted_rand_score(labels, clusters)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _environment_inventory(metadata_by_split: dict[str, list[dict[str, str]]]) -> dict:
    fields = {
        "location_domain": "recording location domain",
        "observation_date": "observation date",
        "sound_channel_source_code": "sound/channel source code",
    }
    result: dict[str, Any] = {}
    for split, rows in metadata_by_split.items():
        result[split] = {}
        for field, description in fields.items():
            values = [row.get(field, "").strip() for row in rows]
            known = [value for value in values if value]
            result[split][field] = {
                "description": description,
                "known_segments": len(known),
                "segments": len(rows),
                "unique_values": len(set(known)),
            }
    result["availability"] = {
        "recording_device_id": False,
        "recording_session_id": False,
        "location_proxy": True,
        "date": True,
        "note": (
            "The channel/source code is not a verified recorder/device identifier; "
            "location is a coarse environmental proxy."
        ),
    }
    return result


def _make_pca_plot(
    baseline: np.ndarray,
    learned: np.ndarray,
    labels: np.ndarray,
    segment_ids: list[str],
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(labels), size=min(6000, len(labels)), replace=False)
    models = {
        "stage4_clap": PCA(n_components=2, random_state=seed).fit(baseline),
        "stage5_ssl": PCA(n_components=2, random_state=seed).fit(learned),
    }
    coordinates: list[dict[str, Any]] = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    species_names = sorted(set(labels.tolist()))
    species_to_id = {label: index for index, label in enumerate(species_names)}
    color_values = np.asarray([species_to_id[label] for label in labels[indices]])
    for axis, (name, model), data in zip(
        axes,
        models.items(),
        (baseline, learned),
    ):
        points = model.transform(data[indices])
        axis.scatter(
            points[:, 0],
            points[:, 1],
            c=color_values,
            cmap="turbo",
            s=5,
            alpha=0.65,
            linewidths=0,
        )
        axis.set_title(
            f"{name} — train fit "
            f"({model.explained_variance_ratio_.sum():.3f} variance)"
        )
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        coordinates.extend(
            {
                "representation": name,
                "segment_id": segment_ids[index],
                "pc1": float(point[0]),
                "pc2": float(point[1]),
                "species": labels[index],
            }
            for index, point in zip(indices, points)
        )
    fig.savefig(output_dir / "pca_baseline_vs_ssl.png", dpi=160)
    plt.close(fig)
    _write_csv(output_dir / "pca_coordinates.csv", coordinates)
    return {
        name: {
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
            "sampled_train_points": len(indices),
        }
        for name, model in models.items()
    }


def _make_umap_plot(
    baseline: np.ndarray,
    learned: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    try:
        import umap
    except ImportError as error:
        return {"status": "skipped", "reason": str(error)}

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(labels), size=min(5000, len(labels)), replace=False)
    species_names = sorted(set(labels.tolist()))
    species_to_id = {label: index for index, label in enumerate(species_names)}
    colors = np.asarray([species_to_id[label] for label in labels[indices]])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    outputs: dict[str, Any] = {}
    for axis, name, features in (
        (axes[0], "stage4_clap", baseline),
        (axes[1], "stage5_ssl", learned),
    ):
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
            n_jobs=1,
        )
        points = reducer.fit_transform(features[indices])
        axis.scatter(
            points[:, 0],
            points[:, 1],
            c=colors,
            cmap="turbo",
            s=5,
            alpha=0.65,
            linewidths=0,
        )
        axis.set_title(f"{name} — train sample")
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        outputs[name] = {
            "sampled_train_points": len(indices),
            "parameters": {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "metric": "cosine",
            },
        }
    fig.savefig(output_dir / "umap_baseline_vs_ssl.png", dpi=160)
    plt.close(fig)
    outputs["status"] = "complete"
    return outputs


def _write_report(
    output_dir: Path,
    metrics: dict[str, Any],
    training: dict[str, Any],
    config: dict[str, Any],
) -> None:
    train_row = _read_rows(output_dir / "training_log.csv")[-1]
    best_loss = float(training["best_validation_loss"])
    aug = config["augmentations"]
    similarity = metrics["cosine_similarity"]
    probe = metrics["linear_probe"]
    cluster = metrics["clustering"]
    environment = metrics["environment_metadata"]
    environment_gap = [
        row
        for row in similarity
        if row["pair_type"]
        in ("same_species_same_location", "same_species_different_location")
    ]
    lines = [
        "# Stage 5 — Waveform-level self-supervised learning",
        "",
        "- Status: **complete**.",
        "- Method: SimCLR-style NT-Xent on two independent augmented waveform views.",
        f"- Encoder: `{config['model_id']}`; all encoder parameters frozen.",
        f"- Projection head: 512 → {config['projection_hidden_dim']} → "
        f"{config['projection_dim']} dimensions; trainable parameters: "
        f"{training['projection_trainable_parameters']:,}.",
        f"- Device: `{training['device']}`; mixed precision: "
        f"{bool(config['mixed_precision'])}; peak allocated CUDA memory: "
        f"{training['cuda_peak_allocated_mib']:.1f} MiB.",
        f"- Best checkpoint: `{training['checkpoint']}`.",
        f"- Training duration: {training['elapsed_training_seconds']:.1f} s.",
        f"- Training segments: {training['train_samples_per_epoch']:,} per epoch; "
        f"validation segments: {training['validation_samples_per_epoch']:,} per epoch.",
        f"- Total training examples processed across completed epochs: "
        f"{training['train_samples_processed']:,}.",
        f"- Best epoch: {training['best_epoch']}; best validation contrastive loss: "
        f"{best_loss:.6f}.",
        f"- Final epoch training loss: {float(train_row['train_loss']):.6f}; "
        f"validation loss: {float(train_row['validation_loss']):.6f}.",
        "",
        "## Method and augmentations",
        "",
        "For each segment, the original Stage 4 waveform reconstruction is resampled "
        "to 48 kHz mono and sent through the released CLAP processor. No species or "
        "environment labels are read by the training loss. The CLAP feature extractor "
        "is frozen; only the small MLP projection head is optimized. NT-Xent uses "
        f"temperature {config['temperature']}.",
        "",
        f"- Independent zero-filled time shifts: up to ±{aug['max_time_shift_ms']:g} ms.",
        f"- Independent gain change: ±{aug['gain_db']:g} dB.",
        f"- Mild additive Gaussian noise: per-view SNR uniformly sampled from "
        f"{aug['min_noise_snr_db']:g}–{aug['max_noise_snr_db']:g} dB.",
        "- No pitch shift, frequency warp, time/frequency masking, or label-conditioned "
        "augmentation is applied.",
        "",
        "## Held-out evaluation",
        "",
        "Linear probes are fit on training labels only and scored on validation/test. "
        "Clustering is fit on training embeddings. Test labels are used only for the "
        "post-training comparison. Metrics are descriptive; they are not claims of "
        "biological categories or unknown vocalization discovery.",
        "",
        "| Representation | Val accuracy | Val macro-F1 | Test accuracy | Test macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("stage4_clap", "stage5_ssl"):
        row = probe[name]
        lines.append(
            f"| {name} | {row['validation']['accuracy']:.4f} | "
            f"{row['validation']['macro_f1_train_classes']:.4f} | "
            f"{row['test']['accuracy']:.4f} | "
            f"{row['test']['macro_f1_train_classes']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Representation | Train silhouette (sampled) | Train species NMI | Train species ARI |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("stage4_clap", "stage5_ssl"):
        row = cluster[name]
        lines.append(
            f"| {name} | {row['sampled_train_silhouette']:.4f} | "
            f"{row['train_species_nmi']:.4f} | {row['train_species_ari']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Cosine similarities use identical, distinct-recording test pairs for both "
            "representations:",
            "",
            "| Pair type | Representation | Pairs | Mean cosine | Median cosine |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in similarity:
        lines.append(
            f"| {row['pair_type']} | {row['representation']} | {row['pairs']} | "
            f"{row['mean_cosine']:.4f} | {row['median_cosine']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"PCA visualization: `pca_baseline_vs_ssl.png`; PCA coordinates: "
            f"`pca_coordinates.csv`. PCA is fit on the training split only. "
            f"UMAP status: {metrics['umap']['status']}; plot: "
            "`umap_baseline_vs_ssl.png` when complete.",
            "",
            "## Environment metadata and limitations",
            "",
            "Location domain and observation-date metadata are available in the segment "
            "metadata; the sound/channel source code is not a verified device ID. "
            "Recording-session and recorder/device identifiers are not present, so "
            "session/device invariance cannot be tested. Location is only a coarse "
            "environment proxy and is correlated with species. Same-species "
            "same-location versus cross-location similarities are reported below as "
            "exploratory evidence only; they do not establish environment invariance.",
            "",
            "| Test pair subset | Representation | Pairs | Mean cosine |",
            "|---|---|---:|---:|",
        ]
    )
    for row in environment_gap:
        lines.append(
            f"| {row['pair_type']} | {row['representation']} | {row['pairs']} | "
            f"{row['mean_cosine']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Metadata inventory (known segments / split size; unique non-empty values):",
            "",
            "| Split | Location domains | Observation dates | Sound/channel source codes |",
            "|---|---|---|---|",
        ]
    )
    for split in SPLITS:
        values = environment[split]
        lines.append(
            f"| {split} | {values['location_domain']['known_segments']}/"
            f"{values['location_domain']['segments']} "
            f"({values['location_domain']['unique_values']} values) | "
            f"{values['observation_date']['known_segments']}/"
            f"{values['observation_date']['segments']} "
            f"({values['observation_date']['unique_values']} values) | "
            f"{values['sound_channel_source_code']['known_segments']}/"
            f"{values['sound_channel_source_code']['segments']} "
            f"({values['sound_channel_source_code']['unique_values']} values) |"
        )
    lines.extend(
        [
            "",
            "No claim of environment invariance is made. Frozen CLAP may constrain the "
            "representation, and gains here apply only to the learned projection and "
            "these evaluation protocols. Silhouette and UMAP depend on representation "
            "geometry and sampling; they should not be read as biological validity.",
            "",
        ]
    )
    (output_dir / "stage5_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def evaluate_and_report(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    metadata_by_split: dict[str, list[dict[str, str]]] = {}
    baseline: dict[str, np.ndarray] = {}
    learned: dict[str, np.ndarray] = {}
    for split in SPLITS:
        metadata_by_split[split] = _read_rows(
            output_dir / split / "metadata.csv"
        )
        baseline[split] = np.load(
            Path(__file__).resolve().parent
            / "watkins_stage4"
            / split
            / "embeddings.npy",
            mmap_mode="r",
        )
        learned[split] = np.load(
            output_dir / split / "embeddings.npy", mmap_mode="r"
        )
        ids = [row["segment_id"] for row in metadata_by_split[split]]
        stage4_ids = [
            row["segment_id"]
            for row in _read_rows(
                Path(__file__).resolve().parent
                / "watkins_stage4"
                / split
                / "metadata.csv"
            )
        ]
        if ids != stage4_ids:
            raise ValueError(f"{split} Stage 5 embedding ID order mismatch")
        if baseline[split].shape[0] != len(ids) or learned[split].shape[0] != len(
            ids
        ):
            raise ValueError(f"{split} embedding row count does not match metadata")
        if not np.isfinite(learned[split]).all():
            raise ValueError(f"{split} Stage 5 embeddings contain non-finite values")

    seed = int(config["random_seed"])
    label_arrays = {
        split: np.asarray(
            [row.get("species_label", "").strip() for row in metadata_by_split[split]]
        )
        for split in SPLITS
    }
    metrics: dict[str, Any] = {
        "embedding_dimensions": {
            "stage4_clap": int(baseline["train"].shape[1]),
            "stage5_ssl": int(learned["train"].shape[1]),
        }
    }
    metrics["linear_probe"] = {
        name: _linear_probe(
            baseline["train"] if name == "stage4_clap" else learned["train"],
            label_arrays["train"],
            baseline["validation"] if name == "stage4_clap" else learned["validation"],
            label_arrays["validation"],
            baseline["test"] if name == "stage4_clap" else learned["test"],
            label_arrays["test"],
            seed,
        )
        for name in ("stage4_clap", "stage5_ssl")
    }
    metrics["clustering"] = {
        "stage4_clap": _cluster_metrics(
            baseline["train"], label_arrays["train"], seed
        ),
        "stage5_ssl": _cluster_metrics(learned["train"], label_arrays["train"], seed),
    }
    pair_indices = _sample_pairs(metadata_by_split["test"], seed)
    metrics["cosine_similarity"] = _cosine_rows(
        pair_indices, baseline["test"], learned["test"]
    )
    metrics["environment_metadata"] = _environment_inventory(metadata_by_split)
    metrics["pca"] = _make_pca_plot(
        baseline["train"],
        learned["train"],
        label_arrays["train"],
        [row["segment_id"] for row in metadata_by_split["train"]],
        output_dir,
        seed,
    )
    metrics["umap"] = _make_umap_plot(
        baseline["train"],
        learned["train"],
        label_arrays["train"],
        output_dir,
        seed,
    )

    training = json.loads(
        (output_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    _write_csv(
        output_dir / "cosine_similarity_comparison.csv",
        metrics["cosine_similarity"],
    )
    _write_csv(
        output_dir / "cluster_metrics.csv",
        [
            {"representation": name, **values}
            for name, values in metrics["clustering"].items()
        ],
    )
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    _write_report(output_dir, metrics, training, config)
    return metrics
