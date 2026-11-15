from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE4_DIR = PROJECT_ROOT / "watkins_stage4"
STAGE6_DIR = PROJECT_ROOT / "watkins_stage6"
OUTPUT_DIR = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")
METADATA_COLUMNS = (
    "segment_id",
    "recording_id",
    "audio_path",
    "audio_storage_type",
    "audio_parquet_row_index",
    "parquet_file",
    "parquet_row_index",
    "segment_start_seconds",
    "segment_valid_seconds",
    "location_names",
    "observation_date",
    "source_sample_rate_hz",
    "channel_count",
)
LOGGER = logging.getLogger("watkins_stage7")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "stage7.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _priority_for_segment(
    cluster_id: int | None, novelty_flag: bool
) -> tuple[str, str]:
    if cluster_id is None:
        return "not_clustered" if not novelty_flag else "medium", "not_clustered_by_stage6"
    if cluster_id == -1:
        return ("high" if novelty_flag else "review"), "noise"
    return ("medium" if novelty_flag else "normal"), "clustered"


def _load_split(split: str, expected_dim: int) -> tuple[np.ndarray, pd.DataFrame]:
    split_dir = STAGE4_DIR / split
    embeddings = np.load(split_dir / "embeddings.npy", mmap_mode="r")
    metadata = pd.read_csv(
        split_dir / "metadata.csv",
        usecols=list(METADATA_COLUMNS),
        dtype={"segment_id": "string", "recording_id": "string"},
    )
    manifest_path = PROJECT_ROOT / "watkins_splits" / f"{split}.csv"
    manifest = pd.read_csv(
        manifest_path,
        usecols=["segment_id", "recording_id"],
        dtype={"segment_id": "string", "recording_id": "string"},
    )
    if embeddings.ndim != 2 or embeddings.shape != (len(metadata), expected_dim):
        raise ValueError(
            f"{split} embedding/metadata mismatch: {embeddings.shape} vs "
            f"{len(metadata)} rows at {expected_dim} dimensions."
        )
    if metadata["segment_id"].isna().any() or metadata["recording_id"].isna().any():
        raise ValueError(f"{split} metadata contains missing IDs.")
    if metadata["segment_id"].duplicated().any():
        raise ValueError(f"{split} metadata contains duplicate segment IDs.")
    if not np.array_equal(metadata["segment_id"].to_numpy(), manifest["segment_id"].to_numpy()):
        raise ValueError(f"{split} metadata IDs do not match its split manifest.")
    if not np.array_equal(
        metadata["recording_id"].to_numpy(), manifest["recording_id"].to_numpy()
    ):
        raise ValueError(f"{split} recording IDs do not match its split manifest.")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{split} Stage 4 embeddings contain non-finite values.")

    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{split} Stage 4 embeddings contain zero vectors.")
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        LOGGER.warning("L2-normalizing %s vectors in memory for analysis", split)
        vectors = vectors / norms[:, None]
    return vectors, metadata


def _load_stage6_assignments(train_metadata: pd.DataFrame) -> pd.DataFrame:
    path = STAGE6_DIR / "cluster_assignments.csv"
    assignments = pd.read_csv(
        path,
        dtype={"segment_id": "string", "recording_id": "string"},
    )
    required = {
        "segment_id",
        "recording_id",
        "cluster",
        "membership_probability",
    }
    if not required.issubset(assignments.columns):
        raise ValueError(f"Stage 6 assignments lack columns: {sorted(required - set(assignments.columns))}")
    if assignments["segment_id"].duplicated().any():
        raise ValueError("Stage 6 cluster assignments contain duplicate segment IDs.")
    if not np.array_equal(
        train_metadata["segment_id"].to_numpy(), assignments["segment_id"].to_numpy()
    ):
        raise ValueError("Stage 6 assignment IDs are not aligned with Stage 4 train IDs.")
    if not np.array_equal(
        train_metadata["recording_id"].to_numpy(),
        assignments["recording_id"].to_numpy(),
    ):
        raise ValueError("Stage 6 recording IDs are not aligned with Stage 4 train IDs.")
    if assignments["cluster"].isna().any():
        raise ValueError("Stage 6 train assignments contain missing cluster IDs.")
    probabilities = assignments["membership_probability"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("Stage 6 membership probabilities must be finite values in [0, 1].")
    return assignments


def _percentile_against_train(
    training_scores_sorted: np.ndarray, scores: np.ndarray
) -> np.ndarray:
    return (
        100.0
        * np.searchsorted(training_scores_sorted, scores, side="right")
        / len(training_scores_sorted)
    )


def _score_summary(
    frame: pd.DataFrame, split: str, threshold: float
) -> dict[str, Any]:
    scores = frame["novelty_score"].to_numpy(dtype=np.float64)
    flagged = frame["novelty_flag"].to_numpy(dtype=bool)
    return {
        "segments": int(len(frame)),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_min": float(scores.min()),
        "score_median": float(np.quantile(scores, 0.50)),
        "score_p90": float(np.quantile(scores, 0.90)),
        "score_p95": float(np.quantile(scores, 0.95)),
        "score_p99": float(np.quantile(scores, 0.99)),
        "score_max": float(scores.max()),
        "threshold": float(threshold),
        "flagged_segments": int(flagged.sum()),
        "flagged_segment_percentage": float(100 * flagged.mean()),
        "flagged_recordings": int(frame.loc[flagged, "recording_id"].nunique()),
    }


def _training_cluster_evaluation(
    scores: np.ndarray, flags: np.ndarray, cluster_ids: np.ndarray
) -> dict[str, Any]:
    noise = cluster_ids == -1
    clustered = cluster_ids >= 0
    if not noise.any() or not clustered.any():
        auc: float | None = None
        correlation: float | None = None
    else:
        auc = float(roc_auc_score(noise.astype(np.int8), scores))
        correlation_value = spearmanr(scores, noise.astype(np.int8)).statistic
        correlation = (
            float(correlation_value) if np.isfinite(correlation_value) else None
        )
    return {
        "hdbscan_noise_segments": int(noise.sum()),
        "hdbscan_clustered_segments": int(clustered.sum()),
        "noise_flagged_segments": int(flags[noise].sum()),
        "noise_flagged_percentage": float(100 * flags[noise].mean()) if noise.any() else None,
        "clustered_flagged_segments": int(flags[clustered].sum()),
        "clustered_flagged_percentage": (
            float(100 * flags[clustered].mean()) if clustered.any() else None
        ),
        "mean_novelty_score_noise": float(scores[noise].mean()) if noise.any() else None,
        "mean_novelty_score_clustered": (
            float(scores[clustered].mean()) if clustered.any() else None
        ),
        "roc_auc_novelty_score_vs_hdbscan_noise": auc,
        "spearman_novelty_score_vs_hdbscan_noise": correlation,
        "interpretation": (
            "Agreement with HDBSCAN noise is descriptive only; HDBSCAN noise "
            "is not ground-truth novelty."
        ),
    }


def _write_plots(
    output_dir: Path, combined: pd.DataFrame, threshold: float
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = {"train": "#2878b5", "validation": "#f0a202", "test": "#5a9f68"}
    for split in SPLITS:
        values = combined.loc[
            combined["split"] == split, "novelty_score"
        ].to_numpy()
        axis.hist(
            values,
            bins=50,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=colors[split],
            label=f"{split} (n={len(values):,})",
        )
    axis.axvline(
        threshold,
        color="#d62728",
        linestyle="--",
        label=f"Training threshold ({threshold:.4f})",
    )
    axis.set(
        title="Isolation Forest novelty-score distributions",
        xlabel="Novelty score (higher means more anomalous)",
        ylabel="Density",
    )
    axis.legend()
    figure.savefig(output_dir / "novelty_score_distribution.png", dpi=180)
    plt.close(figure)

    training = combined.loc[combined["split"] == "train"]
    noise = training["hdbscan_cluster_id"].to_numpy(dtype=np.int64) == -1
    clustered = ~noise
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    rng = np.random.default_rng(29)
    for mask, label, color in (
        (clustered, "Stage 6 clustered", "#2878b5"),
        (noise, "Stage 6 noise (-1)", "#d62728"),
    ):
        values = training.loc[mask, "novelty_score"].to_numpy()
        axis.boxplot(
            values,
            positions=[1 if label == "Stage 6 clustered" else 2],
            widths=0.45,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.28},
            medianprops={"color": color, "linewidth": 2},
        )
        sample_count = min(1500, len(values))
        sample = rng.choice(values, size=sample_count, replace=False)
        x_center = 1 if label == "Stage 6 clustered" else 2
        jitter = rng.uniform(-0.15, 0.15, size=sample_count)
        axis.scatter(
            x_center + jitter,
            sample,
            s=7,
            alpha=0.22,
            color=color,
            label=f"{label} (n={len(values):,})",
            rasterized=True,
        )
    axis.axhline(threshold, color="#333333", linestyle="--", label="Novelty cutoff")
    axis.set_xticks([1, 2], ["Clustered", "Noise (-1)"])
    axis.set(
        title="Novelty scores by Stage 6 HDBSCAN status (training only)",
        ylabel="Novelty score (higher means more anomalous)",
    )
    axis.legend()
    figure.savefig(output_dir / "cluster_vs_novelty.png", dpi=180)
    plt.close(figure)


def _write_report(
    output_dir: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    split_lines = "\n".join(
        f"| {split} | {values['segments']:,} | {values['flagged_segments']:,} | "
        f"{values['flagged_recordings']:,} | {values['flagged_segment_percentage']:.2f}% |"
        for split, values in summary["score_summary_by_split"].items()
    )
    cluster_eval = summary["training_cluster_evaluation"]
    noise_rate = cluster_eval["noise_flagged_percentage"]
    clustered_rate = cluster_eval["clustered_flagged_percentage"]
    report = f"""# Stage 7 — Novelty detection and investigation prioritisation

- Status: **complete**.
- Algorithm: Isolation Forest (`n_estimators={config['isolation_forest']['n_estimators']}`, `max_samples={config['isolation_forest']['max_samples']}`, contamination `{config['isolation_forest']['contamination']}`).
- Representation: original, L2-normalized Stage 4 CLAP vectors in **512-D** space. No UMAP coordinates or Stage 5 embeddings were used as novelty-model inputs.
- Model fit: training split only ({summary['training_segments']:,} segments). Validation/test were scored only after fitting.
- Segments scored: {summary['total_scored_segments']:,} across train/validation/test.
- High-novelty threshold: {summary['novelty_threshold']:.6f}, derived from the training-fit contamination cutoff.
- Higher novelty score means more anomalous (`-IsolationForest.score_samples`).
- Novelty-flagged recordings: {summary['flagged_recordings']:,} unique recordings.
- High-priority recordings (Stage 6 noise **and** high novelty): {summary['high_priority_recordings']:,}.
- Execution time: {summary['execution_seconds']:.1f} seconds.

## Input and alignment checks

Stage 4 embedding rows were checked against metadata and each split manifest. Train cluster assignments from Stage 6 were checked against training segment and recording IDs. Recording IDs were checked for overlap across splits to preserve the existing split integrity. Missing Stage 6 labels on validation/test are represented as unavailable—not as noise or as a predicted cluster.

| Split | Segments | Novelty-flagged segments | Unique flagged recordings | Flagged segment rate |
|---|---:|---:|---:|---:|
{split_lines}

## Priority scheme

The threshold is fixed from the training-only Isolation Forest fit; a segment is flagged when its score is above the cutoff. The investigation queue contains the highest-scoring flagged segment for each recording.

| Stage 6 status | Novelty flag | Priority |
|---|---|---|
| Noise (-1) | Yes | High |
| Clustered | Yes | Medium |
| Stage 6 status unavailable (validation/test) | Yes | Medium |
| Noise (-1) | No | Review (not novelty-flagged; excluded from queue) |
| Clustered | No | Normal |
| Stage 6 status unavailable (validation/test) | No | Not clustered |

Noise alone does not trigger a novelty flag. High priority requires both HDBSCAN noise and a high Isolation Forest score.

## Comparison with HDBSCAN noise (training only)

- Noise segments flagged: {cluster_eval['noise_flagged_segments']:,}/{cluster_eval['hdbscan_noise_segments']:,} ({noise_rate:.2f}%).
- Clustered segments flagged: {cluster_eval['clustered_flagged_segments']:,}/{cluster_eval['hdbscan_clustered_segments']:,} ({clustered_rate:.2f}%).
- Mean novelty score: HDBSCAN noise {cluster_eval['mean_novelty_score_noise']:.6f}; clustered {cluster_eval['mean_novelty_score_clustered']:.6f}.
- ROC AUC of novelty scores for distinguishing HDBSCAN noise from clustered samples: {cluster_eval['roc_auc_novelty_score_vs_hdbscan_noise']:.4f}.
- Spearman correlation of novelty score with HDBSCAN noise indicator: {cluster_eval['spearman_novelty_score_vs_hdbscan_noise']:.4f}.

These are descriptive comparisons, not accuracy or precision: HDBSCAN noise is not verified novelty ground truth. No novelty labels or species labels were used to fit or tune Isolation Forest.

## Researcher audio access

`investigation_queue.csv` includes the source Parquet path, Parquet row, valid segment offset/duration, original sample rate, and channel count. Audio is embedded in Parquet and is not a standalone WAV. To export a listenable crop, run:

```powershell
python watkins_stage7\\extract_stage7_audio.py --segment-id "<segment_id from investigation_queue.csv>"
```

The helper verifies the recording ID at the Parquet row and writes a PCM-16 WAV crop at the source sample rate/channels under `watkins_stage7\\audio_samples`. The original Parquet and Stage 4/5/6 artifacts are read-only.

## Outputs

- `isolation_forest.joblib` — fitted training-only novelty detector.
- `segment_novelty_scores.csv` — scores, training-reference percentiles, threshold flags, split IDs.
- `cluster_novelty_results.csv` — scores combined with Stage 6 cluster IDs/probabilities where available and priorities.
- `investigation_queue.csv` — one top flagged segment per unique recording, ranked high priority first.
- `novelty_score_distribution.png`, `cluster_vs_novelty.png`, `run_summary.json`, `stage7_config.json`, and `stage7.log`.
- `extract_stage7_audio.py` — on-demand Parquet segment retrieval for listening.

## Interpretation and limitations

This is an unsupervised prioritisation baseline, not a biological discovery system. A high anomaly score means the segment is unusual relative to the fitted Isolation Forest's training representation; it does not identify why it is unusual. Results can reflect recording conditions, rare but known calls, preprocessing artefacts, or embedding limitations. Stage 6 cluster IDs are available only for training segments because Stage 6 fit HDBSCAN on training data and did not generate held-out cluster predictions. There are no verified novelty ground-truth labels, so accuracy/precision are not reported. High priority is a request for expert review, never a new-species claim.

## Reproducibility

Seed: {config['random_state']}; Python {summary['software']['python']}; scikit-learn {summary['software']['scikit_learn']}; NumPy {summary['software']['numpy']}; pandas {summary['software']['pandas']}. The full model and score parameters are recorded in `stage7_config.json` and `run_summary.json`.
"""
    (output_dir / "stage7_report.md").write_text(report, encoding="utf-8")


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    output_resolved = output_dir.resolve()
    for protected in (STAGE4_DIR.resolve(), (PROJECT_ROOT / "watkins_stage5").resolve(), STAGE6_DIR.resolve()):
        if output_resolved == protected or output_resolved.is_relative_to(protected):
            raise ValueError("Stage 7 output must not be inside or replace a prior stage.")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "stage7.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("fit_split") != "train":
        raise ValueError("Isolation Forest must be fit using the training split only.")

    started_at = _utc_now()
    timer = time.perf_counter()
    matrices: dict[str, np.ndarray] = {}
    metadata: dict[str, pd.DataFrame] = {}
    recording_ids: dict[str, set[str]] = {}
    for split in SPLITS:
        matrices[split], metadata[split] = _load_split(
            split, int(config["embedding_dim"])
        )
        recording_ids[split] = set(metadata[split]["recording_id"].astype(str))
        LOGGER.info(
            "Validated Stage 4 split=%s segments=%d dim=%d",
            split,
            len(metadata[split]),
            matrices[split].shape[1],
        )
    for left_index, left_split in enumerate(SPLITS):
        for right_split in SPLITS[left_index + 1 :]:
            overlap = recording_ids[left_split] & recording_ids[right_split]
            if overlap:
                raise ValueError(
                    f"Recording-level split overlap between {left_split} and "
                    f"{right_split}: {len(overlap)} recording IDs."
                )

    stage6 = _load_stage6_assignments(metadata["train"])
    model_config = config["isolation_forest"]
    model = IsolationForest(
        n_estimators=int(model_config["n_estimators"]),
        max_samples=min(int(model_config["max_samples"]), len(matrices["train"])),
        contamination=float(model_config["contamination"]),
        n_jobs=int(model_config["n_jobs"]),
        random_state=int(config["random_state"]),
    )
    LOGGER.info(
        "Fitting Isolation Forest on %d Stage 4 training vectors",
        len(matrices["train"]),
    )
    model.fit(matrices["train"])
    threshold = float(-model.offset_)
    training_scores = -model.score_samples(matrices["train"])
    if not np.isfinite(training_scores).all() or not np.isfinite(threshold):
        raise ValueError("Isolation Forest produced non-finite training scores.")
    sorted_training_scores = np.sort(training_scores)
    LOGGER.info("Training novelty threshold=%.8f", threshold)

    score_frames: list[pd.DataFrame] = []
    combined_frames: list[pd.DataFrame] = []
    score_summary: dict[str, dict[str, Any]] = {}
    split_scores: dict[str, np.ndarray] = {}
    split_flags: dict[str, np.ndarray] = {}
    split_clusters: dict[str, np.ndarray] = {}
    for split in SPLITS:
        scores = (
            training_scores
            if split == "train"
            else -model.score_samples(matrices[split])
        )
        if not np.isfinite(scores).all():
            raise ValueError(f"Isolation Forest produced non-finite {split} scores.")
        percentiles = _percentile_against_train(sorted_training_scores, scores)
        flags = scores > threshold
        if split == "train":
            cluster_ids = stage6["cluster"].to_numpy(dtype=np.int32)
            probabilities = stage6["membership_probability"].to_numpy(dtype=np.float32)
        else:
            cluster_ids = np.full(len(scores), -2, dtype=np.int32)
            probabilities = np.full(len(scores), np.nan, dtype=np.float32)

        priorities: list[str] = []
        statuses: list[str] = []
        for cluster_id, flag in zip(cluster_ids, flags):
            known_cluster = None if cluster_id == -2 else int(cluster_id)
            priority, status = _priority_for_segment(known_cluster, bool(flag))
            priorities.append(priority)
            statuses.append(status)

        base = metadata[split].copy()
        base.insert(0, "split", split)
        base["hdbscan_cluster_id"] = pd.array(
            [pd.NA if cluster_id == -2 else int(cluster_id) for cluster_id in cluster_ids],
            dtype="Int64",
        )
        base["hdbscan_membership_probability"] = probabilities
        base["hdbscan_status"] = statuses
        base["novelty_score"] = scores.astype(np.float64)
        base["novelty_percentile_vs_train"] = percentiles
        base["novelty_threshold"] = threshold
        base["novelty_flag"] = flags
        base["investigation_priority"] = priorities

        score_frames.append(
            base[
                [
                    "split",
                    "segment_id",
                    "recording_id",
                    "novelty_score",
                    "novelty_percentile_vs_train",
                    "novelty_threshold",
                    "novelty_flag",
                ]
            ]
        )
        combined_frames.append(base)
        split_scores[split] = scores
        split_flags[split] = flags
        split_clusters[split] = cluster_ids
        score_summary[split] = _score_summary(base, split, threshold)

    scores_frame = pd.concat(score_frames, ignore_index=True)
    combined = pd.concat(combined_frames, ignore_index=True)
    scores_frame.to_csv(output_dir / "segment_novelty_scores.csv", index=False)
    combined.to_csv(output_dir / "cluster_novelty_results.csv", index=False)

    flagged = combined.loc[combined["novelty_flag"]].copy()
    priority_order = {"high": 0, "medium": 1}
    flagged["priority_sort"] = flagged["investigation_priority"].map(priority_order)
    queue_columns = [
        "split",
        "segment_id",
        "recording_id",
        "hdbscan_cluster_id",
        "hdbscan_membership_probability",
        "hdbscan_status",
        "novelty_score",
        "novelty_percentile_vs_train",
        "investigation_priority",
        "location_names",
        "observation_date",
        "audio_path",
        "audio_storage_type",
        "audio_parquet_row_index",
        "parquet_file",
        "parquet_row_index",
        "segment_start_seconds",
        "segment_valid_seconds",
        "source_sample_rate_hz",
        "channel_count",
    ]
    queue = (
        flagged.sort_values(
            ["priority_sort", "novelty_score"],
            ascending=[True, False],
            kind="stable",
        )
        .drop_duplicates("recording_id", keep="first")
        .drop(columns=["priority_sort"])
        .loc[:, queue_columns]
        .reset_index(drop=True)
    )
    queue.insert(0, "investigation_rank", np.arange(1, len(queue) + 1))
    queue.to_csv(output_dir / "investigation_queue.csv", index=False)

    train_evaluation = _training_cluster_evaluation(
        split_scores["train"],
        split_flags["train"],
        split_clusters["train"],
    )
    _write_plots(output_dir, combined, threshold)
    joblib.dump(model, output_dir / "isolation_forest.joblib")
    completed_at = _utc_now()
    elapsed_seconds = time.perf_counter() - timer

    priority_counts = queue["investigation_priority"].value_counts().to_dict()
    summary = {
        "status": "complete",
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "execution_seconds": float(elapsed_seconds),
        "input_representation": {
            "model": "laion/clap-htsat-unfused",
            "dimension": int(config["embedding_dim"]),
            "source": "Stage 4 embeddings.npy",
            "normalization": "L2-normalized; any correction is in-memory only",
        },
        "fit_split": "train",
        "training_segments": int(len(matrices["train"])),
        "total_scored_segments": int(len(combined)),
        "validated_recordings_by_split": {
            split: int(len(recording_ids[split])) for split in SPLITS
        },
        "training_contamination": float(model_config["contamination"]),
        "novelty_threshold": threshold,
        "score_interpretation": "Higher -score_samples means more anomalous.",
        "score_summary_by_split": score_summary,
        "flagged_segments": int(len(flagged)),
        "flagged_recordings": int(queue["recording_id"].nunique()),
        "high_priority_recordings": int(
            queue.loc[queue["investigation_priority"] == "high", "recording_id"].nunique()
        ),
        "queue_priority_counts": {
            key: int(value) for key, value in priority_counts.items()
        },
        "training_cluster_evaluation": train_evaluation,
        "training_hdbscan_clusters": int(
            len(set(split_clusters["train"].tolist()) - {-1})
        ),
        "training_hdbscan_noise_segments": int((split_clusters["train"] == -1).sum()),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "config": config,
        "outputs": [
            "isolation_forest.joblib",
            "segment_novelty_scores.csv",
            "cluster_novelty_results.csv",
            "investigation_queue.csv",
            "novelty_score_distribution.png",
            "cluster_vs_novelty.png",
            "stage7_report.md",
            "stage7_config.json",
            "stage7.log",
            "extract_stage7_audio.py",
        ],
        "species_labels_used_for_fitting": False,
        "verified_novelty_ground_truth_available": False,
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    _write_report(output_dir, config, summary)
    LOGGER.info(
        "Stage 7 complete segments=%d flagged_recordings=%d high_priority=%d "
        "elapsed=%.1fs output=%s",
        len(combined),
        summary["flagged_recordings"],
        summary["high_priority_recordings"],
        elapsed_seconds,
        output_dir,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and score Stage 7 novelty detection.")
    parser.add_argument("--config", type=Path, default=OUTPUT_DIR / "stage7_config.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    run(arguments.config, arguments.output_dir)


if __name__ == "__main__":
    main()
