# Stage 6 — Unsupervised acoustic pattern discovery

- Status: **complete**.
- Primary representation: Stage 4 `laion/clap-htsat-unfused` embeddings.
- Discovery fit split: **training only**; validation and test embeddings were validated but not clustered.
- Samples clustered: 21,191 training segments (512-D CLAP embeddings).
- Execution duration: 58.3 seconds (from 2026-11-13T14:45:25.589006+00:00 to 2026-11-13T14:46:23.864069+00:00).
- Random seed: 29.
- HDBSCAN clusters (noise excluded): 104.
- Noise: 3,342 segments (15.77%).
- Assigned to a cluster: 17,849 segments (84.23%).

## Input validation

| Split | Segments | Dimensions | Finite | L2-normalized on input | Manifest IDs aligned |
|---|---:|---:|---|---|---|
| train | 21,191 | 512 | yes | True | True |
| validation | 4,479 | 512 | yes | True | True |
| test | 4,262 | 512 | yes | True | True |

Non-unit vectors, if any, were L2-normalized in memory for analysis; no Stage 4 or Stage 5 files were modified.

## UMAP and HDBSCAN settings

- Clustering space: UMAP to 15 dimensions, cosine metric, `n_neighbors=30`, `min_dist=0.0`.
- Visualization: separate 2-D UMAP, cosine metric, `n_neighbors=15`, `min_dist=0.1`.
- HDBSCAN: `min_cluster_size=50`, `min_samples=10`, metric `euclidean`, selection `eom`.
- UMAP/HDBSCAN saw only the unlabeled training embedding matrix; validation and test were not used for clustering or parameter selection.

## Cluster sizes

| Cluster ID | Segments | Share of training split |
|---:|---:|---:|
| 0 | 80 | 0.38% |
| 1 | 137 | 0.65% |
| 2 | 108 | 0.51% |
| 3 | 61 | 0.29% |
| 4 | 419 | 1.98% |
| 5 | 1,408 | 6.64% |
| 6 | 360 | 1.70% |
| 7 | 193 | 0.91% |
| 8 | 65 | 0.31% |
| 9 | 95 | 0.45% |
| 10 | 211 | 1.00% |
| 11 | 69 | 0.33% |
| 12 | 133 | 0.63% |
| 13 | 90 | 0.42% |
| 14 | 66 | 0.31% |
| 15 | 72 | 0.34% |
| 16 | 81 | 0.38% |
| 17 | 424 | 2.00% |
| 18 | 100 | 0.47% |
| 19 | 59 | 0.28% |
| 20 | 224 | 1.06% |
| 21 | 184 | 0.87% |
| 22 | 89 | 0.42% |
| 23 | 106 | 0.50% |
| 24 | 58 | 0.27% |
| 25 | 91 | 0.43% |
| 26 | 115 | 0.54% |
| 27 | 104 | 0.49% |
| 28 | 111 | 0.52% |
| 29 | 77 | 0.36% |
| 30 | 67 | 0.32% |
| 31 | 518 | 2.44% |
| 32 | 158 | 0.75% |
| 33 | 221 | 1.04% |
| 34 | 166 | 0.78% |
| 35 | 399 | 1.88% |
| 36 | 119 | 0.56% |
| 37 | 143 | 0.67% |
| 38 | 60 | 0.28% |
| 39 | 85 | 0.40% |
| 40 | 140 | 0.66% |
| 41 | 53 | 0.25% |
| 42 | 99 | 0.47% |
| 43 | 115 | 0.54% |
| 44 | 76 | 0.36% |
| 45 | 149 | 0.70% |
| 46 | 83 | 0.39% |
| 47 | 104 | 0.49% |
| 48 | 719 | 3.39% |
| 49 | 389 | 1.84% |
| 50 | 66 | 0.31% |
| 51 | 50 | 0.24% |
| 52 | 80 | 0.38% |
| 53 | 51 | 0.24% |
| 54 | 376 | 1.77% |
| 55 | 328 | 1.55% |
| 56 | 55 | 0.26% |
| 57 | 339 | 1.60% |
| 58 | 98 | 0.46% |
| 59 | 90 | 0.42% |
| 60 | 77 | 0.36% |
| 61 | 118 | 0.56% |
| 62 | 57 | 0.27% |
| 63 | 396 | 1.87% |
| 64 | 110 | 0.52% |
| 65 | 784 | 3.70% |
| 66 | 137 | 0.65% |
| 67 | 83 | 0.39% |
| 68 | 126 | 0.59% |
| 69 | 184 | 0.87% |
| 70 | 58 | 0.27% |
| 71 | 64 | 0.30% |
| 72 | 74 | 0.35% |
| 73 | 164 | 0.77% |
| 74 | 342 | 1.61% |
| 75 | 170 | 0.80% |
| 76 | 103 | 0.49% |
| 77 | 115 | 0.54% |
| 78 | 55 | 0.26% |
| 79 | 113 | 0.53% |
| 80 | 141 | 0.67% |
| 81 | 190 | 0.90% |
| 82 | 126 | 0.59% |
| 83 | 124 | 0.59% |
| 84 | 123 | 0.58% |
| 85 | 573 | 2.70% |
| 86 | 63 | 0.30% |
| 87 | 132 | 0.62% |
| 88 | 76 | 0.36% |
| 89 | 168 | 0.79% |
| 90 | 108 | 0.51% |
| 91 | 54 | 0.25% |
| 92 | 211 | 1.00% |
| 93 | 235 | 1.11% |
| 94 | 333 | 1.57% |
| 95 | 91 | 0.43% |
| 96 | 95 | 0.45% |
| 97 | 187 | 0.88% |
| 98 | 75 | 0.35% |
| 99 | 173 | 0.82% |
| 100 | 80 | 0.38% |
| 101 | 113 | 0.53% |
| 102 | 54 | 0.25% |
| 103 | 208 | 0.98% |

Noise is reported separately from the clusters. The cluster assignment file contains a row for every training segment.

## Evaluation

- Sampled non-noise silhouette in the 15-D UMAP space: 0.6294634342193604 (3,000 points). Sampled non-noise points in the 15-D UMAP clustering space.
- Post-clustering species ARI/NMI: 0.17456995074340517 / 0.6193026515822699 on 17,849 clustered labeled training samples. This is exploratory external evaluation; species labels were not inputs to clustering.

## Representative samples

`representative_samples.csv` contains up to 3 training examples per cluster, chosen as nearest members to each cluster centroid in the 15-D UMAP space. Rows include segment and recording IDs plus location/date and audio pointers where available. In this dataset, audio is embedded in Parquet; the listed Parquet path/row and segment offset identify the source but are not standalone WAV files.

## Outputs

- `cluster_assignments.csv` — train IDs, cluster IDs, HDBSCAN membership probabilities, and 2-D coordinates.
- `umap_coordinates.csv` and `umap_clustering_coordinates.npy` — visualization and 15-D clustering coordinates.
- `cluster_sizes.csv`, `representative_samples.csv`, and `run_summary.json`.
- `umap_clusters.png`, `umap_noise.png`, and `cluster_size_distribution.png`.
- `stage6_config.json` and `stage6.log`.
- Representative rows written: 312.

## Interpretation and limitations

These are **candidate acoustic patterns**, not confirmed biological categories, species, communication signals, or meanings. UMAP changes neighborhood geometry and HDBSCAN results depend on its parameters; clusters/noise should be treated as exploratory. The embeddings are CLAP representations (512 dimensions, with the Stage 4 frontend's frequency limitations), and clustering was performed only on the training split. No Stage 4/5 retraining, embedding regeneration, preprocessing, or split modification was performed.

## Reproducibility

Start time: `2026-11-13T14:45:25.589006+00:00`. Completion time: `2026-11-13T14:46:23.864069+00:00`. Total execution: `58.3` seconds. Python `3.14.0b1`, scikit-learn `1.8.0`, UMAP `0.6.0`. Parameters and seed are recorded in `stage6_config.json` and `run_summary.json`.
