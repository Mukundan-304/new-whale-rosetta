# Stage 4 — Pretrained Audio Embedding Baseline

- Status: **complete**.
- Encoder: `laion/clap-htsat-unfused` (checkpoint `model revision not reported`).
- Device: `cuda`; peak allocated CUDA memory 0.67 GiB.
- Embedding size: **512** dimensions; float32 L2-normalized embeddings.
- Extraction elapsed: **1959.5 s** (32.7 min across split timers); analysis elapsed: **76.9 s**.
- Process CPU utilization: mean 105.61%, peak 129.2%; peak process RAM 8.07 GiB; minimum observed system-available RAM 4.50 GiB.
- Sampled GPU utilization: mean 29.6%, peak 100% across 63 `nvidia-smi` samples; peak driver-reported GPU memory 0.85 GiB.

## Encoder selection and input handling

CLAP (`laion/clap-htsat-unfused`) was selected for this executable baseline because its pretrained weights and processor are already available in the repository's Transformers environment, its released frontend accepts 48 kHz waveforms, and the same encoder is already used by the project's Watkins and BE3C baselines. Its released checkpoint is trained on LAION-Audio-630K audio-text pairs, not specifically on underwater recordings. AVES2 BEATs is a more bioacoustics-focused alternative, but its AVEX loader currently requires Python 3.11–3.13 while this Windows environment is Python 3.14; the PANN/AudioSet-style options are also not marine-specific.

The CLAP frontend expects 48 kHz mono input, 10 s windows, 64 Mel bins, and uses frequencies from 50 Hz to 14 kHz. Each segment was reconstructed from the existing Parquet audio using the Stage 2 192 kHz resampling, recording-level peak normalization, and 5 s windowing/padding, then resampled to 48 kHz. CLAP's configured padding extends the 5 s segment to its 10 s input length. Original files and Stage 1–3 outputs were not changed. The model discards frequencies above 14 kHz, so these embeddings do not represent the full bandwidth retained in Stage 2 and require in-domain validation before ecological interpretation.

## Embedding counts and validation

| Split | Segments | Dimensions | Recordings | Species labels |
|---|---:|---:|---:|---:|
| train | 21,191 | 512 | 10,673 | 57 |
| validation | 4,479 | 512 | 2,287 | 53 |
| test | 4,262 | 512 | 2,288 | 51 |

- Total: 29,932 embeddings for 29,932 unique segment IDs.
- Invalid vectors: 0; embedding dimensions are consistent; every output metadata row corresponds to one embedding; all source paths remain valid.
- Duplicate segment IDs and train/validation/test overlap: none.
- Peak CUDA allocator usage: 0.67 GiB. The separate driver-reported GPU memory metric includes display/runtime use.

## Similarity analysis (training split only)

| Pair type | Pairs | Mean cosine | Median cosine | 5th percentile | 95th percentile |
|---|---:|---:|---:|---:|---:|
| same_species | 2,000 | 0.6221 | 0.6408 | 0.2747 | 0.8847 |
| different_species | 2,000 | 0.3593 | 0.3595 | 0.1004 | 0.6342 |

Pairs use distinct original recordings; species are sampled uniformly from eligible labels. These descriptive similarities are not a classifier score or a test-set evaluation.

## PCA and clustering

- PCA was fit using training embeddings only. The first two components explain 0.1437 and 0.1338 of training variance.
- PCA plot: `pca_embeddings.png`; coordinates: `pca_coordinates.csv`.
- UMAP status: **complete** on 10,000 training embeddings.
- K-Means was fit on training embeddings only; validation and test were not used for clustering or model selection.
- MiniBatchKMeans with 20 clusters; sampled train silhouette 0.1365; train species NMI 0.4596; train species ARI 0.2691.

| Cluster | Segments | Five most common labels (count) | Leading-label fraction |
|---:|---:|---|---:|
| 0 | 991 | BA2A:760 | BD15A:74 | BE3D:45 | BD3B:39 | BE9A:21 | 0.767 |
| 1 | 974 | BE9A:194 | BD15A:109 | BD15C:101 | BE7A:94 | BD3B:88 | 0.199 |
| 2 | 1,290 | BD3B:224 | BA2A:177 | BE7A:136 | AC2A:86 | CB1A:84 | 0.174 |
| 3 | 978 | BA2A:886 | BE3D:45 | BE7A:22 | BE3C:6 | BD15C:4 | 0.906 |
| 4 | 1,349 | BA2A:1333 | BE7A:6 | BD3B:5 | CC2A:2 | CB1A:1 | 0.988 |
| 5 | 2,419 | AC1F:2324 | AC1E:39 | AC1A:36 | AB1A:13 | AC2A:4 | 0.961 |
| 6 | 868 | AA3A:256 | AA1A:229 | AC2A:152 | CB1A:46 | AA3B:39 | 0.295 |
| 7 | 760 | BA2A:690 | BD15F:11 | BD15A:11 | BD6A:9 | CC5A:6 | 0.908 |
| 8 | 764 | BD15L:125 | BE3D:90 | BD4A:88 | BD15A:79 | BD3B:71 | 0.164 |
| 9 | 1,000 | BD15A:460 | BD6A:147 | BE3D:112 | BA2A:106 | BD4A:49 | 0.460 |
| 10 | 1,130 | AC2A:749 | AA1A:314 | CC2A:35 | CC4A:10 | CC5A:5 | 0.663 |
| 11 | 845 | BA2A:800 | BD3B:13 | BE3C:6 | BD10A:6 | BE3D:4 | 0.947 |
| 12 | 1,185 | BE7A:559 | AC2A:207 | BE3C:129 | CC14A:76 | CC2A:60 | 0.472 |
| 13 | 738 | BA2A:307 | BD10A:50 | BD4A:47 | BE7A:47 | AC1F:26 | 0.416 |
| 14 | 1,463 | BE7A:1005 | BD15F:123 | BE3C:69 | BD19D:53 | BE3D:30 | 0.687 |
| 15 | 1,262 | BA2A:731 | CC5A:164 | BD15B:60 | BD15A:52 | BE3D:38 | 0.579 |
| 16 | 1,205 | BD15C:240 | BE3C:170 | BD15L:141 | BD3B:132 | BD6A:80 | 0.199 |
| 17 | 729 | BE3D:284 | BA2A:223 | BD15L:103 | BE3C:78 | CC5A:19 | 0.390 |
| 18 | 913 | BE3C:377 | BD15B:116 | BD6A:88 | BD3B:78 | BA2A:72 | 0.413 |
| 19 | 328 | BA2A:274 | [UNLABELED] WestIndianManatee:37 | CB1A:9 | CC5A:6 | BD6H:1 | 0.835 |

These clusters are candidate acoustic groupings only. They do not prove communication patterns, animal language, or biological categories.

## Generated outputs

- `train/`, `validation/`, and `test/`: `embeddings.npy`, `metadata.csv`, `completed.npy`, and resumable `progress.json`.
- `encoder_configuration.json`, `generation_summary.json`, `stage4.log`.
- `quality_checks.json`, `cosine_similarity_pairs.csv`, `cosine_similarity_summary.csv`.
- `pca_embeddings.png`, `pca_coordinates.csv`, `train_cluster_assignments.csv`, `train_cluster_species_composition.csv`.
- `umap_train_embeddings.png` and `umap_train_coordinates.csv`.

## Limitations

- CLAP was not trained specifically on Watkins underwater mammal audio; the similarity and clustering results are exploratory.
- Frequencies above 14 kHz are not represented in CLAP embeddings.
- PCA/UMAP plots are projections and do not establish separability or environment invariance. Held-out validation/test data are preserved for later evaluation.
- No CNN–Transformer, self-supervised training, or Stage 5 work was run.

## Processing times

- train: 1285.0 s.
- validation: 340.8 s.
- test: 333.7 s.
- Total extraction: 1959.5 s.
- Baseline analysis: 76.9 s.
