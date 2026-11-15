# new-whale-rosetta

Marine-mammal sound analysis on the Watkins Marine Mammal Sound Database (plus a BE3C feasibility study): preprocessing, splits, embeddings, self-supervised learning, clustering/discovery, novelty detection, robustness and temporal analysis.

## What is in this repository

| Content | Location |
|---|---|
| Code, tests, configs, reports | repository root, `watkins_stage*/` |
| **Audio recordings** (example/QC clips) | `sample.wav`, `be3c_feasibility/be3c_examples/`, `be3c_feasibility/be3c_audio_qc_examples/`, `be3c_feasibility/be3c_cluster_examples/`, `watkins_stage7/audio_samples/` |
| **Model weights** | `watkins_stage5/checkpoints/` (`best.pt`, `latest.pt`), `watkins_stage10/checkpoints/best.pt`, `watkins_stage7/isolation_forest.joblib` |
| **Feature arrays** (`.npz` / `.npy`) | `watkins_preprocessed/features/`, `be3c_feasibility/be3c_embedding_features.npz`, `watkins_baseline/watkins_embedding_features.npz`, `watkins_stage{4,5,10}/{train,validation,test}/embeddings.npy`, `watkins_stage6/umap_clustering_coordinates.npy` |
| **Experiment outputs** (reports, metrics, plots, logs-free summaries) | `watkins_baseline/`, `watkins_splits/`, `watkins_stage4/` … `watkins_stage10/`, `be3c_feasibility/` |

## Not included: the dataset

The raw dataset folder `watkins-marine-mammal-full-cuts/` (about 20 GB) is **intentionally not pushed** and is listed in `.gitignore`. Download it separately from its source and place it in the repository root under that name before re-running the pipeline.

## Git LFS

Three feature files exceed GitHub's 100 MB limit and are stored with Git LFS (`.gitattributes`):

- `watkins_preprocessed/features/train-00006-of-00009-rg000-batch000009.npz`
- `watkins_preprocessed/features/train-00006-of-00009-rg000-batch000010.npz`
- `watkins_preprocessed/features/train-00007-of-00009-rg004-batch000007.npz`

Install [Git LFS](https://git-lfs.com) and run `git lfs pull` after cloning to fetch them.

## Pipeline stages

1. `preprocess_watkins_audio.py` – audio preprocessing and feature extraction
2. `split_watkins_dataset.py` – train/validation/test splits (`watkins_splits/`)
3. `watkins_embedding_baseline.py` – baseline embeddings (`watkins_baseline/`)
4. `watkins_pretrained_embeddings.py` – pretrained encoder embeddings (`watkins_stage4/`)
5. `watkins_stage5_ssl.py`, `watkins_stage5_evaluate.py` – self-supervised learning (`watkins_stage5/`)
6. Discovery / clustering (`watkins_stage6/`)
7. Novelty detection (`watkins_stage7/`)
8. Robustness (`watkins_stage8/`)
9. Temporal analysis (`watkins_stage9/`)
10. CNN-Transformer model (`watkins_stage10/`)

Run the tests with `pytest`. Dependencies: `requirements-be3c-baseline.txt`, `requirements-stage4.txt`.
