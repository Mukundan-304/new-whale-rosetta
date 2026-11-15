# Stage 7 — Novelty detection and investigation prioritisation

- Status: **complete**.
- Algorithm: Isolation Forest (`n_estimators=300`, `max_samples=256`, contamination `0.05`).
- Representation: original, L2-normalized Stage 4 CLAP vectors in **512-D** space. No UMAP coordinates or Stage 5 embeddings were used as novelty-model inputs.
- Model fit: training split only (21,191 segments). Validation/test were scored only after fitting.
- Segments scored: 29,932 across train/validation/test.
- High-novelty threshold: 0.473345, derived from the training-fit contamination cutoff.
- Higher novelty score means more anomalous (`-IsolationForest.score_samples`).
- Novelty-flagged recordings: 1,049 unique recordings.
- High-priority recordings (Stage 6 noise **and** high novelty): 141.
- Execution time: 2.9 seconds.

## Input and alignment checks

Stage 4 embedding rows were checked against metadata and each split manifest. Train cluster assignments from Stage 6 were checked against training segment and recording IDs. Recording IDs were checked for overlap across splits to preserve the existing split integrity. Missing Stage 6 labels on validation/test are represented as unavailable—not as noise or as a predicted cluster.

| Split | Segments | Novelty-flagged segments | Unique flagged recordings | Flagged segment rate |
|---|---:|---:|---:|---:|
| train | 21,191 | 1,060 | 738 | 5.00% |
| validation | 4,479 | 269 | 157 | 6.01% |
| test | 4,262 | 248 | 154 | 5.82% |

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

- Noise segments flagged: 199/3,342 (5.95%).
- Clustered segments flagged: 861/17,849 (4.82%).
- Mean novelty score: HDBSCAN noise 0.445713; clustered 0.444300.
- ROC AUC of novelty scores for distinguishing HDBSCAN noise from clustered samples: 0.5131.
- Spearman correlation of novelty score with HDBSCAN noise indicator: 0.0165.

These are descriptive comparisons, not accuracy or precision: HDBSCAN noise is not verified novelty ground truth. No novelty labels or species labels were used to fit or tune Isolation Forest.

## Researcher audio access

`investigation_queue.csv` includes the source Parquet path, Parquet row, valid segment offset/duration, original sample rate, and channel count. Audio is embedded in Parquet and is not a standalone WAV. To export a listenable crop, run:

```powershell
python watkins_stage7\extract_stage7_audio.py --segment-id "<segment_id from investigation_queue.csv>"
```

The helper verifies the recording ID at the Parquet row and writes a PCM-16 WAV crop at the source sample rate/channels under `watkins_stage7\audio_samples`. The original Parquet and Stage 4/5/6 artifacts are read-only.

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

Seed: 29; Python 3.14.0b1; scikit-learn 1.8.0; NumPy 2.3.5; pandas 2.3.3. The full model and score parameters are recorded in `stage7_config.json` and `run_summary.json`.
