# Watkins dataset split report

## Standard recording-grouped split

- All segments from an original recording remain together.
- Exact duplicate source-audio payloads are grouped together using the verified SHA-256 values in `D:\marine\watkins_baseline\watkins_location_split.csv`.
- StratifiedGroupKFold is used to form deterministic folds; whole folds are selected to best match the requested segment and recording proportions and species distributions.
- Rare species are not physically oversampled or undersampled. Species with fewer than three independent audio groups cannot appear in all three splits.

| Split | Recordings | Recording % | Segments | Segment % |
|---|---:|---:|---:|---:|
| train | 10,673 | 70.00% | 21,191 | 70.80% |
| validation | 2,287 | 15.00% | 4,479 | 14.96% |
| test | 2,288 | 15.01% | 4,262 | 14.24% |

The exact 70/15/15 ratios are targets. Group sizes, duplicate audio constraints, and species stratification can cause small deviations; actual percentages are reported above.

## Separate location-held-out evaluation split

- Every exact normalized location-name group stays in one split. Coordinates are used only when a name is absent; exact duplicate audio is tied across locations to prevent content leakage.
- Location metadata is available for 15,207 of 15,248 recordings (99.73%). There are 212 distinct normalized names and 14 coordinate-only domains.
- This can test generalization to held-out *locations*, but the dataset has no explicit recording-session or recorder/device identifier. Location is only an environment proxy; this split does not establish generalization across devices or sessions.
- For strict location-held-out evaluation, score only examples with known location metadata in the environment test split. Rows without location metadata remain available but cannot substantiate a held-out-location claim.
- The environment test contains 2,368 recordings with known location. Its validation and test splits contain 22 and 22 of 57 species labels. Location groups have highly unequal sizes and are correlated with species, so these environment-held-out percentages and species coverage are less balanced than the standard split. Treat this as a secondary generalization check, not as the main training/validation partition.

| Split | Recordings | Recording % | Segments | Segment % |
|---|---:|---:|---:|---:|
| train | 11,495 | 75.39% | 24,874 | 83.10% |
| validation | 1,352 | 8.87% | 1,907 | 6.37% |
| test | 2,401 | 15.75% | 3,151 | 10.53% |

## Paths and metadata

- `audio_path` points to the source Parquet shard because the original audio is embedded in that file; `audio_parquet_row_index` identifies the record. It is not a standalone WAV path.
- `feature_path` and `feature_index` point to each segment's cached log-Mel array. `spectrogram_path` is blank because individual spectrogram image files are not stored for every segment.
- Species balancing is not applied to validation or test. Use the existing balanced batch sampler for training only.
- Per-split species counts are in `species_distribution.csv`; missing metadata is in `metadata_missingness.csv`; full assignments and rare-species coverage are in each `split_summary.json`.
