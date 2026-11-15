# Watkins dataset imbalance and quality summary

- Parquet shards: 9
- Records scanned: 15,248
- Inferred species label field: `species_code`
- Inferred audio field: `audio`
- Audio representation types: {'bytes': 15248}
- Records with decoded duration: 15,248
- Total decoded audio duration: 28.24 hours
- Missing labels: 140; missing primary labels with fallback: 0
- Missing audio: 0; empty payloads: 0; corrupt payloads: 0
- Duplicate audio payloads: 13; repeated record IDs: 0
- Audio sample-rate metadata differs from the decoded file header in 74 records.

## Imbalance tiers

Labeled species are divided into count-based tertiles: rare at or below the 33rd percentile, moderate above that through the 67th percentile, and dominant above the 67th percentile. Thresholds are dataset-derived, not biological definitions.

- Rare: 18 species, 232 records
- Moderate: 17 species, 1,758 records
- Dominant: 18 species, 13,118 records
- Unlabeled display-name groups: 4 groups, 140 records

## Recommended training strategy

Keep all original records and their natural distribution for the initial self-supervised baseline. Do not use species labels as prediction targets, synthesize audio with SMOTE, or delete rare/common-species records. The optional `SpeciesBalancedBatchSampler` creates batches with equal samples per species, shuffles each species' rows, and never repeats a row within an epoch; singleton/rare rows are therefore not oversampled. Configure one sample per species per batch (`batch_size == classes_per_batch`) to include singleton taxa. If more examples per species are requested, species with fewer records are ineligible for balanced batches. Some leftovers may not fit a complete balanced batch and remain available to the natural-distribution loader.

For validation, split by a verified recording session, location, or recording source using `grouped_split_indices`. Missing group metadata remains in training rather than being randomly split. Do not treat clip IDs as sessions unless their semantics are verified.

Evaluate clusters separately by common/moderate/rare tier. Species-cluster association is only a taxonomic diagnostic, not validation of call types. Include recording-domain association where those labels exist; inspect representative audio and report limitations.

## Generated outputs

- `species_distribution.csv`: per-label counts, tiers, durations, and quality/metadata coverage.
- `dataset_quality_report.csv`: dataset-level schema, missing/corrupt/duplicate audio, and metadata summary.
- `species_distribution.png`
- `species_duration_distribution.png`

Original Parquet shards were read only and were not modified.
