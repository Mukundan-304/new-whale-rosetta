# Watkins species-identification baseline

This is a supervised benchmark using existing species labels and frozen audio features. It does not train the audio encoder.

- All chunks from each recording were averaged into one recording-level feature vector.
- The existing location-held-out training and validation split was preserved.
- Feature scaling and classifiers were fitted on training records only.
- The balanced logistic regression gives each training species equal total class weight.
- The majority-class model is a simple reference baseline.
- Unlabeled records were excluded. Validation species absent from training are counted as unseen and cannot be identified by these classifiers.
- Validation accuracy across all labeled records and accuracy on seen species are reported separately; macro metrics are calculated over species represented in both training and validation.
- This is one held-out split, not yet a final estimate of real-world performance.

## Validation results

| Features | Model | Validation accuracy (all labeled) | Accuracy (seen species) | Macro F1 (seen species) | Species coverage | Unseen validation species |
|---|---|---:|---:|---:|---:|---:|
| clap | majority | 0.308 | 0.308 | 0.015 | 100.0% | 0 |
| clap | balanced_logistic_regression | 0.313 | 0.313 | 0.283 | 100.0% | 0 |
| full_band | majority | 0.308 | 0.308 | 0.015 | 100.0% | 0 |
| full_band | balanced_logistic_regression | 0.144 | 0.144 | 0.150 | 100.0% | 0 |

Per-species results and validation predictions are in `watkins_species_classification_per_species.csv` and `watkins_species_classification_predictions.csv`.
