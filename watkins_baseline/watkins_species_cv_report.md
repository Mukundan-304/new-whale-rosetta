# Watkins species identification: repeated location-grouped evaluation

- Labeled recordings evaluated: 15,108.
- Species labels: 53 across 153 location groups.
- Repeated outer evaluation: 5 folds × 2 repeats.
- Hyperparameters selected by 3-fold grouped cross-validation inside each outer training set.
- All chunks were averaged per original recording. Scaling and model fitting were confined to training folds.
- The primary comparison metric is macro-F1 among species represented in each training fold. Accuracy and the fraction of validation recordings from seen species are also reported.
- 17 species occur in only one location group and cannot be tested as known species when that group is held out.
- Accuracy and macro-F1 are unweighted means across outer folds. The repeat SD describes variation between the two repeat-level means, not a confidence interval.

## Results

| Features | Evaluation | Mean fold accuracy, all records | Repeat SD | Mean fold accuracy, seen species | Mean fold macro-F1, seen species | Mean seen-species coverage |
|---|---|---:|---:|---:|---:|---:|
| clap | fixed_balanced_logistic_c1 | 0.352 | 0.001 | 0.365 | 0.293 | 95.7% |
| clap | majority | 0.180 | 0.000 | 0.186 | 0.012 | 95.7% |
| clap | nested_selected | 0.357 | 0.003 | 0.370 | 0.296 | 95.7% |
| full_band | fixed_balanced_logistic_c1 | 0.244 | 0.004 | 0.253 | 0.189 | 95.7% |
| full_band | majority | 0.180 | 0.000 | 0.186 | 0.012 | 95.7% |
| full_band | nested_selected | 0.255 | 0.005 | 0.263 | 0.182 | 95.7% |

The nested-selected row is an estimate of the full grouped model-selection procedure, not the score of a single fixed final model. The fixed balanced logistic row is directly comparable to the earlier one-split baseline.

## Limitations

- Location groups are a geographic proxy, not verified recording sessions.
- One-location species are not learnable under a location-held-out design; see the per-species results for their coverage.
- Even nested group cross-validation is an internal estimate. Independent recording-site or session metadata would provide a stronger test.

Detailed folds, predictions, per-species scores, confusion pairs, and selected hyperparameters are in the accompanying CSV files.
