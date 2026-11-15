# Species-class weighting comparison

Balanced and unweighted logistic regression (C=1) were compared on the same 5-fold grouped splits across 2 repeats. All scaling and fitting used training groups only. Means below average fold-level metrics; Fold SD is the standard deviation across the 10 outer folds, not a confidence interval.

| Features | Class weighting | Mean fold accuracy, all records | Fold SD | Mean fold accuracy, seen species | Mean fold macro-F1, seen species | Mean coverage |
|---|---|---:|---:|---:|---:|---:|
| clap | balanced | 0.352 | 0.146 | 0.365 | 0.293 | 95.7% |
| clap | unweighted | 0.361 | 0.138 | 0.375 | 0.287 | 95.7% |
| full_band | balanced | 0.244 | 0.115 | 0.253 | 0.189 | 95.7% |
| full_band | unweighted | 0.269 | 0.113 | 0.278 | 0.154 | 95.7% |

## Paired unweighted-minus-balanced difference

| Features | Accuracy difference (mean ± fold SD) | Macro-F1 difference (mean ± fold SD) |
|---|---:|---:|
| clap | +0.009 ± 0.014 | -0.005 ± 0.026 |
| full_band | +0.024 ± 0.016 | -0.035 ± 0.035 |

Fold-level measurements are in `watkins_species_cv_weighting_folds.csv`; paired differences are in `watkins_species_cv_weighting_deltas.csv`.
