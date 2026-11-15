# BE3C unsupervised clustering baseline

- Record-level features: 1,169 source recordings; chunk features were averaged per original recording to avoid giving longer recordings extra clustering weight.
- K-means was fitted using training records only; validation records were assigned to the nearest training centroid.
- The number of clusters was selected separately per representation by maximum training-only silhouette score among the tested values.
- Selected cluster counts: CLAP 2; 50 Hz–24 kHz acoustic features 2.
- Training silhouette scores at the selected counts: CLAP 0.269; full-band 0.313.
- Train location/sample-rate NMI: CLAP 0.319/0.226; full-band 0.070/0.051.
- Validation location/sample-rate NMI: CLAP 0.061/0.143; full-band 0.024/0.089.
- Train cluster sizes (cluster 0/1): CLAP 137/798; full-band 276/659.
- Validation cluster sizes (cluster 0/1): CLAP 19/215; full-band 41/193.
- Location and sample-rate normalized mutual information values are diagnostics for recording-condition association, not model targets.
- This focused set contains one species, so it cannot assess common, moderate, and rare species performance. Treat clusters as candidate acoustic patterns, not confirmed call types.

See `be3c_cluster_scores.csv` for every tested cluster count and `be3c_cluster_assignments.csv` for the selected train/validation clusters. Run `make_be3c_cluster_previews.py` to export representative WAVs listed in `be3c_cluster_examples.csv`.
