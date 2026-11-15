# Watkins full-dataset frozen representation baseline

- Source recordings: 15,248; location/date/sample-rate metadata and audio were read without modifying Parquet shards.
- Location-connected groups: 154; split: 12,198 train / 3,050 validation (20.0%).
- Groups represented: 105 train / 49 validation. The split targets the requested validation size, then favors more held-out groups when multiple group combinations are equally close.
- Rows without usable location metadata stay in training. Exact duplicate audio payloads are linked into one group so they cannot cross the split.
- Species tiers in original records: {'moderate': 1758, 'dominant': 13118, 'rare': 232, 'unlabeled': 140}.
- Validation species tiers: {'dominant': 2719, 'moderate': 264, 'rare': 29, 'unlabeled': 38}.
- Non-overlapping chunks: 21,951; CLAP embeddings were extracted from a frozen pretrained encoder; no model weights were trained.
- A parallel 64-bin log-power representation spans nominally 50 Hz–24 kHz. It is an acoustic baseline, not a learned embedding.
- CLAP operates at 48 kHz and focuses on approximately 50 Hz–14 kHz; up-sampling lower-rate recordings cannot restore missing frequencies.
- K-means is fit only on training-record features; held-out recordings are assigned to the fitted centroids. All chunks from a recording stay together and are averaged before record-level clustering.
- Species labels are not model targets. Tier metrics summarize species/cluster and recording-condition associations.
- Treat clusters as candidate acoustic groups, not confirmed call types. Location is coarse metadata, not verified session identity.

See `watkins_tier_evaluation.csv`, `watkins_cluster_scores.csv`, `watkins_cluster_assignments.csv`, and `watkins_location_split.csv`.
