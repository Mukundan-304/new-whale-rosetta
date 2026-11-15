# BE3C frozen representation baseline

- Source recordings embedded: 1,169.
- Ten-second-or-shorter non-overlapping chunks: 1,239.
- Chunk train / validation counts: 967 / 272.
- Source recording sample rates: {23500: 14, 25900: 7, 30000: 83, 40000: 47, 40960: 21, 47600: 36, 60600: 11, 80000: 99, 81600: 176, 81920: 74, 166600: 601}.
- Frozen pretrained encoder: `laion/clap-htsat-unfused`; no model parameters were trained.
- CLAP input sample rate: 48,000 Hz; the model processor uses approximately 50 Hz–14 kHz.
- A companion 64-bin log-power representation spans 50 Hz–24 kHz after resampling.
- Chunks retain the source recording, location-held-out split, and location group so all chunks from a recording remain in its original split.
- The full-band features retain the 14–24 kHz region that CLAP does not represent. They are acoustic baseline features, not learned embeddings.
- Resampling cannot restore frequencies above a source recording's original Nyquist limit.
- Source WAVs are resampled in memory only. Original Parquet files are unchanged.

These features are ready for a train-only clustering baseline and separate validation assignment. Species labels are metadata for later evaluation, not model targets.
