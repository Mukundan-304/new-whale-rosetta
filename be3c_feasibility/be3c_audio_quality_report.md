# BE3C audio quality check

- Records checked against the location split: 1,169.
- Train / validation: 935 / 234.
- Audio decode status: {'decoded': 1169}.
- Duration, seconds (10th / median / 90th percentile): 0.82 / 2.06 / 3.35; range 0.09–215.67.
- Recordings shorter than 1 second: 160; shorter than 3 seconds: 967.
- Header sample rates: {23500: 14, 25900: 7, 30000: 83, 40000: 47, 40960: 21, 47600: 36, 60600: 11, 80000: 99, 81600: 176, 81920: 74, 166600: 601}.
- Header/metadata sample-rate mismatches: 1 (5402400A).
- Silent by RMS threshold (1e-06): 0.
- Any samples at or above 0.999 full scale: 6 recordings; recordings with at least 1% such samples: 0.
- Recordings with >1% of measured spectral power above 14 kHz: 813; with >10%: 389; dominant spectral peak above 14 kHz: 2.
- Exact duplicate audio payloads: 0 duplicate groups (0 records involved).
- Repeated non-empty record numbers: 0.

These checks confirm files decode and flag basic signal issues; they cannot confirm that clips contain identifiable whale calls or are suitable for a specific research question. Spectral power is not a call detector. Because many recordings contain measurable energy above 14 kHz, do not rely on a pretrained representation that discards that band as the only baseline. Listen to the provided examples before training.

The original Parquet files were read only and were not modified.
