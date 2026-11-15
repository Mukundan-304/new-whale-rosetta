# Watkins Stage 2 preprocessing validation

- Source recordings with features: 15,248.
- Segment windows: 29,932; fixed duration 5 s.
- Short recordings kept and zero-padded: 13,455.
- Final partial windows zero-padded: 15,246.
- Processing failures: 0; see `processing_errors.csv`.
- Header/metadata sample-rate mismatches found during this pass: 74; decoded audio header was authoritative.
- Feature shape: 128 Mel bins × 934 frames, float16.
- Cached partitions written: 504; reused without recomputation: 0.

## Parameter rationale

- Target sample rate: 192,000 Hz. The existing metadata inventory contains audio through 192 kHz; a 192 kHz target has a 96 kHz Nyquist and avoids the irreversible 24 kHz ceiling imposed by the prior 48 kHz baseline. Resampling does not invent frequencies absent from a source recording.
- Segment duration: 5 seconds, configurable in `watkins_preprocessing_config.json`; shorter recordings and partial final windows are retained with zero padding and valid-duration metadata.
- STFT: n_fft=4096 (~21.3 ms), hop=1024 (~5.3 ms), 128 Mel bins from 20 Hz to 96000 Hz.
- Resampled channels are arithmetically averaged into one mono waveform before segmentation. The source channel count and processed mono channel count are retained in the manifest. The audited full-dataset cache contains mono source recordings; the downmix path supports future multichannel records.
- Each mono recording is peak-normalized once to 0.99 before segmentation. The original peak/RMS, normalized RMS, and applied gain are preserved in segment metadata. This equalizes each recording's maximum level but intentionally removes between-recording absolute peak-level differences; use the stored raw levels if amplitude/domain analyses are needed.
- No high-pass filter, denoising, silence removal, loudness gate, or class rebalancing is applied. Low-amplitude and rare-species recordings are not discarded.

## Outputs

- `features/`: compressed float16 log-Mel arrays partitioned by source shard, row group, and record batch.
- `segment_metadata.csv`: one row per segment with source recording ID, label, recording metadata, timing, actual sample rate, and feature partition/index.
- `processing_errors.csv`: per-record failures; failed recordings do not stop the rest of the shard.
- `spectrogram_previews/`: representative short, typical, and long-recording examples where available.
- `preprocessing_validation.csv`: machine-readable counts and processing configuration.
- `cache_state/`: source-file/config fingerprints enabling safe incremental reruns.

Stage 3 must split by the original recording/group before using segments; segment rows from one recording must never cross split boundaries.
