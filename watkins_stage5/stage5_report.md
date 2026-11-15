# Stage 5 — Waveform-level self-supervised learning

- Status: **complete**.
- Method: SimCLR-style NT-Xent on two independent augmented waveform views.
- Encoder: `laion/clap-htsat-unfused`; all encoder parameters frozen.
- Projection head: 512 → 512 → 128 dimensions; trainable parameters: 328,320.
- Device: `cuda`; mixed precision: True; peak allocated CUDA memory: 959.5 MiB.
- Best checkpoint: `D:\marine\watkins_stage5\checkpoints\best.pt`.
- Training duration: 11413.2 s.
- Training segments: 21,191 per epoch; validation segments: 4,479 per epoch.
- Total training examples processed across completed epochs: 63,573.
- Best epoch: 3; best validation contrastive loss: 1.060458.
- Final epoch training loss: 0.249439; validation loss: 1.060458.

## Method and augmentations

For each segment, the original Stage 4 waveform reconstruction is resampled to 48 kHz mono and sent through the released CLAP processor. No species or environment labels are read by the training loss. The CLAP feature extractor is frozen; only the small MLP projection head is optimized. NT-Xent uses temperature 0.1.

- Independent zero-filled time shifts: up to ±50 ms.
- Independent gain change: ±2 dB.
- Mild additive Gaussian noise: per-view SNR uniformly sampled from 35–40 dB.
- No pitch shift, frequency warp, time/frequency masking, or label-conditioned augmentation is applied.

## Held-out evaluation

Linear probes are fit on training labels only and scored on validation/test. Clustering is fit on training embeddings. Test labels are used only for the post-training comparison. Metrics are descriptive; they are not claims of biological categories or unknown vocalization discovery.

| Representation | Val accuracy | Val macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| stage4_clap | 0.8259 | 0.4108 | 0.8393 | 0.4088 |
| stage5_ssl | 0.7156 | 0.3147 | 0.7166 | 0.3112 |

| Representation | Train silhouette (sampled) | Train species NMI | Train species ARI |
|---|---:|---:|---:|
| stage4_clap | 0.1352 | 0.4618 | 0.2479 |
| stage5_ssl | 0.0955 | 0.3293 | 0.1918 |

Cosine similarities use identical, distinct-recording test pairs for both representations:

| Pair type | Representation | Pairs | Mean cosine | Median cosine |
|---|---|---:|---:|---:|
| same_species | stage4_clap | 2000 | 0.5613 | 0.5699 |
| same_species | stage5_ssl | 2000 | 0.3884 | 0.3971 |
| different_species | stage4_clap | 2000 | 0.3215 | 0.3175 |
| different_species | stage5_ssl | 2000 | 0.1227 | 0.1136 |
| same_species_same_location | stage4_clap | 2000 | 0.6276 | 0.6531 |
| same_species_same_location | stage5_ssl | 2000 | 0.4706 | 0.5107 |
| same_species_different_location | stage4_clap | 2000 | 0.5042 | 0.5140 |
| same_species_different_location | stage5_ssl | 2000 | 0.3182 | 0.3301 |

PCA visualization: `pca_baseline_vs_ssl.png`; PCA coordinates: `pca_coordinates.csv`. PCA is fit on the training split only. UMAP status: complete; plot: `umap_baseline_vs_ssl.png` when complete.

## Environment metadata and limitations

Location domain and observation-date metadata are available in the segment metadata; the sound/channel source code is not a verified device ID. Recording-session and recorder/device identifiers are not present, so session/device invariance cannot be tested. Location is only a coarse environment proxy and is correlated with species. Same-species same-location versus cross-location similarities are reported below as exploratory evidence only; they do not establish environment invariance.

| Test pair subset | Representation | Pairs | Mean cosine |
|---|---|---:|---:|
| same_species_same_location | stage4_clap | 2000 | 0.6276 |
| same_species_same_location | stage5_ssl | 2000 | 0.4706 |
| same_species_different_location | stage4_clap | 2000 | 0.5042 |
| same_species_different_location | stage5_ssl | 2000 | 0.3182 |

Metadata inventory (known segments / split size; unique non-empty values):

| Split | Location domains | Observation dates | Sound/channel source codes |
|---|---|---|---|
| train | 21080/21191 (222 values) | 18938/21191 (259 values) | 21186/21191 (37 values) |
| validation | 4470/4479 (176 values) | 3560/4479 (227 values) | 4476/4479 (26 values) |
| test | 4251/4262 (178 values) | 3956/4262 (230 values) | 4260/4262 (28 values) |

No claim of environment invariance is made. Frozen CLAP may constrain the representation, and gains here apply only to the learned projection and these evaluation protocols. Silhouette and UMAP depend on representation geometry and sampling; they should not be read as biological validity.
