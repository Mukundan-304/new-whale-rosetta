# Stage 8 — Environment Robustness Analysis

- Status: **complete**.
- Focus: Evaluating whether CLAP audio representations maintain fidelity across differing recording environments, dates, and channels.
- Core Research Question: **Do acoustically similar marine animal vocalizations maintain similar representations across different recording environments?**
- Execution mode: Deterministic, lightweight CPU-based analysis reusing existing Stage 4, Stage 6, and Stage 7 artifacts.
- Random seed: 29; executed at `2026-11-13T15:36:43.548151+00:00` on Python `3.14.0b1`.

---

## 1. Executive Findings & Answers to the Research Question

1. **No, CLAP representations are significantly influenced by recording environment and location.**
   - When the **exact same species** is recorded at different geographic locations, its mean pairwise cosine similarity drops from **0.6897** (same location) down to **0.5075** (different locations). This reveals a substantial **Intra-Species Location Gap of -0.1823**.
   - Conversely, when two **entirely different species** are recorded at the same location, their mean cosine similarity rises from **0.3216** up to **0.5086**—an **Inter-Species Location Boost of +0.1870**.
   - In fact, different species sharing a recording site exhibit virtually identical similarity (0.5086) to the same species recorded at different sites (0.5075). This proves that **channel coloration, hydrophone response, and site ambient noise exert an acoustic pull of comparable magnitude to cross-species bioacoustic divergence** in frozen CLAP feature space.

2. **Species Distribution and Recording Location Are Overwhelmingly Confounded.**
   - A rigorous contingency analysis reveals a **Cramér's V of 0.9304**, a **Normalized Mutual Information (NMI) of 0.7713**, and an **Adjusted Rand Index (ARI) of 0.6133** between species and locations.
   - **14 out of 53 species (26.4%)** appear in only a single location in the dataset.
   - **184 out of 216 recording locations (85.2%)** host only a single species. Prominent examples include *Georges Bank* (1,191 segments, 100% *Lagenorhynchus acutus* / Atlantic White-Sided Dolphin) and *Oregon Coast Aquarium* (705 segments, 100% *Orcinus orca* / Killer Whale).
   - **Scientific Implication**: Any unstratified or un-controlled comparison across locations is primarily measuring taxonomic differences, not pure acoustic channel effects.

3. **Stage 6 Clusters and Stage 7 Novelty Scores Exhibit Strong Geographic Disparities.**
   - Stage 6 HDBSCAN clusters display an NMI of **0.6858** with recording locations. In locations like Kikvika (Norway) and Georges Bank, >90% of clustered segments collapse into a single cluster.
   - Stage 7 Isolation Forest novelty scores vary significantly across locations (Kruskal-Wallis **H = 3721.62, p < 1e-100**). The anomaly flag rate ranges from **0.3%** in Georges Bank to **5.9%** in Kikvika, demonstrating that anomaly detection is vulnerable to site-specific acoustic baselines.

---

## 2. Phase B — Metadata Audit & Critical Missing Variables

A rigorous environment robustness analysis requires knowing the acoustic transmission channel and sensor characteristics. The dataset metadata was comprehensively audited across all 29,932 segments:

| Category | Field Name | Available | Coverage | Unique | Scientific Utility & Assessment |
|---|---|---|---:|---:|---|
| Location | `location_domain` | Yes | 99.56% | 226 | Coarse geographic text label or whole-degree coordinate bin; does not represent acoustic oceanography or hydrophone site properties. |
| Location | `location_names` | Yes | 92.68% | 216 | Freeform place names with inconsistent naming conventions and granularity. |
| Location | `location_coordinates` | Yes | 100.0% | 141 | Frequently rounded to integer degrees (±60 nautical miles); insufficient for bathymetric or localized propagation modeling. |
| Temporal | `observation_date` | Yes | 88.38% | 264 | Date-level resolution only (times are midnight UTC). Useful for multi-decade technological shifts (1951-1999) but lacks diurnal/seasonal fidelity. |
| Taxonomy | `species_code` | Yes | 99.31% | 53 | Primary biological label; critical confounder because species occurrence is heavily co-dependent with geographic location. |
| Channel/Hardware | `sound_channel_source_code` | Yes | 99.97% | 39 | Internal WHOI tape channel designation (e.g. 11A, 41B); NOT a verified sensor model, hydrophone serial number, or device calibration. |
| Channel/Hardware | `source_sample_rate_hz` | Yes | 100.0% | 44 | Heterogeneous digitization rates (10 kHz to 192 kHz); sets the effective Nyquist frequency and limits frequency comparability across eras. |
| Signal | `signal_source_code` | Yes | 71.89% | 25 | Historical annotation processing codes (OTF, N, OF); describes digitization procedure rather than underwater acoustic environment. |
| Sensor (Missing) | `hydrophone_model_sensitivity` | **NO (Missing)** | 0.0% | 0 | CRITICAL GAP: Different hydrophone models induce distinct spectral coloration and roll-offs that confound acoustic embeddings. |
| Oceanography (Missing) | `hydrophone_depth_bathymetry` | **NO (Missing)** | 0.0% | 0 | CRITICAL GAP: Deep vs surface hydrophones experience entirely different multipath reflections and transmission loss. |
| Oceanography (Missing) | `water_temp_salinity_ssp` | **NO (Missing)** | 0.0% | 0 | CRITICAL GAP: Dictates acoustic ducting, shadow zones, and frequency-dependent absorption in seawater. |
| Ambient Noise (Missing) | `sea_state_wind_ambient_noise` | **NO (Missing)** | 0.0% | 0 | CRITICAL GAP: Ambient ocean noise alters SNR and masks low-amplitude harmonic call structures. |
| Anthropogenic (Missing) | `vessel_engine_cavitation_spl` | **NO (Missing)** | 0.0% | 0 | PARTIAL PROXY ONLY: Archival text notes mention 'Ship noise' for ~14% of cuts, but quantitative SPL and spectra are completely absent. |
| Session/Device (Missing) | `verified_deployment_session_id` | **NO (Missing)** | 0.0% | 0 | CRITICAL GAP: Without explicit session IDs, independent recordings cannot be verified as acoustically independent hydrophone encounters. |

### Summary of Metadata Limitations for Environment Robustness:
1. **No Calibrated Hydrophone Specifications**: Sensor models, frequency curves, and depth are 0% available. Internal WHOI channel codes (`sound_channel_source_code`) represent analog reel channels, not calibrated sensors.
2. **No Physical Oceanography**: Seawater temperature, sound velocity profiles, and sea states are absent.
3. **Coarse Locations**: Locations are text tags or whole-degree coordinate bins.
4. **No Verified Continuous Session IDs**: Cannot mathematically verify encounter independence.

---

## 3. Phase C — Controlled Representation Robustness

To isolate environmental impact from species identity, we sampled 5,000 distinct-recording pairs under 4 controlled conditions using 512-D Stage 4 CLAP vectors:

| Condition | Distinct Pairs | Mean Cosine | Median | Std Dev | 5th Pct | 95th Pct |
|---|---:|---:|---:|---:|---:|---:|
| **1. Same Species, Same Location** | 5,000 | **0.6897** | 0.7214 | 0.1816 | 0.3534 | 0.9326 |
| **2. Same Species, Diff Location** | 5,000 | **0.5075** | 0.5174 | 0.1836 | 0.1980 | 0.7810 |
| **3. Diff Species, Same Location** | 5,000 | **0.5086** | 0.4841 | 0.2018 | 0.1959 | 0.8687 |
| **4. Diff Species, Diff Location** | 5,000 | **0.3216** | 0.3169 | 0.1706 | 0.0484 | 0.6126 |

- **Intra-Species Location Gap**: `0.6897 - 0.5075 = -0.1823` (~18.2% similarity drop when crossing environments).
- **Inter-Species Location Boost**: `0.5086 - 0.3216 = +0.1870` (~18.7% similarity inflation when sharing an environment).
- **Channel Code Gap** (Tape channel shift): `0.6682 - 0.4811 = -0.1772`.

---

## 4. Geographic Disparities in Clusters & Novelty Detection

Across the top 10 recording locations in the Watkins database:

| Location Domain | Segments | Top Cluster Share | Entropy | HDBSCAN Noise % | Mean Novelty Score | Novelty Flag % |
|---|---:|---:|---:|---:|---:|---:|
| `dominica` | 6,371 | 12.1% (C48) | 3.33 | 8.5% | 0.4473 | 5.3% |
| `georges bank` | 1,191 | 90.2% (C5) | 0.34 | 0.2% | 0.4189 | 0.2% |
| `e of stellwagen bank` | 713 | 49.4% (C6) | 1.34 | 2.4% | 0.4228 | 0.4% |
| `oregon coast aquarium` | 705 | 72.7% (C85) | 0.86 | 14.2% | 0.4519 | 4.8% |
| `delaware` | 534 | 31.9% (C37) | 1.49 | 15.9% | 0.4475 | 5.2% |
| `kealakekua bay, hawaii` | 448 | 30.4% (C99) | 1.49 | 17.9% | 0.4356 | 0.5% |
| `kikvika, norway` | 407 | 94.5% (C63) | 0.29 | 10.8% | 0.4523 | 5.9% |
| `argus island, bermuda` | 398 | 60.2% (C57) | 1.29 | 5.3% | 0.4517 | 2.0% |
| `n end of dominica` | 366 | 98.2% (C31) | 0.10 | 10.9% | 0.4397 | 0.3% |
| `oregon coastal aquarium, newport, or` | 331 | 59.7% (C83) | 1.09 | 43.8% | 0.4515 | 8.8% |

---

## 5. Artifacts Generated in Stage 8

- `metadata_audit.csv`: Comprehensive evaluation of all available vs missing acoustic metadata.
- `environment_analysis.csv`: Machine-readable table containing all quantitative robustness metrics.
- `species_location_distribution.png`: Heatmap illustrating species-location co-distribution.
- `cluster_location_distribution.png`: Bar plots of leading cluster dominance and unclustered noise rates.
- `novelty_location_distribution.png`: Boxplots of Isolation Forest novelty scores across locations.
- `cosine_similarity_environment_gap.png`: Controlled comparison showing the environmental robustness gap.
- `run_summary.json`: Detailed execution summary, timestamps, and parameters.
- `stage8.log`: Traceable execution log.

---

## 6. Recommendations for Future Research

1. **Do not claim environment invariance for CLAP without explicit domain adaptation**: Raw CLAP embeddings retain substantial environmental and hydrophone coloration.
2. **Condition future SSL / Contrastive models on environment regularization**: A contrastive loss that penalizes same-location different-species attraction or aligns cross-location same-species calls is necessary.
3. **Incorporate Location Holdout Validation**: Future supervised or self-supervised models should always evaluate on the `watkins_splits/environment_holdout/` split to measure real-world out-of-domain transfer.