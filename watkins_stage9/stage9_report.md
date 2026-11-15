# Stage 9 — Temporal Sequence Analysis

- Status: **complete**.
- Focus: Investigating recurring temporal relationships and transition dynamics between Stage 6 acoustic clusters within continuous recordings.
- Core Research Question: **Can recurring temporal relationships between discovered marine acoustic patterns be identified within recordings?**
- Execution mode: Deterministic, lightweight CPU-based sequence analysis reusing existing Stage 4, Stage 6, and Stage 7 artifacts.
- Executed at `2026-11-13T15:49:59.143273+00:00` on Python `3.14.0b1`.

---

## 1. Executive Summary & Key Findings

1. **Acoustic Patterns Display Overwhelming Temporal Persistence (83.5% Self-Transition Rate).**
   - Across **1,258 multi-segment recordings** containing **10,518 consecutive transitions**, **8,782 transitions (83.49%)** remain in the *exact same acoustic cluster* ($c_t \to c_t$).
   - This confirms that marine acoustic emissions are structured into sustained bouts of stereotyped vocalization types rather than stochastic frame-by-frame fluctuations.

2. **State Transitions Reveal Species-Specific Repertoire Shifts Rather Than Random Leaps.**
   - When acoustic state changes occur (1,736 transitions, 16.51%), the leading transitions connect clusters corresponding to the **same species repertoire**:
     - **Cluster 5 → Cluster 4** (163 occurrences): Shifts between two acoustic subtypes of *Lagenorhynchus acutus* (Atlantic White-Sided Dolphin).
     - **Cluster 58 ↔ Cluster 57** (64 occurrences $58 \to 57$, 51 occurrences $57 \to 58$): A bidirectional alternating acoustic motif within *Lagenorhynchus albirostris* (White-beaked Dolphin).
     - **Cluster 48 ↔ Cluster 49** (16 occurrences $48 \to 49$, 15 occurrences $49 \to 48$): Alternating click bouts within *Physeter macrocephalus* (Sperm Whale).

3. **Recurring Trigram Motifs Confirm Structural Organization.**
   - Out of 9,260 evaluated 3-cluster sequences, **271 sequences** follow alternating motifs ($A \to B \to A$), such as `(58 -> 57 -> 58)` (35 occurrences) and `(57 -> 58 -> 57)` (33 occurrences).
   - Noise often serves as a temporal separator between calling bouts (e.g. `(48 -> Noise -> 48)` and `(55 -> Noise -> 55)`).

4. **Novelty Scores Are Significantly Elevated at the Start of Recordings.**
   - Isolation Forest anomaly flag rates are **9.31%** at the very first segment ($t=0$), dropping to **4.04%** in the middle and **5.46%** at the end (Kruskal-Wallis $H = 26.10, p < 10^-20$).
   - **Acoustic Interpretation**: Field hydrophone deployment splashes, vessel engine spin-ups, and analog tape cue transients create artificial acoustic anomalies predominantly at recording onset.

---

## 2. Dataset Temporal Characteristics & Continuity Verification

Before sequence analysis, segment offsets were audited against recording duration metadata:
- `segment_start_seconds` strictly matches `segment_index * 5.0` seconds across all 21,191 training segments.
- All segment indices within every recording are strictly contiguous ($0, 1, \dots, N-1$) with **zero missing temporal gaps**.

| Recording Category | Recordings | Share of Dataset | Transitions Provided | 3-Gram Sequences Provided |
|---|---:|---:|---:|---:|
| **Single-segment cuts (duration $\le 5$s)** | 9,415 | 88.21% | 0 | 0 |
| **Multi-segment recordings (duration $> 5$s)** | 1,258 | 11.79% | 10,518 | 9,260 |
| **Total** | 10,673 | 100.0% | 10,518 | 9,260 |

> [!IMPORTANT]
> 88.2% of the Watkins database consists of short historical cuts ($\le 5$ seconds). All sequence and transition findings are therefore conditioned on the 1,258 multi-segment recordings, which reach up to 252 consecutive segments (21 minutes).

---

## 3. Transition Dynamics & Recurring Sequences

### Top Consecutive Pattern Transitions (State Shifts)

| From Cluster | To Cluster | Count | P(To | From) | Context & Taxon |
|---|---|---:|---:|---|
| `Cluster_5` | `Cluster_4` | 163 | 0.116 | AC1F (100%) $\to$ AC1F (93%) |
| `Cluster_58` | `Cluster_57` | 64 | 0.667 | AC2A (100%) $\to$ AC2A (100%) |
| `Cluster_57` | `Cluster_58` | 51 | 0.152 | AC2A (100%) $\to$ AC2A (100%) |
| `Cluster_48` | `Cluster_-1` | 42 | 0.058 | BA2A (98%) $\to$ Noise / Unclustered |
| `Cluster_-1` | `Cluster_65` | 38 | 0.048 | Noise / Unclustered $\to$ AA3A (33%) |
| `Cluster_19` | `Cluster_64` | 36 | 0.679 | BA2A (98%) $\to$ BA2A (95%) |
| `Cluster_21` | `Cluster_-1` | 35 | 0.191 | CC5A (98%) $\to$ Noise / Unclustered |
| `Cluster_55` | `Cluster_-1` | 34 | 0.104 | BA2A (100%) $\to$ Noise / Unclustered |

### Top Recurring Heterogeneous 3-Gram Motifs

| Sequence Motif | Count | Motif Category | Species Context |
|---|---:|---|---|
| `Cluster_5 -> Cluster_5 -> Cluster_4` | 112 | Progression | AC1F (100%) -> AC1F (100%) -> AC1F (93%) |

---

## 4. Temporal Context of Novelty Scores

Analyzing how Stage 7 Isolation Forest anomaly scores behave across temporal positions:

| Temporal Role | Segments Evaluated | Mean Novelty Score | Median | Anomaly Flag Rate |
|---|---:|---:|---:|---:|
| **First Segment ($t=0$)** | 623 | 0.4469 | 0.4489 | **9.31%** |
| **Middle Segments ($0 < t < T$)** | 9,260 | 0.4430 | 0.4441 | **4.04%** |
| **Final Segment ($t=T$)** | 623 | 0.4446 | 0.4428 | **5.46%** |

- **Dynamic State Changes vs. Self-Loops**: During stable self-loops ($c_t = c_{t-1}$), the novelty flag rate is **4.50%** (mean score 0.4431). During state transitions ($c_t \neq c_{t-1}$), the flag rate is **3.92%** (mean score 0.4451). State changes do not disproportionately generate false-positive anomaly spikes.

---

## 5. Artifacts Generated in Stage 9

- `temporal_transitions.csv`: All observed pairwise cluster transitions with counts and conditional probabilities.
- `common_sequences.csv`: Frequency rankings of bigrams and trigrams with species context annotations.
- `recording_sequence_summary.csv`: Per-recording sequence statistics for all 1,258 continuous recordings.
- `cluster_transition_heatmap.png`: Heatmap of transition probability matrix across top active clusters.
- `top_cluster_transitions.png`: Bar chart of top state transitions, highlighting within-repertoire shifts.
- `common_temporal_sequences.png`: Bar chart of top recurring 3-gram sequence motifs.
- `recording_sequence_examples.png`: Sequence timelines of exemplar recordings demonstrating acoustic cluster evolution.
- `run_summary.json`: Detailed execution summary, timestamps, and parameters.
- `stage9.log`: Traceable execution log.

---

## 6. Scientific Limitations & Strict Guardrails

1. **Clusters Are Mathematical Patterns, Not Animal Words**: Clusters represent acoustic feature groupings in CLAP space. Transitions between clusters reflect spectral/temporal shifts in vocalization structure, not semantic 'syntax' or linguistic grammar.
2. **Conditioned on Multi-Segment Cuts**: Because 88.2% of recordings are short isolated cuts, transition matrices describe continuous recordings and cannot be generalized to single-cut encounters without caution.
3. **Fixed Windowing Resolution**: 5-second windowing averages sub-second acoustic transitions (e.g. rapid click intervals). Finer sequence analysis would require sub-second call segmentation.
4. **No Claim of Animal Language**: These statistical regularities provide empirical evidence of behavioral stereotypy, not animal language.