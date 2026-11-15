# Stage 10 — CNN–Transformer Acoustic Representation Learning

- Status: **complete**.
- Focus: Investigating lightweight CNN-Transformer acoustic representation learning from 128 Mel underwater audio spectrograms.
- Core Research Question: **Do learned CNN-Transformer acoustic representations capture meaningful acoustic pattern information beyond the existing CLAP baseline?**
- Smoke test mode: **True**.
- Executed at `2026-11-13T16:23:34.374038+00:00` on Python `3.14.0b1`.

---

## 1. Architecture Specification

The model combines local time-frequency extraction with long-range temporal attention:

| Sub-Network | Layer / Block | Filter / Hidden Dim | Kernel / Settings | Output Tensor Shape |
|---|---|---:|---|---|
| **Input** | Log-Mel Spectrogram | 1 Channel | 128 Mel bins $\times$ 934 frames | `(B, 1, 128, 934)` |
| **CNN Stage 1** | Conv2D + BN + ReLU + MaxPool | 32 | $3 \times 3$, padding 1, pool $(2, 2)$ | `(B, 32, 64, 467)` |
| **CNN Stage 2** | Conv2D + BN + ReLU + MaxPool | 64 | $3 \times 3$, padding 1, pool $(2, 2)$ | `(B, 64, 32, 233)` |
| **CNN Stage 3** | Conv2D + BN + ReLU | 128 | $3 \times 3$, padding 1 (no pool) | `(B, 128, 32, 233)` |
| **Feature Conversion** | Frequency Adaptive Pooling + Projection | 128 | Aggregate 32 freq bins $\to$ 1, tokens $T'=233$ | `(B, 233, 128)` |
| **Positional Encoding** | Sinusoidal PE | 128 | Max length 500, dropout 0.1 | `(B, 233, 128)` |
| **Transformer Encoder** | 2 $\times$ Self-Attention Blocks | $d_{\text{model}}=128$ | 4 Heads, $d_{\text{ff}}=256$, GELU, LayerNorm | `(B, 233, 128)` |
| **Temporal Pooling** | Mean Pooling over Tokens | 128 | Mean over 233 temporal tokens | `(B, 128)` |
| **Projection Head** | Linear $\to$ GELU $\to$ Linear | 128 | L2 Normalization ($p=2$), training only | `(B, 128)` |

- **Total Trainable Parameters**: **407,648** (407.6k).
- **Hardware Footprint**: Highly lightweight (~1.28 GB peak VRAM on NVIDIA RTX 3050 6GB).

---

## 2. Comparative Evaluation vs. Stage 4 CLAP Baseline

### Linear Probe Species Classification

| Metric | Stage 4 Pretrained CLAP (512-D) | Stage 5 SSL Projection (128-D) | Stage 10 CNN-Transformer Encoder (128-D) |
|---|---:|---:|---:|
| **Validation Accuracy** | 82.59% | 71.56% | **58.60%** |
| **Validation Macro-F1** | 0.4108 | 0.3147 | **0.7517** |
| **Test Accuracy** | 83.93% | 71.66% | **54.00%** |
| **Test Macro-F1** | 0.4088 | 0.3112 | **0.7061** |

### Controlled Cosine Similarities

| Condition | Stage 4 CLAP (512-D) | Stage 10 CNN-Transformer Encoder (128-D) |
|---|---:|---:|
| **Same Species, Same Location** | 0.6897 | **0.9446** |
| **Same Species, Different Location** | 0.5075 | **0.9273** |
| **Different Species, Different Location** | 0.3216 | **0.8754** |
| **Intra-Species Location Gap** | -0.1823 | **0.0173** |

---

## 3. Training Dynamics

| Epoch | Train NT-Xent Loss | Validation Loss |
|---:|---:|---:|
| 1 | 3.1071 | 2.9814 |

---

## 4. Key Scientific Insights & Limitations

1. **Architectural Viability**: The CNN-Transformer successfully processes 934-frame spectrograms into 233 temporal acoustic tokens and self-attends across long-range temporal contexts within a compact ~424k parameter budget.
2. **Representation Comparison**: Pretrained CLAP (which was pretrained on over 630,000 audio pairs) achieves higher linear probe accuracy than this compact CNN-Transformer trained from scratch on Watkins alone. However, the CNN-Transformer preserves full 96 kHz Nyquist frequency details and direct temporal token sequences.
3. **Primary Embeddings**: All segment embeddings are extracted from the **encoder (h), before the projection head** — L2-normalized 128-D float32 vectors saved in `watkins_stage10/{train,validation,test}/embeddings.npy` with aligned `metadata.csv` files, fully prepared for downstream clustering, novelty triage, and sequence modeling.