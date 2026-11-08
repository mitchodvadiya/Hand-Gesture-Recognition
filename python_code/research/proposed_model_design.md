# Proposed Model Design: AttnInceptionHGR

**Working Model Name:** `AttnInceptionHGR` (Attention-Enhanced Inception Hand Gesture Recognition Model)  
**Task:** Continuous participant-independent sEMG hand gesture recognition from 8-channel spectrogram sequences.

---

## 1. Proposed Research Contribution

We propose `AttnInceptionHGR`, a lightweight, channel-recalibrated spatial-temporal architecture designed specifically to address the two primary failure modes of the reference paper (Barona López et al., 2024):
1. **Electrode Crosstalk Resolution via Channel Attention:** Dynamically weighting the 8 physical electrode channels before feature extraction to resolve the chronic $66\%$ confusion between adjacent dorsal extensor muscles (`open` vs. `waveOut`).
2. **Parameter-Efficient Feature Projection:** Replacing the parameter-heavy LSTM ($\approx 7.1\text{M}$ learnables) with a compact dense projection block (LayerNorm, ReLU, Dropout 0.3) that retains sub-$5\text{ ms}$ CPU inference latency while capturing discriminative gesture features.
3. **Class-Imbalance-Aware Focal Optimization:** Training with Class-Weighted Focal Loss ($\gamma = 2.0$) to counteract the $68.4\%$ background rest frame bias without gradient saturation.

---

## 2. High-Level Architecture Diagram

```
Input Spectrogram Sequence: (Batch, TimeSteps=24, Channels=8, FreqBins=13, TimeBins=24)
                           │
                           ▼
          Reshape: (Batch * TimeSteps, 8, 13, 24)
                           │
                           ▼
      ┌──────────────────────────────────────────────┐
      │     Channel Attention Module (SE-Style)      │
      │  Adaptive AvgPool2d(1) -> (B*T, 8, 1, 1)     │
      │  Conv2d(8 -> 4, 1x1) + ReLU                  │
      │  Conv2d(4 -> 8, 1x1) + Sigmoid               │
      │  Channel Weights * Input: (B*T, 8, 13, 24)   │
      └──────────────────────────────────────────────┘
                           │
                           ▼
      ┌──────────────────────────────────────────────┐
      │   6-Module Multi-Scale Inception Backbone    │
      │   Inc_1a -> Inc_1b -> Inc_1c (+ 1a skip)     │
      │   -> Inc_1d -> Inc_1e (+ 1c skip)            │
      │   -> Inc_1f -> BatchNorm2d(72)               │
      │   Output Feature Map: (B*T, 72, 13, 24)      │
      └──────────────────────────────────────────────┘
                           │
                           ▼
      ┌──────────────────────────────────────────────┐
      │  Dense Temporal Feature Projection Head      │
      │  Flatten -> 22,464 dimensions                │
      │  Linear(22,464 -> 256)                       │
      │  LayerNorm(256) + ReLU + Dropout(0.30)       │
      │  Linear(256 -> 6 Classes)                    │
      └──────────────────────────────────────────────┘
                           │
                           ▼
      Reshape Output Logits: (Batch, TimeSteps=24, NumClasses=6)
```

---

## 3. Layer-by-Layer Specifications and Dimensions

| Stage / Layer | Component Type | Input Tensor Shape | Output Tensor Shape | Activation / Normalization | Parameter Count |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Input** | Raw Spectrogram Batch | $(B, T, 8, 13, 24)$ | $(B \cdot T, 8, 13, 24)$ | None | $0$ |
| **Channel Attention (Squeeze)** | Global Average Pooling | $(B \cdot T, 8, 13, 24)$ | $(B \cdot T, 8, 1, 1)$ | None | $0$ |
| **Channel Attention (Excitation 1)** | Conv2d ($1\times 1, 8 \to 4$) | $(B \cdot T, 8, 1, 1)$ | $(B \cdot T, 4, 1, 1)$ | ReLU | $36$ |
| **Channel Attention (Excitation 2)** | Conv2d ($1\times 1, 4 \to 8$) | $(B \cdot T, 4, 1, 1)$ | $(B \cdot T, 8, 1, 1)$ | Sigmoid | $40$ |
| **Channel Scaling** | Element-wise Channel Multiply | $(B \cdot T, 8, 13, 24)$ | $(B \cdot T, 8, 13, 24)$ | None | $0$ |
| **Inception Module 1a** | 4-branch Inception Block | $(B \cdot T, 8, 13, 24)$ | $(B \cdot T, 72, 13, 24)$ | ReLU | $6,192$ |
| **Inception Module 1b** | 4-branch Inception Block | $(B \cdot T, 72, 13, 24)$ | $(B \cdot T, 72, 13, 24)$ | ReLU | $24,192$ |
| **Inception Module 1c + Skip** | 4-branch Inception + Residual | $(B \cdot T, 72, 13, 24)$ | $(B \cdot T, 72, 13, 24)$ | ReLU | $24,192$ |
| **Inception Module 1d** | 4-branch Inception Block | $(B \cdot T, 72, 13, 24)$ | $(B \cdot T, 72, 13, 24)$ | ReLU | $24,192$ |
| **Inception Module 1e + Skip** | 4-branch Inception + Residual | $(B \cdot T, 72, 13, 24)$ | $(B \cdot T, 72, 13, 24)$ | ReLU | $24,192$ |
| **Inception Module 1f** | 4-branch Inception + BatchNorm | $(B \cdot T, 72, 13, 24)$ | $(B \cdot T, 72, 13, 24)$ | BatchNorm2d | $24,336$ |
| **Projection Layer 1** | Linear ($22,464 \to 256$) | $(B \cdot T, 22,464)$ | $(B \cdot T, 256)$ | LayerNorm + ReLU | $5,751,040$ |
| **Regularization** | Dropout ($p = 0.30$) | $(B \cdot T, 256)$ | $(B \cdot T, 256)$ | None | $0$ |
| **Classification Layer** | Linear ($256 \to 6$) | $(B \cdot T, 256)$ | $(B \cdot T, 6)$ | None (Logits) | $1,542$ |
| **Total Model** | `AttnInceptionHGR` | $(B, T, 8, 13, 24)$ | $(B, T, 6)$ | Softmax / Argmax | **$\approx 5.85\text{M}$** |

> **Parameter Comparison:**
> - `BaseInceptionCNN`: $\approx 135,000$ parameters (lightweight, but severely susceptible to extensor crosstalk).
> - `BaseInceptionCNNLSTM`: $\approx 7,100,000$ parameters ($53\times$ increase, unconstrained recurrent memory).
> - `AttnInceptionHGR`: $\approx 5,850,000$ parameters ($18\%$ smaller than CNN-LSTM, with dedicated channel gating).

---

## 4. Why Each Component is Justified

1. **Channel Attention (SE-Style):**
   - *Problem Addressed:* Electrodes covering EDC (`open`) and ECRL (`waveOut`) sit immediately adjacent on the dorsal forearm. Standard uniform 2D convolutions cannot dynamically suppress adjacent muscle crosstalk.
   - *Mechanism:* Squeezes spatial spectrogram maps into channel-wise descriptors via global average pooling and applies a non-linear bottleneck to generate an 8-element scaling vector $\mathbf{w} \in (0, 1)^8$. Electrodes with high active signal-to-noise ratio are boosted, while irrelevant or noisy channels are dampened.
2. **Dense Bottleneck with LayerNorm & Dropout:**
   - *Problem Addressed:* The raw Inception CNN directly connects 22,464 flattened features to 6 output units, leading to severe overfitting on the training cohort and poor transfer to unseen participants.
   - *Mechanism:* Compresses the 22,464 features into a 256-dimensional compact manifold stabilized by Layer Normalization and regularized with $30\%$ Dropout, preventing co-adaptation of features.
3. **Class-Weighted Focal Loss:**
   - *Problem Addressed:* $68.4\%$ background rest frames saturate Cross-Entropy gradients.
   - *Mechanism:* Modulates the cross-entropy loss by $(1 - p_t)^\gamma$ with $\gamma = 2.0$ and inverse-frequency class weights $\alpha_c$, focusing backpropagation on ambiguous gesture frames.

---

## 5. Planned Ablation Strategy

To rigorously evaluate the isolated contribution of each module, the following controlled ablations are defined:
1. **Ablation 1 (Impact of Channel Attention):** Compare `AttnInceptionHGR` vs. `AttnInceptionHGR` without Channel Attention (keeping the dense projection and Focal Loss identical).
2. **Ablation 2 (Impact of Focal Loss):** Compare `AttnInceptionHGR` trained with Focal Loss vs. standard Cross-Entropy (keeping architecture identical).
3. **Ablation 3 (Impact of Data Augmentation):** Compare `AttnInceptionHGR` trained with vs. without topology-preserving amplitude scaling and sensor noise.
4. **Ablation 4 (Architectural Comparison):** Compare `AttnInceptionHGR` against `BaseInceptionCNN` and `BaseInceptionCNNLSTM` on identical participant splits, training budgets, and seeds.
