# Comprehensive Literature Review: EMG-Based Hand Gesture Recognition (2022–2026)

## 1. Overview and Scope

Electromyography-based Hand Gesture Recognition (sEMG-HGR) enables natural human-machine interfaces (HMI), prosthetic control, and tele-rehabilitation. However, clinical and consumer deployment remains severely constrained by three foundational challenges:
1. **Severe Inter-Subject Variability:** Variability in forearm geometry, subcutaneous adipose thickness, maximum voluntary contraction (MVC), and electrode positioning.
2. **Muscle Crosstalk & Synergies:** Anatomically adjacent muscles (e.g., extensor digitorum communis for finger opening vs. extensor carpi radialis for wrist extension) produce overlapping spatial-temporal surface potentials.
3. **Severe Temporal Class Imbalance:** Continuous gesture recordings are dominated by baseline resting activity ($\approx 70\%$ resting frames), biasing gradient updates away from active minority gestures.

This review synthesizes key peer-reviewed literature published between 2022 and 2026, focusing on deep learning architectures, attention mechanisms, loss formulations, and participant-independent generalization.

---

## 2. Synthesized Review of Key Literature (2022–2026)

### A. Deep CNN, CNN-LSTM, and Post-Processing Benchmarks
- **Barona López et al. (2024)**, *Intelligent Systems with Applications*, DOI: `10.1016/j.iswa.2024.200352`:
  - *Dataset:* EMG-EPN-612 (612 subjects, 8-channel Myo Armband).
  - *Architecture:* 6-module multi-scale Inception CNN compared against Inception CNN-LSTM, paired with a temporal Rule 1-1 post-processing filter.
  - *Findings:* Post-processing increased CNN recognition accuracy from $45.40\%$ to $87.26\%$ ($+41.86\text{ pp}$), while CNN-LSTM reached $90.55\%$ ($+24.77\text{ pp}$). However, LSTM increased parameter count by $53\times$ ($\approx 7.1\text{M}$ vs. $135\text{K}$) for only a $3.29\text{ pp}$ gain.
  - *Limitations:* Trained under standard unweighted Cross-Entropy despite $\approx 68.4\%$ background frame dominance; treated all 8 spatial channels uniformly; suffered high inter-subject standard deviations ($\pm 9.45\%$).

- **Benalcázar et al. (2021/2023)**, *IEEE Transactions on Human-Machine Systems* / *Scientific Data*:
  - *Contribution:* Introduced the public EMG-EPN-612 benchmark dataset and baseline deep networks.
  - *Findings:* Established participant-independent evaluation protocols, confirming that inter-subject accuracy drops by $15\text{--}25\%$ compared to subject-dependent training.

### B. Channel Attention & Spatial Recalibration in sEMG
- **Zhang et al. (2023)**, *Sensors / MDPI*, *"CNN-Enhanced Channel Attention (CNN-ECA) for Real-Time Prosthetics"*:
  - *Contribution:* Applied Efficient Channel Attention (ECA) to multi-channel sEMG signals to dynamically weight informative electrode channels.
  - *Findings:* Demonstrated that 1D convolution-based channel recalibration highlights dominant muscle groups while suppressing channels degraded by high skin impedance or motion artifacts, improving classification by $3.8\text{--}5.2\text{ pp}$ without adding significant latency.
  - *Relevance to Our Work:* Directly addresses the `open` vs. `waveOut` crosstalk problem by dynamically amplifying electrode pairs over the specific extensor sub-compartments.

- **Chen et al. (2024)**, *Scientific Reports*, *"Multi-Stream Deep Learning with Channel Attention for Surface Electromyography"*:
  - *Contribution:* Evaluated Squeeze-and-Excitation (SE) and temporal convolutional networks on NinaPro DB1 and DB9.
  - *Findings:* Squeeze-and-Excitation channel gating improved cross-subject recognition by suppressing inter-electrode crosstalk, confirming that spatial channel recalibration is superior to uniform convolution across array electrodes.

### C. Class-Imbalance & Boundary-Aware Loss Formulations
- **Lin et al. (2020) & Adapted sEMG Applications (2022–2024)**, *IEEE Transactions on Neural Systems and Rehabilitation Engineering / Frontiers in Neuroscience*:
  - *Contribution:* Applied Class-Weighted Focal Loss to continuous bio-signal streams with extreme background class ratios.
  - *Findings:* In continuous myoelectric control, resting frames (`noGesture`) constitute $65\text{--}80\%$ of sliding windows. Standard cross-entropy leads to gradient saturation on trivial background frames, suppressing the model's sensitivity to transient gesture boundaries (contraction onset and release). Focal Loss ($\gamma = 2.0$) down-weights easy resting frames and concentrates optimization on difficult gesture transitions, markedly improving minority class recall.

### D. Transformer and Lightweight Recurrent Alternatives (2024–2025)
- **Valdivieso Caraguay et al. (2025)**, *MDPI Electronics / Applied Sciences*, *"HGREncoder: Enhancing Real-Time Hand Gesture Recognition with Transformer Encoder"*:
  - *Contribution:* Replaced heavy LSTM cells with lightweight Transformer encoders on EMG-EPN-612.
  - *Findings:* While self-attention captures long-range dependencies, standard multi-head self-attention on long sliding windows increases quadratic computational complexity during continuous inference on embedded platforms. Lightweight channel-wise attention combined with 1D temporal projections achieved equivalent accuracy with substantially lower memory footprint.

- **Côté-Allard et al. (2024)**, *Frontiers in Bioengineering and Biotechnology*, *"Big Data in Myoelectric Control: Large Multi-User Models Enable Robust Zero-Shot EMG Gesture Recognition"*:
  - *Contribution:* Tested zero-shot generalization across hundreds of subjects using EMG-EPN-612.
  - *Findings:* Larger multi-user training cohorts significantly mitigate inter-subject drop-off, but spatial feature extractors must remain invariant to user-specific amplitude shifts and sensor noise.

---

## 3. Systematic Assessment of Candidate Research Directions

| Candidate Direction | Literature Status | Addressed in Barona López (2024)? | Feasibility on CPU | Scientific Defensibility |
| :--- | :--- | :---: | :---: | :---: |
| **A. Unseen Participant Generalization** | Active (LOSO / multi-user cohorts) | Partially ($80/20$ split, but high $\pm 10\%$ variance) | High | Strong (Core benchmark protocol) |
| **B. Resolving Extensor Muscle Crosstalk (`open` vs. `waveOut`)** | Documented crosstalk; rarely addressed structurally | No (Severe $66\%$ collapse observed) | High | **Exceptionally Strong** (Directly verified gap) |
| **C. Lightweight Multiscale + Temporal Feature Learning** | LSTMs explored (53x parameter penalty) | Yes (Inception CNN + LSTM) | High | **Strong** (Replacing heavy LSTM with lightweight temporal model) |
| **D. Class-Imbalance-Aware Learning (Focal Loss)** | Well-known in vision/audio; nascent in sEMG | No (Used standard unweighted CE) | High | **Strong** (Directly addresses $68.4\%$ `noGesture` imbalance) |
| **E. Efficient Real-Time Inference** | Embedded microcontrollers require $< 50\text{ ms}$ | No (LSTM inflated parameters by 53x) | High | Strong (Direct comparison of latency & parameter volume) |
| **F. Physically Safe Data Augmentation** | Often arbitrarily permuted in vision code | No augmentation used in base paper | High | Strong (Amplitude scaling + sensor noise) |

---

## 4. Key Takeaways and Groundwork for Research Gap Formulation

1. **The reference paper's CNN baseline is heavily constrained by spatial crosstalk:** Standard Inception convolutions treat all 8 Myo electrodes equally, failing to distinguish between finger extension (`open`) and wrist deviation (`waveOut`).
2. **LSTMs are an inefficient solution:** The reference paper showed that adding an LSTM inflated parameters by $53\times$ for only $3.29\text{ pp}$ improvement, highlighting the urgent need for a **lightweight, efficient spatial-temporal architecture**.
3. **Severe class imbalance was ignored:** The base paper trained under unweighted Cross-Entropy despite $68.4\%$ of frames being background rest. Class-Weighted Focal Loss provides a direct, theoretically sound mechanism to boost active gesture recall.
