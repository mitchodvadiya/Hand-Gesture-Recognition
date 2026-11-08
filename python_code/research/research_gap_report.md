# Research Gap Report: Identified Opportunities, Evidence, and Strategic Recommendations

## 1. Analysis of Candidate Research Directions

Based on our comprehensive literature review and rigorous evaluation of the baseline checkpoint (`best_model_cnn_base_crossentropy.pth`), we systematically evaluate the candidate research directions:

### Candidate A: Generalization to Unseen Participants
- **Current State:** Handled in the reference paper via an 80/20 partition (244 train / 62 val users). However, standard deviations remain very high ($\pm 9.45\%$ to $\pm 15.51\%$).
- **Evidence:** Evaluating our checkpoint on 62 unseen users dropped accuracy from $62.40\%$ to $58.00\%$, demonstrating that inter-subject variance remains a dominant barrier.
- **Evaluation:** Must serve as the foundation of our experimental protocol (strict participant-independent evaluation).

### Candidate B: Spatial Crosstalk & Muscle Confusion (`open` vs. `waveOut`)
- **Current State:** Completely neglected in the reference paper's architectural design. The reference paper treated all 8 channels as uniform input feature maps.
- **Evidence:** In our 9,300-sequence evaluation, **1,023 out of 1,550 actual `open` sequences (66%) were misclassified as `waveOut`**, yielding an abysmal `open` recall of only $5.42\%$ and F1 of $10.07\%$. Electrophysiologically, both activate the dorsal forearm (extensor digitorum vs. extensor carpi radialis).
- **Literature Solution:** Channel Attention / Squeeze-and-Excitation mechanisms (Zhang et al., 2023; Chen et al., 2024) dynamically emphasize dominant electrode channels and suppress adjacent crosstalk.

### Candidate C: Temporal Modeling vs. Parameter Efficiency
- **Current State:** Barona López et al. (2024) demonstrated that adding an LSTM inflated parameters by $53\times$ ($\approx 7.1\text{M}$ vs. $135\text{K}$) for only a $3.29\text{ pp}$ accuracy gain.
- **Evidence:** Stacking an unconstrained LSTM on a deep CNN creates severe computational overhead on CPU and edge microcontrollers without addressing spatial channel crosstalk.
- **Opportunity:** A compact temporal model or direct dense temporal projection with LayerNorm and Dropout can capture temporal gesture dynamics at a fraction of the parameter cost.

### Candidate D: Class-Imbalance-Aware Learning (Focal Loss)
- **Current State:** The reference paper used standard unweighted Cross-Entropy.
- **Evidence:** Continuous EMG streaming trials contain $\approx 68.4\%$ resting frames (`noGesture`). Cross-Entropy gradients are overwhelmed by easy background frames, preventing the network from learning sharp decision boundaries for transient active gestures.
- **Literature Solution:** Class-Weighted Focal Loss down-weights easy background frames and scales gradients toward hard minority gesture transitions.

### Candidate E: Physically Grounded Data Augmentation
- **Current State:** No data augmentation was used in the reference paper.
- **Evidence:** Arbitrary image augmentations (random cropping, rotations, or circular channel permutations) destroy the physiological muscle-electrode correspondence.
- **Opportunity:** Small amplitude scaling ($0.90\times \text{--} 1.10\times$, simulating contraction force variations) and additive Gaussian sensor noise ($1.5\%$ of channel standard deviation, simulating electrode contact impedance) preserve physical topology while improving robustness to unseen users.

---

## 2. Ranking of the Top 3 Research Directions

| Rank | Research Direction | Scientific Value | Literature Evidence | Dataset Compatibility | Feasibility on CPU | Measurable Impact |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1 (Primary)** | **Spatial Channel Attention for Extensor Crosstalk Resolution** | **Highest** | Strong (Zhang 2023, Chen 2024) | Perfect (8 Myo channels) | Immediate (< 1% param overhead) | Solves the 66% `open`/`waveOut` confusion |
| **2 (Secondary)** | **Class-Weighted Focal Loss for Acute Frame Imbalance** | **High** | Strong (Lin 2020, NIH 2023) | Perfect (68.4% noGesture) | Zero parameter or latency cost | Boosts active gesture recall & minority F1 |
| **3 (Secondary)** | **Physiologically Grounded EMG Augmentation (Topology-Preserving)** | **High** | Strong (Côté-Allard 2024) | Perfect (Spectrogram tensors) | Training-time only | Enhances unseen participant generalization |

---

## 3. Recommended Research Contribution & Working Hypothesis

### Primary Research Contribution:
> **An Attention-Enhanced, Imbalance-Aware Deep Learning Framework for Participant-Independent EMG Hand Gesture Recognition.**

### Working Hypotheses:
1. **Hypothesis 1 (Channel Attention vs. Crosstalk):** Integrating a lightweight Channel Attention module prior to multi-scale Inception convolution will dynamically modulate channel gains across the 8 electrode sites, significantly reducing the $66\%$ misclassification of `open` into `waveOut` and elevating `open` recall without parameter bloat.
2. **Hypothesis 2 (Focal Loss vs. Background Dominance):** Replacing standard Cross-Entropy with Class-Weighted Focal Loss will prevent gradient saturation from the $68.4\%$ resting frames, enhancing sensitivity to contraction boundaries and improving Macro F1.
3. **Hypothesis 3 (Parameter Efficiency vs. LSTM):** An attention-augmented spatial CNN with dense feature projection will achieve or exceed the accuracy of the heavy CNN-LSTM baseline while maintaining $\approx 96\%$ fewer parameters and sub-$5\text{ ms}$ inference latency.

---

## 4. Defensibility & Falsification Criteria

- **What would count as a meaningful contribution?**
  1. A statistically verifiable reduction in `open` $\to$ `waveOut` confusion (lifting `open` F1 from $\sim 10\%$ to $> 50\%$).
  2. An overall improvement in Macro F1 across unseen validation participants compared to the standard CNN baseline.
  3. Achieving competitive accuracy with the reference paper's CNN-LSTM while eliminating its $53\times$ parameter penalty.
- **What result would show that the proposed idea did NOT work?**
  1. If Channel Attention fails to improve `open` recall or decreases overall accuracy on held-out users.
  2. If Focal Loss causes the model to over-predict minority gestures in resting sequences, inflating false positives in `noGesture`.
  3. If inference latency or training stability degrades without measurable gains in sequence-level accuracy.
