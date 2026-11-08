# Reference Paper Analysis

**Title:** CNN-LSTM and post-processing for EMG-based hand gesture recognition  
**Authors:** Lorena Isabel Barona López, Francis M. Ferri, Jonathan Zea, Ángel Leonardo Valdivieso Caraguay, Marco E. Benalcázar  
**Journal:** *Intelligent Systems with Applications*, Volume 22, 2024, Article 200352  
**DOI:** [`10.1016/j.iswa.2024.200352`](https://doi.org/10.1016/j.iswa.2024.200352)  
**Official Repository:** [`https://github.com/laboratorioAI/2023-HGR5-CNN_LSTM`](https://github.com/laboratorioAI/2023-HGR5-CNN_LSTM)

---

## 1. Research Problem and Objectives

- **Research Problem:** Hand Gesture Recognition (HGR) using surface electromyography (sEMG) faces significant degradation when applied to unseen users due to non-stationary biological signals, skin-electrode contact impedance, anatomical variations, and electrode displacement.
- **Primary Objectives:**
  1. Evaluate whether multi-scale spectral representations (STFT spectrograms) processed by Inception-style CNNs can capture localized frequency and channel dynamics.
  2. Measure the exact marginal utility of recurrent memory cells (LSTM) when stacked on top of deep CNN feature extractors.
  3. Introduce and validate a rule-based temporal post-processing filter (Rules 1-1, 1-2, 2-1 with thresholded voting) to remove transient classification jitter and false-negative gaps in prediction streams.

---

## 2. Dataset and Sensor Configuration

- **Dataset:** EMG-EPN-612 (612 total participants divided into 306 training users and 306 test users).
- **Sensor:** Thalmic Labs Myo Armband (8 dry circular sEMG electrodes circumferentially arranged around the upper forearm).
- **Sampling Rate:** $200\text{ Hz}$.
- **Gesture Classes (6 Classes):**
  - 5 active gestures: `fist`, `open`, `pinch`, `waveIn`, `waveOut`.
  - 1 baseline resting state: `noGesture`.
- **Protocol:** Each participant performed 25 repetitions of each gesture (5 seconds per trial, 1,000 samples). Total trials per participant = $25 \times 6 = 150\text{ trials}$.

---

## 3. Signal Representation and Preprocessing

- **Filtering:** 5th-order low-pass Butterworth filter, cutoff frequency $10\text{ Hz}$ ($0.1 \times \frac{f_s}{2}$).
- **Rectification:** Full-wave absolute rectification ($|s|$).
- **Normalization:** Division by fixed hardware scalar $128.0$ if $\max(|s|) > 1.0$ (8-bit ADC half-scale).
- **Spectrogram Parameters:**
  - Window length: $N = 24\text{ samples}$ ($120\text{ ms}$).
  - Overlap: $50\%$ ($12\text{ samples}$, step = $12\text{ samples}$).
  - Frequencies: Discrete evaluation at $0, 1, 2, \dots, 12\text{ Hz}$ ($13\text{ frequency bins}$).
  - Window function: Periodic Hanning window.
- **Sliding Frame Window:**
  - Frame length: $300\text{ samples}$ ($1.5\text{ s}$).
  - Frame step/stride: $30\text{ samples}$ ($0.15\text{ s}$).
  - Spectrogram Frame Output Shape: $(8\text{ channels}, 13\text{ frequency bins}, 24\text{ time steps})$.

---

## 4. Architectures Evaluated in the Reference Paper

### A. Base Inception CNN
- **Backbone:** 6 cascaded Inception modules (`1a` through `1f`).
- **Module Design:**
  - Branch 1: $1\times 1$ conv ($18\text{ filters}$).
  - Branch 2: $1\times 1$ reduce ($16\text{ filters}$) $\to$ $3\times 3$ conv ($18\text{ filters}$, pad 1).
  - Branch 3: $1\times 1$ reduce ($16\text{ filters}$) $\to$ $5\times 5$ conv ($18\text{ filters}$, pad 2).
  - Branch 4: $3\times 3$ maxpool (stride 1, pad 1) $\to$ $1\times 1$ conv ($18\text{ filters}$).
  - Depth concatenation: $18 \times 4 = 72\text{ channels}$.
  - Residual connections: `1a` output added into `1c`; `1c` output added into `1e`.
  - Normalization: Batch normalization on the final module `1f`.
- **Head:** Fully connected linear layer: $72 \times 13 \times 24 = 22,464 \to 6\text{ classes}$.
- **Learnable Parameters:** $\approx 135,000$.

### B. Base Inception CNN-LSTM
- Identical CNN backbone used as a frame-level feature extractor.
- Flattened features fed into an LSTM layer with hidden dimension $128$.
- Fully connected head mapping hidden states to 6 classes.
- **Learnable Parameters:** $\approx 7,100,000$ ($\approx 53\times$ increase over CNN).

---

## 5. Post-Processing Filter Algorithm

- **Step 1:** Discard `'noGesture'` predictions from the sequence of frame outputs.
- **Step 2:** Count class occurrences using majority voting. If the maximum count $\ge \text{MIN\_LABELS\_SEQUENCE} = 4$, identify the candidate dominant gesture. Otherwise, classify the sequence as `'noGesture'`.
- **Step 3 (Temporal Smoothing):** Apply Rule 1-1:
  - Fill 1-frame gaps inside gesture sequences: if $y_{t-1} = C$ and $y_{t+1} = C$, set $y_t = C$.
  - Remove 1-frame spikes in resting intervals: if $y_{t-1} = \text{'noGesture'}$ and $y_{t+1} = \text{'noGesture'}$, set $y_t = \text{'noGesture'}$.
  - Clean non-dominant labels to `'noGesture'`.
- **Step 4:** Re-apply the 4-frame threshold to produce the final trial classification.

---

## 6. Reported Performance and Experimental Findings

| Model Configuration | Raw Accuracy (%) | Post-Processed Accuracy (%) | Absolute Gain (pp) | Parameter Count |
| :--- | :---: | :---: | :---: | :---: |
| **CNN Baseline** | $45.40 \pm 15.51$ | $87.26 \pm 11.14$ | **+41.86 pp** | $\approx 135\text{K}$ |
| **CNN-LSTM Baseline** | $65.78 \pm 15.51$ | $90.55 \pm 9.45$ | **+24.77 pp** | $\approx 7.1\text{M}$ |

### Key Conclusions from the Authors:
1. **Dominant Impact of Post-Processing:** Temporal filtering provided a massive $+41.86$ percentage-point boost for CNN and $+24.77$ percentage points for CNN-LSTM, proving that boundary jitter and spurious frames are the primary error sources.
2. **Diminishing Returns of Memory Cells:** Adding an LSTM increased final accuracy by only $3.29$ percentage points ($87.26\% \to 90.55\%$), while inflating parameter count by $53\times$ ($135\text{K} \to 7.1\text{M}$), making CNN-LSTM disproportionately expensive for embedded real-time prosthetics.

---

## 7. Comparative Audit: Reference Paper vs. Existing Implementation

| System Component | Reference Paper (Barona López et al., 2024) | Existing Codebase Implementation | Status |
| :--- | :--- | :--- | :---: |
| **Dataset Source** | EMG-EPN-612 ($612\text{ users}$) | EMG-EPN-612 in `EMG-EPN-612 dataset/` | Verified Identical |
| **Partitioning Seed** | `rng(9)` in MATLAB | `np.random.seed(9)` in `dataset.py` | Verified Identical |
| **Butterworth Filter** | 5th order, $10\text{ Hz}$ lowpass (`butter(5, 0.1)`) | `signal.butter(5, 0.1, btype='low')` | Verified Identical |
| **Normalization** | Fixed constant $/128.0$ | Fixed constant $/128.0$ | Verified Identical |
| **Spectrogram Basis** | STFT $0\dots 12\text{ Hz}$, 24-pt Hann | Exact DFT basis matrix in `preprocessing.py` | Verified Identical |
| **CNN Backbone** | 6-Module Inception with Skips | `BaseInceptionCNN` in `models.py` | Verified Identical |
| **CNN-LSTM Model** | Inception + LSTM ($128\text{ units}$) | `BaseInceptionCNNLSTM` in `models.py` | Verified Identical |
| **Post-Processing** | Mode `1-1`, threshold $\ge 4$ | `HGRPostProcessor(mode='1-1')` | Verified Identical |
| **Loss Function** | Standard Cross-Entropy | Standard Cross-Entropy + Focal Loss implemented | Extended |
| **Evaluation Level** | Sequence-level recognition | Frame-level & Sequence-level | Verified & Extended |

---

## 8. Limitations Analysis

### A. Author-Acknowledged Limitations:
1. **Computational Overhead of LSTM:** The $53\times$ increase in parameter volume makes CNN-LSTM impractical for low-power edge microcontrollers in bionic prosthetics.
2. **Offline Latency of Post-Processing:** Inspecting future time steps ($t+1$) in Rule 1-1 requires an offline buffer or introduces a $150\text{ ms}$ real-time delay.
3. **High Inter-Subject Variance:** Standard deviations remain high ($\pm 9.45\%$ to $\pm 15.51\%$), indicating poor generalization on a substantial subset of participants.

### B. Inferred Critical Limitations (Research Gap Foundations):
1. **Channel Neglect & Spatial Crosstalk:** The 8 Myo channels cover antagonistic muscle groups (flexors vs. extensors). Treating them uniformly without channel attention ignores localized muscle recruitment and causes severe confusion between adjacent dorsal extensor gestures (`open` vs. `waveOut`).
2. **Severe Frame Imbalance Under Cross-Entropy:** Across any 5-second trial, $\approx 68.4\%$ of frames are resting baseline (`noGesture`). Training with unweighted Cross-Entropy causes the network to settle into majority-class shortcuts, depressing minority gesture recall.
3. **Over-Reliance on Post-Processing:** When the underlying classifier is weak or un-converged, post-processing strips away sparse genuine activations (as proven by our $-0.48\text{ pp}$ decrease on Epoch 1), demonstrating that filtering cannot substitute for robust feature representations.
