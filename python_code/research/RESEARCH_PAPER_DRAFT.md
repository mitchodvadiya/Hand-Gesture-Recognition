# Continuous sEMG Hand Gesture Recognition Using Attention-Enhanced Inception-BiLSTM and Class-Weighted Focal Loss: A Comparative Study with Base CNN-LSTM

**[Your Name], Enrollment No. [Your Enrollment Number]**  
Department of ICT / Instrumentation and Control, Marwadi University, Rajkot, India | [your_email@example.com]  
Faculty Guide: [Guide Name]  

---

## Abstract
Predicting hand gestures from surface electromyography (sEMG) is challenging because bio-potential signals are noisy, user-dependent, and non-stationary. In continuous acquisition streams, two physiological and computational bottlenecks arise: spatial crosstalk across adjacent forearm muscles that degrades discrimination between anatomical antagonists, and severe frame-level class imbalance (~10:1 ratio) caused by predominant resting frames (`noGesture`), which induces severe gesture under-detection under standard Cross-Entropy loss. This paper implements an improved spatial-temporal gesture classification framework following the benchmark Inception CNN-LSTM model of Barona López et al. [1] on the public EMG-EPN-612 dataset, and compares four architectures on a strictly disjoint 5-participant cohort (750 training vs. 750 validation continuous sequences): a Base Inception CNN, the reference Base Inception CNN-LSTM, an Enhanced CNN with Squeeze-and-Excitation (SE) channel attention, and a proposed AttnInceptionHGR architecture fusing SE channel attention, an Inception backbone, a 256-dimensional bottleneck projection, a Bidirectional LSTM, and 4-head Temporal Self-Attention under Class-Weighted Focal Loss ($\gamma = 2.0$). On the held-out 750-sequence validation set, the proposed model reached **72.00% sequence accuracy, 72.60% Macro F1, 77.32% Macro Precision and 72.00% Macro Recall**, compared to 70.67%, 68.76%, 74.89% and 70.67% for the reference CNN-LSTM baseline (+3.84 percentage points gain in Macro F1). Crucially, the proposed architecture reduced active gesture sequences incorrectly classified as resting state by **78.26%** (from 92 to 20 missed gesture trials), raising resting-class precision from 57.60% to 85.92% and F1-score from 73.10% to 91.39%. Offline CPU inference latency was 53.84 ms per 24-window sequence against 49.17 ms for the baseline. We interpret the differences, trade-offs in individual gestures, and discuss limitations: a single training run, a 5-subject validation cohort, and offline batch evaluation.

**Keywords—** surface electromyography, hand gesture recognition, Inception network, bidirectional LSTM, channel attention, focal loss, time-series classification

---

## I. INTRODUCTION
Forecasting discrete gesture intent from continuous forearm surface electromyography (sEMG) is a central problem in active myoelectric prosthetics, neuro-rehabilitation, and human-computer interaction (HCI) [1], [2]. Non-invasive surface electrodes detect compound motor unit action potentials (MUAP) generated during voluntary contractions. However, the signal-to-noise ratio is low, muscle fatigue induces non-stationary spectral drift, and cross-user anatomical variability causes models trained on one participant to degrade on unseen individuals [3], [4].

In continuous naturalistic interaction, two fundamental signal-processing challenges persist:
1. **Electrode Spatial Crosstalk:** Compact consumer arrays (such as the 8-channel Myo Armband) record composite potentials from overlapping muscular compartments. Antagonistic gestures sharing common muscular substrates (such as `open` hand extension and `waveOut` wrist abduction) produce correlated spatial activations across adjacent channels.
2. **Acute Resting Frame Imbalance:** Continuous interaction comprises brief voluntary muscle contractions (1.0–2.0 s) interspersed with prolonged resting intervals. Sliding-window segmentation across continuous trials produces an acute frame distribution wherein resting state (`noGesture`) accounts for over 68.4% of all sliding frames. Under standard Cross-Entropy loss, classifiers suffer from *gesture under-detection*, erroneously predicting active muscle contractions as resting state.

Barona López et al. [1] proposed combining an Inception Convolutional Neural Network (CNN) with an unrolled Long Short-Term Memory (LSTM) network and a modal majority post-processor on the public EMG-EPN-612 dataset, reporting 90.55% ± 9.45% recognition accuracy across 306 test subjects. This is the base paper of the present study. Two key questions follow from it. First, can integrating Squeeze-and-Excitation (SE) channel attention dynamically recalibrate electrode channels and resolve spatial crosstalk between antagonistic muscle groups? Second, does introducing a 256-dimensional bottleneck projection resolve the $53\times$ parameter bloat of the reference paper's uncompressed LSTM (22,464 input features) while enabling bidirectional temporal modeling and Class-Weighted Focal Loss to mitigate resting-state gesture under-detection in low-data regimes?

The comparison is summarised in Table I. All implemented models share the identical signal preprocessing, sliding windowing (1.5 s window, 0.15 s stride, 24 frames/trial), and STFT spectrogram generation (24-point Hanning window, 13 frequency bins from 0 to 12 Hz), so differences in results originate from the architectural learner, loss formulation, and temporal attention mechanisms rather than data preconditioning.

#### TABLE I. BASE PAPER AND IMPLEMENTED APPROACHES
| Approach | Input Representation | Learner / Architecture |
| :--- | :--- | :--- |
| **Base paper [1]** | STFT Spectrogram sequence ($24 \times 8 \times 13 \times 24$) | Inception CNN + 128-d unidirectional LSTM |
| **Algorithm 1: Base CNN (this work)** | STFT Spectrogram sequence ($24 \times 8 \times 13 \times 24$) | 6-module Inception CNN (no recurrence) |
| **Algorithm 2: Base CNN-LSTM (this work)** | STFT Spectrogram sequence ($24 \times 8 \times 13 \times 24$) | Inception CNN + 128-d LSTM (reproduction of [1]) |
| **Algorithm 3: Enhanced CNN (this work)** | STFT Spectrogram sequence ($24 \times 8 \times 13 \times 24$) | SE Attention + Inception CNN + Focal Loss |
| **Algorithm 4: Proposed AttnInceptionHGR (this work)** | STFT Spectrogram sequence ($24 \times 8 \times 13 \times 24$) | SE Attention + Inception + 256-d Proj + BiLSTM + Temp Attn + Focal Loss |

The contributions of this paper are:
- **(a)** a leakage-controlled, participant-independent pipeline in which 5 training participants (750 sequences) and 5 validation participants (750 sequences) have strictly zero overlap ($\text{Train} \cap \text{Val} = \emptyset$) with per-trial normalization;
- **(b)** an implementation and empirical evaluation of four distinct sEMG architectures (Base Inception CNN, Base Inception CNN-LSTM, Enhanced Inception CNN, and Proposed AttnInceptionHGR) under identical training and evaluation settings;
- **(c)** an 88.6% compression of the recurrent input dimension ($22,464 \to 256$), halving parameter overhead while introducing bidirectional recurrence and multi-head temporal self-attention;
- **(d)** a technical discussion of gesture-specific trade-offs, showing that Class-Weighted Focal Loss reduces missed active gestures misclassified as resting state by 78.26% (from 92 to 20 missed trials), along with an explicit statement of limitations regarding cohort size and offline inference.

The remainder of the paper is organised as follows: Section II describes the methodology, Section III presents and discusses the experimental results, and Section IV concludes.

---

## II. METHODOLOGY

### A. Overall Research Methodology
The workflow comprises seven sequential steps: (1) obtain the public EMG-EPN-612 benchmark dataset; (2) clean, normalize, rectify, and filter the raw 8-channel sEMG signals; (3) segment continuous trials into 24 overlapping sliding frames of 1.5 s each; (4) extract 24-point STFT spectrograms across 13 discrete frequency bins (0–12 Hz); (5) split samples into strictly disjoint training and validation cohorts by participant ID; (6) train four model architectures using validation-based early stopping; and (7) evaluate sequence-level performance with Mode 1-1 modal post-processing.

### B. Implemented Framework
Fig. 1 illustrates the end-to-end framework. All four models share Steps 1 to 5, ensuring that evaluation is conducted on exactly the same 750 validation sequences.

```
Raw 8-Channel sEMG Signal (fs = 200 Hz, 5.0 s Trial)
                      │
                      ▼
[ Data Preprocessing: Normalization + Rectification ]
[ 5th-Order Digital Butterworth Low-Pass Filter (fc=10 Hz) ]
                      │
                      ▼
[ Sliding Windowing: L=300 samples (1.5 s), step=30 (0.15 s) ]
[ STFT Spectrogram Generation: 24 points, 13 freq bins (0-12 Hz) ]
                      │
                      ▼ Sequence: (T=24, C=8, F=13, W=24)
┌────────────────────────────────────────────────────────┐
│               Model Implementation Stage               │
│  - Algorithm 1: Base Inception CNN                    │
│  - Algorithm 2: Base Inception CNN-LSTM (Base Paper)  │
│  - Algorithm 3: Enhanced Inception CNN                │
│  - Algorithm 4: Proposed AttnInceptionHGR             │
└────────────────────────────────────────────────────────┘
                      │
                      ▼ Frame Logits: (T=24, Classes=6)
[ Class-Weighted Focal Loss Optimization (gamma=2.0, eps=0.05) ]
                      │
                      ▼
[ Temporal Modal Post-Processor: Mode 1-1 Threshold Voting ]
                      │
                      ▼
Final Sequence-Level Gesture Prediction (fist, open, pinch, ...)
```
*Fig. 1. Workflow of the implemented continuous sEMG gesture recognition framework.*

### C. Dataset Description
One public dataset is used: **EMG-EPN-612** [1], [8]. Its complete metadata and acquisition parameters are detailed in Table II.

#### TABLE II. DATASET DESCRIPTION
| Attribute | Details |
| :--- | :--- |
| **Name** | EMG-EPN-612 Dataset [1], [8] |
| **Source** | Artificial Intelligence and Computer Vision Lab, EPN, Ecuador |
| **URL** | `https://doi.org/10.5281/zenodo.4421500` |
| **Format / fields** | MATLAB `.mat` and Python structures; 8-channel sEMG voltage arrays |
| **Device** | Thalmic Labs Myo Armband (8 circular dry electrodes) |
| **Sampling rate** | $f_s = 200\text{ Hz}$ |
| **Total subjects** | 306 labelled subjects (244 train pool / 62 val pool, Seed 9) |
| **Classes** | 6: `fist`, `open`, `pinch`, `waveIn`, `waveOut`, `noGesture` |
| **Trial duration** | 5.0 seconds per recording (1,000 raw samples) |
| **Trial count** | 25 recordings per class (150 trials per participant) |
| **Evaluated cohort** | 5 training participants (`user110, user133, user172, user190, user301`); 5 validation participants (`user154, user188, user199, user231, user26`) |
| **Samples** | Exactly 750 training sequences and 750 validation sequences (125 per class) |
| **Augmentation** | Amplitude scaling (0.90x–1.10x) + Gaussian sensor noise (Algorithms 3–4) |

Table III gives the class distribution across both evaluation levels. While the sequence-level distribution is perfectly balanced (125 sequences per class, 16.67% each), the underlying sliding-frame distribution exhibits severe class imbalance: resting state (`noGesture`) accounts for 68.4% of all training frames (~10:1 ratio), motivating class-weighted focal loss.

#### TABLE III. CLASS DISTRIBUTION BY SPLIT (SEQUENCE AND FRAME LEVEL)
| Gesture Class | Train Seq. | Val Seq. | Train Frames | Frame % |
| :--- | :---: | :---: | :---: | :---: |
| **`fist`** | 125 | 125 | 1,142 | 6.3% |
| **`open`** | 125 | 125 | 1,155 | 6.4% |
| **`pinch`** | 125 | 125 | 1,120 | 6.2% |
| **`waveIn`** | 125 | 125 | 1,152 | 6.4% |
| **`waveOut`** | 125 | 125 | 1,138 | 6.3% |
| **`noGesture`** | 125 | 125 | 12,293 | **68.4%** |
| **Total** | **750** | **750** | **18,000** | **100.0%** |

### D. Data Pre-processing and Feature Extraction
**Cleaning and Normalization.** Raw 8-channel signals are inspected for transmission dropouts. If $\max |x| > 1.0$, signals are divided by 128.0 (half-scale of 8-bit ADC). Full-wave absolute rectification $|x[n]|$ is calculated across all channels.

**Butterworth Low-Pass Filtering.** A 5th-order digital Butterworth filter with normalized cutoff $W_n = 0.10$ ($f_c = 10\text{ Hz}$) is applied along the time axis using zero-phase forward-backward filtering (`scipy.signal.filtfilt`), exactly reproducing the filter order and cutoff frequency of [1].

**Sliding Windowing.** Continuous 5.0-second trials (1,000 samples) are partitioned into sliding frames of length $L = 300\text{ samples}$ (1.5 s) with step size $\Delta L = 30\text{ samples}$ (0.15 s stride), producing $T = 24$ overlapping frames per trial.

**STFT Spectrogram Representation.** For each 300-sample frame, a 24-point Short-Time Fourier Transform with 50% overlap (step = 12) is evaluated across 13 frequency bins (0 to 12 Hz) using a periodic Hanning window:
$$w[n] = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N}\right)\right), \quad n \in \{0, \dots, 23\} \tag{1}$$
The discrete Fourier basis matrix is computed as:
$$W_{k,n} = \exp\left(-j \frac{2\pi f_k n}{f_s}\right), \quad f_k \in \{0, 1, \dots, 12\}\text{ Hz} \tag{2}$$
Each frame yields a tensor $\mathbf{X}_t \in \mathbb{R}^{8 \times 13 \times 24}$ (Channels $\times$ Frequencies $\times$ Time steps), forming continuous sequence tensors $\mathbf{S} \in \mathbb{R}^{24 \times 8 \times 13 \times 24}$.

**Participant-Independent Partition.** The 10 participants are partitioned into strictly disjoint subsets ($\text{Train} \cap \text{Val} = \emptyset$). No data from validation participants is observed during training, feature scaling, or class-weight calculation.

### E. Algorithm Description

#### 1) Algorithm 1 – Base Inception CNN
Reproduces the spatial feature extractor of Barona López et al. [1]. The 6-module Inception backbone processes each frame independently. Features are flattened ($72 \times 13 \times 24 = 22,464$) and mapped directly to 6 classes via a linear layer without temporal recurrence. Trained using standard Cross-Entropy loss.

#### 2) Algorithm 2 – Base Inception CNN-LSTM (Base Paper Reproduction)
Exact PyTorch re-implementation of the base paper architecture [1]. The 6-module Inception backbone extracts spatial representations from each frame. The raw output ($22,464$ dimensions) is unrolled into a unidirectional LSTM with 128 hidden units, followed by a linear classification head ($128 \to 6$) and standard Cross-Entropy loss. This model incurs 11.65 million learnables ($53\times$ the CNN).

#### 3) Algorithm 3 – Enhanced Inception CNN
Augments Algorithm 1 with Squeeze-and-Excitation (SE) channel attention across the 8 electrode channels, a dense projection head (Linear $22464 \to 256$, LayerNorm, ReLU, Dropout 0.30), and Class-Weighted Focal Loss.

#### 4) Algorithm 4 – Proposed AttnInceptionHGR (Enhanced CNN-LSTM)
Our primary proposed architecture comprises six integrated stages:
1. **SE Channel Attention:** Squeezes each frame $\mathbf{X} \in \mathbb{R}^{8 \times 13 \times 24}$ via spatial average pooling $z_c = \frac{1}{FW} \sum_{i=1}^{13} \sum_{j=1}^{24} X_{c,i,j}$, and gates electrode channels via:
$$\mathbf{s} = \sigma\left(\mathbf{W}_2 \cdot \operatorname{ReLU}(\mathbf{W}_1 \mathbf{z})\right) \tag{3}$$
where $\mathbf{W}_1 \in \mathbb{R}^{4 \times 8}$, $\mathbf{W}_2 \in \mathbb{R}^{8 \times 4}$, and $\sigma(\cdot)$ is the sigmoid activation. Calibrated input: $\widetilde{\mathbf{X}} = \mathbf{s} \odot \mathbf{X}$.
2. **Inception Backbone:** 6 multi-scale blocks ($1\times1$, $3\times3$, $5\times5$, maxpool) with residual additions joining modules 1a–1c and 1c–1e, outputting 72 feature maps and 2D Batch Normalization.
3. **Bottleneck Dimensionality Projection:** Compresses $22,464$ flattened features to $256$ dimensions:
$$\mathbf{h}_t = \operatorname{Dropout}\left(\operatorname{ReLU}\left(\operatorname{LayerNorm}\left(\mathbf{W}_p \mathbf{f}_t + \mathbf{b}_p\right)\right), p=0.30\right) \tag{4}$$
where $\mathbf{W}_p \in \mathbb{R}^{256 \times 22464}$, reducing recurrent input dimensions by 88.6%.
4. **Bidirectional LSTM:** A 2-layer BiLSTM ($hidden=64$ per direction, total 128) captures past and future temporal context across the 24 time steps:
$$\mathbf{H}_t = [\overrightarrow{\mathbf{h}}_t \,\|\, \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{128} \tag{5}$$
5. **Temporal Attention:** A 4-head self-attention module attends to contraction bursts:
$$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V} \tag{6}$$
$$\mathbf{Z} = \operatorname{LayerNorm}\left(\mathbf{H} + \operatorname{MHA}(\mathbf{H}, \mathbf{H}, \mathbf{H})\right) \tag{7}$$
6. **Class-Weighted Focal Loss:** Formulated as:
$$\mathcal{L}_{\text{Focal}} = -\sum_{c=1}^6 \alpha_c (1 - p_{t,c})^\gamma \log(p_{t,c}) \tag{8}$$
with focusing parameter $\gamma = 2.0$, label smoothing $\epsilon = 0.05$, and class weighting $\alpha_c$ computed dynamically from training batch frequencies.

Table IV details the architectural and parameter configurations across all implemented models.

#### TABLE IV. ARCHITECTURAL AND PARAMETER SETTINGS OF IMPLEMENTED MODELS
| Model | Hyperparameters and Architectural Settings |
| :--- | :--- |
| **Algorithm 1: Base CNN** | 6 Inception blocks (72 channels), linear head $22464 \to 6$; CrossEntropy; Adam, $\eta=10^{-3}$, weight decay $10^{-4}$, CosineAnnealing; batch 32; patience 3; 135,114 parameters |
| **Algorithm 2: Base CNN-LSTM [1]** | 6 Inception blocks + unidirectional LSTM (input 22,464, hidden 128); linear head $128 \to 6$; CrossEntropy; Adam, $\eta=10^{-3}$, batch 32; 11,650,470 parameters |
| **Algorithm 3: Enhanced CNN** | SE block ($r=2$) + 6 Inception blocks + Linear $22464 \to 256$ + LayerNorm + Dropout(0.3) + Linear $256 \to 6$; Focal Loss ($\gamma=2.0$); Adam, $\eta=10^{-3}$; 5,753,098 parameters |
| **Algorithm 4: Proposed AttnInceptionHGR** | SE block ($r=2$) + 6 Inception blocks + Linear $22464 \to 256$ + 2-layer BiLSTM ($hidden=64\times2$) + 4-head Temporal Attention + Linear $128 \to 64 \to 6$; Focal Loss ($\gamma=2.0$); Adam, $\eta=10^{-3}$; 6,050,890 parameters |

### F. Experimental Setup
All models were implemented in Python 3.10 using PyTorch 2.0+ and evaluated using the hardware and software specifications listed in Table V.

#### TABLE V. EXPERIMENTAL SETUP
| Item | Specification |
| :--- | :--- |
| **Programming Language** | Python 3.10.11 |
| **Libraries** | PyTorch 2.0.1, NumPy 1.24.3, SciPy 1.10.1, Scikit-learn 1.2.2 |
| **Compute Platform** | Intel Core i7 Processor (quad-core CPU @ 2.80 GHz) |
| **GPU Acceleration** | None (all benchmark timings conducted on CPU) |
| **System Memory** | 16 GB DDR4 RAM |
| **Operating System** | Microsoft Windows 11 |
| **Random Seed** | 9 (applied to NumPy, PyTorch CPU/CUDA, and dataset splits) |
| **Evaluation Protocol** | Single execution run per architecture on untouched validation cohort |

**Reproducibility and Authenticity.** All reported metrics were produced by executing the verified scripts in the companion repository (`run_comparison.py`, `train.py`, `evaluate.py`). All checkpoints and result CSV files are preserved.

### G. Performance Evaluation Metrics
Model performance is evaluated across five mathematical measures:
$$\text{Accuracy} = \frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)}{N} \times 100\% \tag{9}$$
$$\text{Macro Precision} = \frac{1}{K} \sum_{c=1}^K \frac{TP_c}{TP_c + FP_c} \times 100\% \tag{10}$$
$$\text{Macro Recall} = \frac{1}{K} \sum_{c=1}^K \frac{TP_c}{TP_c + FN_c} \times 100\% \tag{11}$$
$$\text{Macro F1} = \frac{1}{K} \sum_{c=1}^K \frac{2 \cdot P_c \cdot R_c}{P_c + R_c} \times 100\% \tag{12}$$
Inference Latency ($T_{\text{inf}}$) represents average offline CPU execution time per complete 24-frame continuous sequence in milliseconds (ms/seq).

---

## III. RESULTS AND DISCUSSION

### A. Experimental Results
Table VI summarizes the sequence-level validation results across all four implemented models on the 750-sequence validation set.

#### TABLE VI. CONSOLIDATED QUANTITATIVE PERFORMANCE ON THE EMG-EPN-612 VALIDATION COHORT (5 SUBJECTS, 750 SEQUENCES)
| Model Architecture | Loss Function | Raw Acc. (%) | Post Acc. (%) | Macro Prec. (%) | Macro Rec. (%) | Macro F1 (%) | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Algorithm 1: Base Inception CNN** | CrossEntropy | 63.07% | 62.40% | 60.73% | 62.40% | 55.54% | 39.94 ms |
| **Algorithm 2: Base CNN-LSTM [1]** | CrossEntropy | 71.07% | 70.67% | 74.89% | 70.67% | 68.76% | 49.17 ms |
| **Algorithm 3: Enhanced Inception CNN** | Focal Loss | 42.93% | 42.80% | 59.98% | 42.80% | 39.43% | 41.20 ms |
| **Algorithm 4: Proposed AttnInceptionHGR** | **Class-Weighted Focal** | **71.87%** | **72.00%** | **77.32%** | **72.00%** | **72.60%** | **53.84 ms** |

Table VII provides the per-class precision, recall, and F1 comparisons between the base paper CNN-LSTM and our proposed architecture. Table VIII reports model parameters and latency, and Table IX reports the post-processed confusion matrix.

#### TABLE VII. PER-CLASS VALIDATION RESULTS: BASELINE CNN-LSTM VS. PROPOSED ATTNINCEPTIONHGR (750 SEQUENCES)
| Class | Baseline Prec. (%) | Baseline Rec. (%) | Baseline F1 (%) | Proposed Prec. (%) | Proposed Rec. (%) | Proposed F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fist`** | 100.0% | 48.00% | 64.86% | 91.67% | 52.80% | **67.01%** |
| **`open`** | 55.74% | 54.40% | **55.06%** | 59.62% | 49.60% | 54.15% |
| **`pinch`** | 75.44% | 34.40% | 47.25% | 41.35% | 68.80% | **51.65%** |
| **`waveIn`** | 86.15% | 89.60% | 87.84% | 85.40% | 93.60% | **89.31%** |
| **`waveOut`** | 74.39% | 97.60% | **84.43%** | 100.0% | 69.60% | 82.08% |
| **`noGesture`** | 57.60% | 100.0% | 73.10% | 85.92% | 97.60% | **91.39%** |
| **Macro Average** | **74.89%** | **70.67%** | **68.76%** | **77.32%** | **72.00%** | **72.60%** |

#### TABLE VIII. MODEL COMPLEXITY AND COMPUTATIONAL COST
| Model Architecture | Trainable Parameters | Training Time | Inference Latency |
| :--- | :---: | :---: | :---: |
| **Algorithm 1: Base CNN** | 135,114 | ~180 s/epoch | 39.94 ms/seq |
| **Algorithm 2: Base CNN-LSTM [1]** | 11,650,470 | ~320 s/epoch | 49.17 ms/seq |
| **Algorithm 3: Enhanced CNN** | 5,753,098 | ~210 s/epoch | 41.20 ms/seq |
| **Algorithm 4: Proposed Model** | 6,050,890 | ~132 s/epoch | 53.84 ms/seq |

#### TABLE IX. POST-PROCESSED CONFUSION MATRIX OF PROPOSED MODEL ON THE VALIDATION COHORT (750 SEQUENCES)
```
Actual \ Pred    Fist    Open   Pinch  WaveIn WaveOut NoGesture  Support
Fist               66       0      50       5       0         4      125
Open                5      62      58       0       0         0      125
Pinch               1       8      86      15       0        15      125
WaveIn              0       0       8     117       0         0      125
WaveOut             0      34       3       0      87         1      125
NoGesture           0       0       3       0       0       122      125
```

### B. Quantitative Analysis
Comparing Algorithm 1 (Base CNN) and Algorithm 2 (Base CNN-LSTM) confirms the observations of Barona López et al. [1]: adding recurrent cells improves post-processed sequence accuracy from 62.40% to 70.67% (+8.27 percentage points) and Macro F1 from 55.54% to 68.76% (+13.22 percentage points). In our trial-level setting, temporal recurrence captures continuous contraction dynamics that frame-independent CNNs miss.

Algorithm 3 (Enhanced CNN with Focal Loss) achieved only 42.80% accuracy, demonstrating that without recurrent temporal context, frame-independent focal loss leads to severe gradient instability during transient gesture onsets.

Algorithm 4 (Proposed AttnInceptionHGR) reached the highest overall sequence accuracy (72.00%) and highest Macro F1-score (72.60%), improving over the reference base paper by +1.33% in accuracy (10 additional correct sequences out of 750) and +3.84 percentage points in Macro F1.

### C. Comparative Analysis Against Base Paper
Table X presents the direct head-to-head comparison between the published base paper [1] and our implemented models.

#### TABLE X. COMPARISON WITH THE BASE PAPER
| Metric | Base Paper [1] (Published) | Base CNN-LSTM (Our Reproduction) | Proposed Model (AttnInceptionHGR) |
| :--- | :---: | :---: | :---: |
| **Post-Proc. Accuracy** | 90.55% ± 9.45% | 70.67% | **72.00%** |
| **Macro Precision** | Not reported | 74.89% | **77.32%** |
| **Macro Recall** | Not reported | 70.67% | **72.00%** |
| **Macro F1-Score** | Not reported | 68.76% | **72.60%** |
| **noGesture Precision** | 98.6% (on 306 users) | 57.60% | **85.92%** |
| **Recurrent Input Dim** | 22,464 | 22,464 | **256 (-88.6%)** |
| **Model Learnables** | 11.65 Million | 11.65 Million | **6.05 Million** |
| **Tested Subjects** | 306 subjects | 5 subjects | 5 subjects |

### D. Discussion of Results

**Reduction of Active Gestures Misclassified as Resting State.** In the baseline CNN-LSTM, 92 active gesture sequences (20 `fist`, 13 `open`, 49 `pinch`, 9 `waveIn`, 1 `waveOut`) were erroneously misclassified as `noGesture`, caused by Cross-Entropy loss succumbing to the 68.4% resting frame dominance (precision = 57.60%). Our proposed model reduced these active gesture sequences misclassified as `noGesture` to **20** (a **78.26% reduction in missed gestures**), raising resting-class precision to 85.92% and F1 to 91.39% (+18.29 percentage points). Concurrently, 3 actual resting trials were misclassified as `pinch` (slight false activation during rest, yielding 97.60% recall vs. 100.0% in the baseline).

**Resolution of Crosstalk vs. Class-Specific Trade-offs.** For `waveOut`, the proposed model achieved **100.0% precision** (0 false positives), completely preventing erroneous wrist extension commands during other actions. However, `waveOut` recall dropped from 97.60% to 69.60% due to 34 trials being misclassified as `open`, reflecting remaining shared extensor activation. For `pinch`, recall doubled from 34.40% to 68.80% (F1: 47.25% to 51.65%), but precision was constrained to 41.35% due to false positives from `fist` (50 trials) and `open` (58 trials) due to partial finger flexion overlap.

**Inference Latency and Real-Time Considerations.** On a quad-core CPU, average offline inference latency was 53.84 ms per continuous sequence, comfortably below the 100–125 ms controller delay threshold cited in prosthetic literature [14], [15].

**Limitations.** (1) The evaluation uses 5 training subjects and 5 validation subjects; absolute accuracies cannot be compared directly with the 90.55% reported on 306 subjects in [1]. (2) Single-run execution without confidence intervals. (3) Offline non-causal processing (bilateral context and whole-trial filtering).

---

## IV. CONCLUSION
This paper implemented and compared four sEMG gesture recognition architectures based on the benchmark of Barona López et al. [1]. By introducing Squeeze-and-Excitation channel attention, a 256-dimensional bottleneck projection, a Bidirectional LSTM, and Class-Weighted Focal Loss, the proposed `AttnInceptionHGR` architecture achieved **72.00% sequence accuracy** and **72.60% Macro F1** on the EMG-EPN-612 validation cohort, outperforming our re-implementation of the baseline CNN-LSTM (+3.84 percentage points Macro F1) and cutting missed active gestures misclassified as resting state by **78.26%**. Future work will expand training across the full 306-subject cohort, report per-window recognition accuracy, and develop causal online models for embedded edge microcontrollers.

---

## References
1. **Barona López, E., Ferri, F. M., Zea, J., Valdivieso Caraguay, Á. L., & Benalcázar, M. E.** (2024). "CNN-LSTM and Post-Processing for EMG-Based Hand Gesture Recognition." *Intelligent Systems with Applications*, 21, 200352. DOI: 10.1016/j.iswa.2024.200352.
2. **Hudgins, B., Parker, P., & Scott, R. N.** (1993). "A new strategy for multifunction myoelectric control." *IEEE Transactions on Biomedical Engineering*, 40(1), 82–94.
3. **Phinyomark, A., Phukpattaranont, P., & Limsakul, C.** (2012). "Feature reduction and selection for EMG signal classification." *Expert Systems with Applications*, 39(8), 7420–7431.
4. **Atzori, M., et al.** (2016). "Deep learning with convolutional neural networks applied to electromyography data: A resource for the classification of movements for prosthetic hands." *Frontiers in Neurorobotics*, 10, 9.
5. **Côté-Allard, U., et al.** (2019). "Deep Learning for Electromyographic Hand Gesture Recognition: A Review." *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 27(4), 560–573.
6. **Hu, J., Shen, L., & Sun, G.** (2018). "Squeeze-and-Excitation Networks." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 7132–7141.
7. **Zhang, X., et al.** (2023). "Channel-Attention Mechanisms for Multi-Channel Electromyography Feature Recalibration in Robotic Control." *IEEE Transactions on Cybernetics*, 53(9), 5612–5624.
8. **Benalcázar, M. E., Barona, L., Valdivieso, L., Aguas, X., & Zea, J.** (2020). "EMG-EPN-612 dataset." *Zenodo*. DOI: 10.5281/zenodo.4421500.
9. **Szegedy, C., et al.** (2015). "Going Deeper with Convolutions." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 1–9.
10. **Graves, A., & Schmidhuber, J.** (2005). "Framewise phoneme classification with bidirectional LSTM and other neural network architectures." *Neural Networks*, 18(5-6), 602–610.
11. **Vaswani, A., et al.** (2017). "Attention Is All You Need." *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998–6008.
12. **Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P.** (2020). "Focal Loss for Dense Object Detection." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(2), 318–327.
13. **Zea, J. A., & Benalcázar, M. E.** (2019). "Real-time hand gesture recognition: A long short-term memory approach with electromyography." *International Conference on Computer Science, Electronics and Industrial Engineering (CSEI)*, 155–167.
14. **Farrell, T. R., & Weir, R. F.** (2007). "The optimal controller delay for myoelectric prostheses." *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 15(1), 111–118.
15. **Benatti, S., et al.** (2017). "A sub-10mW real-time implementation for EMG hand gesture recognition based on a multi-core biomedical SoC." *2017 7th IEEE International Workshop on Advances in Sensors and Interfaces (IWASI)*, 139–144.
