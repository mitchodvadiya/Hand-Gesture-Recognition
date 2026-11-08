# Hand Gesture Recognition (HGR) Research Codebase (PyTorch)

Based on the research paper:
> **"Assessing the influence of LSTM and post-processing in CNN based Hand Gesture Recognition using EMG"**  
> *Lorena Isabel Barona López, Francis M. Ferri, Jonathan Zea, Ángel Leonardo Valdivieso Caraguay, Marco E. Benalcázar* (Intelligent Systems with Applications, Elsevier, 2024).

---

## 🌟 Key Features & Improvements
1. **Base Paper Model Reproduction:** Faithful PyTorch implementation of the 6-module Inception CNN + Residual Skips + LSTM sequence model.
2. **Novel Enhanced Architecture (To Outperform Base Paper > 90.55%):**
   - **Electrode Channel Attention:** Squeeze-and-Excitation across the 8 Myo armband sensors.
   - **Bidirectional LSTM (BiLSTM):** Captures forward and reverse temporal muscle dynamics.
   - **Multi-Head Temporal Self-Attention:** Dynamically focuses on muscle burst contraction phases.
   - **Dropout & Layer Normalization:** Strong regularization for cross-subject generalization.
3. **Myo Armband Circular Electrode Augmentation:** Simulates armband physical rotation around the forearm (`torch.roll`), amplitude scaling for contraction intensity, and Gaussian sensor noise.
4. **Class-Weighted Focal Loss & Label Smoothing:** Dynamically suppresses easy resting frames (`noGesture`) and forces the gradient to focus on hard, ambiguous gesture contractions with label smoothing regularization.
5. **Exact Post-Processing Filter:** Full Python implementation of the `1-1`, `1-2`, and `2-1` temporal label filtering algorithms that provided the massive **+24.77% to +41.86%** accuracy boost in the original paper.
6. **Fast Disk Caching:** Processed STFT spectrograms are cached to disk on first run for instantaneous future loads.
7. **Official Benchmark Generator:** Generates `responses.json` matching the format required for the EMG-EPN-612 public evaluation platform.

---

## 📁 File Structure
```
python_code/
├── config.py              # Central configuration (paths, STFT parameters, classes, hyperparameters)
├── preprocessing.py       # Butterworth 5th order low-pass filter, rectification, STFT spectrogram generation
├── dataset.py             # PyTorch Dataset, DataLoaders, and physical data augmentations
├── models.py              # 4 Models: Base CNN, Enhanced CNN, Base CNN-LSTM, Enhanced CNN-LSTM
├── postprocess.py         # Heuristic 1-1, 1-2, 2-1 sequence filtering & threshold voting
├── metrics.py             # Accuracy, Macro Precision, Recall, Macro F1, Confusion Matrix & CSV exports
├── train.py               # Complete training script with training-only class weighting and Focal Loss
├── evaluate.py            # Evaluation script with per-class metrics, confusion matrix, and CSV exports
├── run_comparison.py      # Automated 4-model comparison on identical participant-independent split
├── unit_tests.py          # Complete unit test & smoke test suite
├── test_evaluation.py     # Generates responses.json on the test set for public benchmark submission
└── requirements.txt       # Dependencies (torch, scipy, numpy, matplotlib, tqdm)
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Training the Model

#### A. Train the Enhanced Model (Recommended - Highest Accuracy)
```bash
python train.py --model enhanced --epochs 15 --batch_size 64 --lr 0.001
```

#### B. Train the Base Paper Inception CNN-LSTM Model
```bash
python train.py --model base --epochs 15 --batch_size 64 --lr 0.001
```

#### Fast Debugging Run (subset of 10 users):
```bash
python train.py --model enhanced --epochs 3 --max_users 10
```

*Trained model checkpoints are automatically saved in `checkpoints/best_model_{model}.pth`.*

---

### 3. Evaluating on the Validation or Test Set
To see accuracy **with** and **without** post-processing, and get per-gesture breakdown:
```bash
python evaluate.py --checkpoint checkpoints/best_model_enhanced.pth --split val
```
Output results summary will be saved to `results/eval_val_enhanced.json`.

---

### 4. Generating Benchmark Responses for Test Set
To generate the final `responses.json` for submission / testing evaluation:
```bash
python test_evaluation.py --checkpoint checkpoints/best_model_enhanced.pth
```
The output file will be saved to `results/responses.json`.
