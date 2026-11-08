# Pipeline Audit Report: Data Splits, Cache Provenance, and Verification

## 1. Participant-Level Data Splitting & Disjointness

- **Dataset Root:** `EMG-EPN-612 dataset/`
- **Training Directory:** `trainingJSON/` contains 306 participant directories (`user1` to `user306`).
- **Testing Directory:** `testingJSON/` contains 306 independent participant directories (`user1` to `user306`).
- **Partitioning Algorithm (`dataset.py:157-164`):**
  - Random seed: Fixed to `np.random.seed(9)` (matching `rng(9)` from reference paper `Shared.m:86`).
  - Total participants in `trainingJSON`: 306.
  - Split Index: `split_idx = int(0.8 * 306) = 244`.
  - Training Pool: First 244 shuffled participants (`shuffled_indices[:244]`).
  - Validation Pool: Remaining 62 shuffled participants (`shuffled_indices[244:]`).
- **Overlap Verification:**
  - $\text{TrainParticipants} \cap \text{ValidationParticipants} = \emptyset$.
  - Enforced dynamically in [`dataset.py:278`](file:///D:/SEM_5/DSIP/Research_Code/python_code/dataset.py#L278) with an assertion:
    ```python
    assert len(overlap) == 0, f"DATA LEAKAGE ERROR: Overlapping participants detected: {overlap}"
    ```
- **Independent Test Set Isolation:**
  - `testingJSON` (306 participants) is reserved exclusively for final benchmark evaluation via `test_evaluation.py` (generating `responses.json`).
  - No training or validation optimization accesses `testingJSON`.

---

## 2. Cache Provenance and Cache-Tag Protection

During Phase 1 auditing, a static cache filename vulnerability was identified and resolved. Caches now strictly embed user count and split identifiers:

| Cache File | Data Split | Participant Count | Total Sequences | Sequences Per Class | Stored Tensor Shape | Integrity Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `train_seq_cache_u1.pt` | Train | 1 (`user172`) | 150 | 25 | $(24, 8, 13, 24)$ | Verified |
| `val_seq_cache_u1.pt` | Val | 1 (`user154`) | 150 | 25 | $(24, 8, 13, 24)$ | Verified |
| `train_seq_cache_u5.pt` | Train | 5 (`user172, user19, ...`) | 750 | 125 | $(24, 8, 13, 24)$ | Verified |
| `val_seq_cache_u5.pt` | Val | 5 (`user154, user188, ...`) | 750 | 125 | $(24, 8, 13, 24)$ | Verified |
| `train_seq_cache.pt` | Train | 10 (Legacy debug cache) | 1,500 | 250 | $(24, 8, 13, 24)$ | Preserved |
| `val_seq_cache.pt` | Val | 10 (Legacy debug cache) | 1,500 | 250 | $(24, 8, 13, 24)$ | Preserved |
| `val_seq_cache_all.pt` | Val | 62 (Full validation pool) | 9,300 | 1,550 | $(24, 8, 13, 24)$ | Verified |

- **Cache Selection Guard:** [`dataset.py:104-129`](file:///D:/SEM_5/DSIP/Research_Code/python_code/dataset.py#L104-L129) verifies metadata manifests and rejects/rebuilds mismatched caches automatically if a requested user count differs from the file's payload.

---

## 3. Label Provenance and Mapping Integrity

- **Visual Stimulus Ground Truth:**
  - In each trial recording, `gestureName` is extracted directly from the JSON metadata (`dataset.py:191`).
  - `sample['true_gesture'] = gesture_name`.
  - Serves as the definitive sequence-level ground truth.
- **Frame-Level Contraction Targets:**
  - Evaluated in [`preprocessing.py:103-104`](file:///D:/SEM_5/DSIP/Research_Code/python_code/preprocessing.py#L103-L104) against binary active contraction interval `groundTruth`.
  - Frame labeled with `gesture_name` if $\ge 75\%$ active samples or $\ge 50\%$ gesture length; else `'noGesture'`.
- **Six-Class Bijection Table:**

  | Index | String Key | Category |
  | :---: | :---: | :---: |
  | 0 | `fist` | Contraction |
  | 1 | `open` | Contraction |
  | 2 | `pinch` | Contraction |
  | 3 | `waveIn` | Contraction |
  | 4 | `waveOut` | Contraction |
  | 5 | `noGesture` | Baseline Rest |

- **Integrity Status:** Bijective, consistent across all modules, zero leakage of predicted labels into targets.

---

## 4. Preprocessing & Normalization Independence

- **Normalization:** In [`preprocessing.py:41`](file:///D:/SEM_5/DSIP/Research_Code/python_code/preprocessing.py#L41):
  ```python
  if max_val > 1.0:
      sig = sig / 128.0
  ```
  Divides raw 8-bit ADC readings by fixed hardware constant $128.0$.
  **Zero dataset-level mean, variance, or min-max normalization is computed.**
- **Filtering:** 5th-order Butterworth lowpass filter with constant coefficients ($f_s = 200\text{ Hz}$, $f_c = 10\text{ Hz}$).
- **Spectrogram:** Fixed 24-point Hann window with discrete Fourier basis over frequencies $0 \dots 12\text{ Hz}$.
- **Zero Information Leakage:** Every signal trial is transformed strictly independently.

---

## 5. Checkpoint Selection & Evaluation Integrity

- **Selection Metric:** Best validation post-processed recognition accuracy ([`train.py:299-316`](file:///D:/SEM_5/DSIP/Research_Code/python_code/train.py#L299-L316)) and Macro F1.
- **Inference Mode:** Evaluated exclusively under `torch.no_grad()`.
- **Latency Measurement:** High-resolution `time.perf_counter()` timing forward pass per sequence.
- **Verification Tests:** Passed all 4 pipeline integrity tests (`test_pipeline_fix.py`) and all 5 evaluation audit tests (`test_evaluation_audit.py`).
- **Pipeline Verdict:** Certified trustworthy for controlled experimental execution.
