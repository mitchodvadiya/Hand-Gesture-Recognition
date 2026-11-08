import numpy as np
from scipy import signal
from config import Config

class EMGPreprocessor:
    """
    Implements signal preprocessing and STFT spectrogram generation
    identical to the MATLAB implementation in the base paper.
    """
    def __init__(self):
        # Precompute 5th order Butterworth lowpass filter coefficients
        # In MATLAB: butter(5, 0.1, 'low')
        self.b, self.a = signal.butter(Config.BUTTER_ORDER, Config.BUTTER_CUTOFF, btype='low')
        
        # Precompute DFT matrix for exact frequencies 0:12 Hz at fs = 200 Hz
        # Window size = 24, frequencies = 0..12
        N = Config.WINDOW_SIZE
        fs = Config.SAMPLING_RATE
        freqs = np.array(Config.FREQUENCIES, dtype=np.float32) # shape (13,)
        n = np.arange(N, dtype=np.float32)                     # shape (24,)
        
        # Periodic Hanning window matching MATLAB: hann(24, 'periodic')
        self.hann_window = 0.5 * (1 - np.cos(2 * np.pi * n / N))
        
        # DFT basis matrix: shape (13, 24)
        # W[k, n] = exp(-j * 2 * pi * f_k * n / fs)
        angle = -2.0 * np.pi * np.outer(freqs, n) / fs
        self.dft_basis = np.exp(1j * angle)

    def preprocess_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Preprocesses raw 8-channel EMG signal:
        1. Normalization: if max(abs) > 1, divide by 128
        2. Full-wave absolute rectification
        3. 5th order low-pass Butterworth filtering with zero phase delay
        """
        sig = np.array(raw_signal, dtype=np.float32)
        
        # 1. Normalization
        max_val = np.max(np.abs(sig))
        if max_val > 1.0:
            sig = sig / 128.0
            
        # 2. Rectification (absolute value)
        sig = np.abs(sig)
        
        # 3. Butterworth low-pass filter (filtfilt along time axis)
        # Pad length handled safely for short signals
        padlen = min(3 * max(len(self.a), len(self.b)), sig.shape[0] - 1)
        filtered_sig = signal.filtfilt(self.b, self.a, sig, axis=0, padlen=padlen)
        return filtered_sig

    def compute_spectrogram(self, frame_signal: np.ndarray) -> np.ndarray:
        """
        Computes 8-channel spectrogram for a frame of 300 samples.
        Frame shape: (300, 8)
        Output shape: (8, 13, 24) -> (Channels, FrequencyBins, TimeColumns)
        """
        num_samples, num_channels = frame_signal.shape
        step = Config.WINDOW_SIZE - Config.OVERLAP # 24 - 12 = 12
        num_cols = (num_samples - Config.OVERLAP) // step # (300 - 12) // 12 = 24
        
        spectrograms = np.zeros((num_channels, len(Config.FREQUENCIES), num_cols), dtype=np.float32)
        
        for c in range(num_channels):
            chan_data = frame_signal[:, c]
            for col in range(num_cols):
                start = col * step
                end = start + Config.WINDOW_SIZE
                segment = chan_data[start:end] * self.hann_window
                
                # Discrete Fourier evaluation at 0:12 Hz
                stft_vals = np.dot(self.dft_basis, segment)
                # Power spectral density: |s|^2
                spectrograms[c, :, col] = np.abs(stft_vals) ** 2
                
        return spectrograms

    def extract_frames_from_sample(self, emg_signal: np.ndarray, ground_truth: np.ndarray, 
                                  num_gesture_points: int, gesture_name: str):
        """
        Extracts sliding window spectrogram frames from a gesture recording.
        Returns:
            frames: list of np.ndarray of shape (8, 13, 24)
            labels: list of str (gesture_name or 'noGesture')
            timestamps: list of int
        """
        preprocessed = self.preprocess_signal(emg_signal)
        sig_len = len(preprocessed)
        
        num_windows = (sig_len - Config.FRAME_WINDOW) // Config.WINDOW_STEP + 1
        frames, labels, timestamps = [], [], []
        
        for i in range(num_windows):
            start = i * Config.WINDOW_STEP
            end = start + Config.FRAME_WINDOW
            timestamp = start + Config.FRAME_WINDOW // 2
            
            gt_slice = ground_truth[start:end]
            total_ones = np.sum(gt_slice == 1)
            
            # Check threshold to classify frame as gesture or noGesture
            is_gesture = (total_ones >= Config.FRAME_WINDOW * Config.TOLERANCE_WINDOW) or \
                         (num_gesture_points > 0 and total_ones >= num_gesture_points * Config.TOLERANCE_GESTURE)
            
            frame_spec = self.compute_spectrogram(preprocessed[start:end, :])
            frames.append(frame_spec)
            labels.append(gesture_name if is_gesture else "noGesture")
            timestamps.append(timestamp)
            
        return frames, labels, timestamps
