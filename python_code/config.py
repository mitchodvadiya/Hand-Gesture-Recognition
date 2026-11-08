import os
import torch

class Config:
    # -------------------------------------------------------------
    # Paths configuration
    # -------------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Path to dataset (handles both folder naming conventions)
    RAW_DATA_DIR = os.path.join(BASE_DIR, "EMG-EPN612 Dataset", "EMG-EPN612 Dataset")
    if not os.path.exists(RAW_DATA_DIR):
        RAW_DATA_DIR = os.path.join(BASE_DIR, "EMG-EPN-612 dataset")
        
    TRAINING_DIR = os.path.join(RAW_DATA_DIR, "trainingJSON")
    TESTING_DIR = os.path.join(RAW_DATA_DIR, "testingJSON")
    
    # Output directories for processed caches and checkpoints
    CACHE_DIR = os.path.join(BASE_DIR, "python_code", "data_cache")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "python_code", "checkpoints")
    RESULTS_DIR = os.path.join(BASE_DIR, "python_code", "results")

    # -------------------------------------------------------------
    # EMG Signal & Spectrogram Parameters (Matching Base Paper)
    # -------------------------------------------------------------
    SAMPLING_RATE = 200        # Hz (Myo Armband)
    NUM_CHANNELS = 8           # 8 electrode sensors
    
    # Butterworth low-pass filter
    BUTTER_ORDER = 5
    BUTTER_CUTOFF = 0.1        # Normalized frequency (0.1 * (fs/2) = 10 Hz)
    
    # Spectrogram parameters
    WINDOW_SIZE = 24           # 24 samples
    OVERLAP = 12               # 12 samples (50% overlapping)
    FREQUENCIES = list(range(13))  # Frequencies 0 to 12 Hz (13 frequency bins)
    
    # Windowing for gesture frames
    FRAME_WINDOW = 300         # 300 samples (1.5 seconds)
    WINDOW_STEP = 30           # 30 samples step (0.15s stride)
    
    # Gesture inclusion thresholds
    TOLERANCE_WINDOW = 0.75    # >= 75% of frame in ground truth
    TOLERANCE_GESTURE = 0.5    # >= 50% of gesture length
    
    # Input spectrogram frame dimensions: (Channels=8, FreqBins=13, TimeSteps=24)
    SPECTROGRAM_SHAPE = (8, 13, 24)
    
    # -------------------------------------------------------------
    # Gesture Classes
    # -------------------------------------------------------------
    GESTURES = ["fist", "open", "pinch", "waveIn", "waveOut"]
    ALL_CLASSES = ["fist", "open", "pinch", "waveIn", "waveOut", "noGesture"]
    NUM_CLASSES = len(ALL_CLASSES)
    
    CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}
    IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(ALL_CLASSES)}

    # -------------------------------------------------------------
    # Post-Processing Parameters (from Base Paper)
    # -------------------------------------------------------------
    MIN_LABELS_SEQUENCE = 4    # Minimum consecutive/majority labels to trigger gesture
    POSTPROCESS_MODE = '1-1'   # '1-1', '1-2', or '2-1'
    
    # -------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 15
    LSTM_HIDDEN_DIM = 128
    
    # Device and Seed configuration
    RANDOM_SEED = 9
    TRAIN_RATIO = 0.8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0  # 0 for safe multiprocessing on Windows

    # -------------------------------------------------------------
    # Data Augmentation & Focal Loss Parameters
    # -------------------------------------------------------------
    # Note on Circular Shift: Verified physical ordering of the 8 Myo electrodes
    # corresponds to antagonistic muscle pairs (flexors vs extensors). Arbitrary
    # circular shifts create physically invalid samples for directional gestures
    # (waveIn, waveOut) and are therefore disabled by default.
    ENABLE_CIRCULAR_SHIFT = False 
    CIRCULAR_SHIFT_PROB = 0.0
    
    # Topology-preserving physical augmentations
    USE_AUGMENTATION = True
    SCALE_AUG_PROB = 0.5         # Simulates varying contraction intensities (0.90 - 1.10)
    NOISE_AUG_PROB = 0.3         # Sensor / electrode contact noise injection
    
    # Loss function configuration
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0            # Down-weights easy background frames
    LABEL_SMOOTHING = 0.05       # Prevents overconfidence on unseen subjects

