import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config
from preprocessing import EMGPreprocessor

def load_user_json(user_dir: str):
    """Loads user JSON data from user folder."""
    user_name = os.path.basename(user_dir)
    json_path = os.path.join(user_dir, f"{user_name}.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_emg_channels(emg_dict: dict) -> np.ndarray:
    """Converts 8-channel EMG dict into numpy array of shape (T, 8)."""
    channel_keys = [f"ch{i}" for i in range(1, Config.NUM_CHANNELS + 1)]
    # Verify channels exist
    for k in channel_keys:
        if k not in emg_dict:
            # Fallback to whatever keys exist sorted
            channel_keys = sorted(emg_dict.keys())
            break
            
    num_pts = len(emg_dict[channel_keys[0]])
    signal = np.zeros((num_pts, Config.NUM_CHANNELS), dtype=np.float32)
    for c, key in enumerate(channel_keys):
        signal[:, c] = emg_dict[key]
    return signal


class EMGAugmentor:
    """
    Physically Verified EMG Data Augmentation Pipeline:
    1. Amplitude Variation: Simulates contraction intensity and effort variations (0.90x - 1.10x).
    2. Additive Sensor Noise: Simulates skin-electrode contact impedance fluctuations.
    Note: Circular channel shifting is disabled by default because the 8 Myo electrodes
    cover distinct antagonistic muscle groups (flexors vs extensors); shifting channels
    creates physiologically invalid patterns for directional gestures (waveIn, waveOut).
    """
    def __init__(self,
                 enable_circular_shift: bool = Config.ENABLE_CIRCULAR_SHIFT,
                 circular_shift_prob: float = Config.CIRCULAR_SHIFT_PROB,
                 scale_prob: float = Config.SCALE_AUG_PROB,
                 noise_prob: float = Config.NOISE_AUG_PROB):
        self.enable_circular_shift = enable_circular_shift
        self.circular_shift_prob = circular_shift_prob
        self.scale_prob = scale_prob
        self.noise_prob = noise_prob

    def apply(self, seq: torch.Tensor) -> torch.Tensor:
        # seq shape: (T, Channels=8, Freq=13, TimeBins=24)
        out = seq.clone()
        
        # 1. Circular channel shift (only if explicitly enabled)
        if self.enable_circular_shift and np.random.rand() < self.circular_shift_prob:
            shift = int(np.random.choice([-1, 1]))
            out = torch.roll(out, shifts=shift, dims=1)
            
        # 2. Random amplitude scaling (contraction intensity variation, preserves spatial arrangement)
        if np.random.rand() < self.scale_prob:
            scale_factor = float(np.random.uniform(0.90, 1.10))
            out = out * scale_factor
            
        # 3. Additive Gaussian noise (sensor impedance fluctuations)
        if np.random.rand() < self.noise_prob:
            seq_std = float(out.std().item())
            noise = torch.randn_like(out) * (0.015 * (seq_std + 1e-6))
            out = torch.clamp(out + noise, min=0.0)
            
        return out


class EMGSequenceDataset(Dataset):
    """
    Sequence-level Dataset for CNN-LSTM and Enhanced HGR Model.
    Each item is:
        spectrogram_sequence: Tensor of shape (T, 8, 13, 24)
        target_labels: Tensor of shape (T,) with integers 0..5
        sample_meta: dict containing true_gesture, user_id, sample_id
    """
    def __init__(self, data_split: str = 'train', max_users: int = None, use_cache: bool = True, augment: bool = False):
        super().__init__()
        self.data_split = data_split
        self.augment = augment
        self.augmentor = EMGAugmentor() if augment else None
        self.preprocessor = EMGPreprocessor()
        self.samples = []
        
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        cache_tag = f"_u{max_users}" if (max_users is not None and max_users > 0) else "_all"
        self.cache_file = os.path.join(Config.CACHE_DIR, f"{data_split}_seq_cache{cache_tag}.pt")
        
        if use_cache and os.path.exists(self.cache_file):
            print(f"Loading cached {data_split} dataset from {self.cache_file}...")
            cached_data = torch.load(self.cache_file, weights_only=False)
            
            # Validate cache integrity against requested configuration
            is_valid = True
            if isinstance(cached_data, dict) and 'manifest' in cached_data:
                m = cached_data['manifest']
                cached_users_count = m.get('actual_num_users')
                if max_users is not None and cached_users_count != max_users:
                    print(f"  [Cache Invalidation] Requested max_users={max_users} but cache has {cached_users_count} users. Rebuilding...")
                    is_valid = False
                if m.get('data_split') != self.data_split:
                    print(f"  [Cache Invalidation] Requested split '{self.data_split}' does not match cached '{m.get('data_split')}'. Rebuilding...")
                    is_valid = False
            elif isinstance(cached_data, dict) and 'samples' in cached_data:
                # Legacy cache: verify user count from sample items
                actual_cached_users = len(set(s['user_id'] for s in cached_data['samples']))
                if max_users is not None and actual_cached_users != max_users:
                    print(f"  [Legacy Cache Invalidation] Requested max_users={max_users} but legacy cache has {actual_cached_users} users. Rebuilding...")
                    is_valid = False
            else:
                is_valid = False

            if is_valid:
                self.samples = cached_data['samples']
                actual_u = len(set(s['user_id'] for s in self.samples))
                print(f"  -> Cache verified: Loaded {len(self.samples)} sequences ({actual_u} participants).")
                return
            else:
                print(f"  -> Regenerating cache for split '{data_split}' with {max_users} users...")

        selected_user_dirs = self._build_dataset(max_users)
        
        if use_cache and len(self.samples) > 0:
            print(f"Saving {len(self.samples)} processed sequences to cache: {self.cache_file}")
            cache_payload = {
                'manifest': {
                    'data_split': self.data_split,
                    'requested_max_users': max_users,
                    'actual_num_users': len(selected_user_dirs),
                    'user_ids': [os.path.basename(d) for d in selected_user_dirs],
                    'num_sequences': len(self.samples),
                    'random_seed': 9,
                    'classes': Config.ALL_CLASSES
                },
                'samples': self.samples
            }
            torch.save(cache_payload, self.cache_file)

    def _build_dataset(self, max_users: int):
        root_dir = Config.TRAINING_DIR if self.data_split in ['train', 'val'] else Config.TESTING_DIR
        user_dirs = sorted([d for d in glob.glob(os.path.join(root_dir, "user*")) if os.path.isdir(d)])
        
        if not user_dirs:
            print(f"Warning: No user directories found in {root_dir}")
            return []
            
        # Split users: 80% train, 20% validation
        np.random.seed(9) # Seed matching paper
        shuffled_indices = np.random.permutation(len(user_dirs))
        split_idx = int(0.8 * len(user_dirs))
        
        if self.data_split == 'train':
            selected_indices = shuffled_indices[:split_idx]
        elif self.data_split == 'val':
            selected_indices = shuffled_indices[split_idx:]
        else: # test
            selected_indices = list(range(len(user_dirs)))
            
        selected_dirs = [user_dirs[i] for i in selected_indices]
        if max_users is not None and max_users > 0:
            selected_dirs = selected_dirs[:max_users]
            
        print(f"Processing {len(selected_dirs)} users for split '{self.data_split}'...")
        
        for u_idx, u_dir in enumerate(selected_dirs):
            user_data = load_user_json(u_dir)
            if user_data is None:
                continue
                
            user_id = os.path.basename(u_dir)
            
            # Select sample source
            samples_dict = user_data.get("trainingSamples", {})
            if self.data_split == 'test' and "testingSamples" in user_data:
                samples_dict = user_data.get("testingSamples", {})
                
            for s_key, s_data in samples_dict.items():
                if "emg" not in s_data:
                    continue
                    
                emg_sig = parse_emg_channels(s_data["emg"])
                gesture_name = s_data.get("gestureName", "noGesture")
                ground_truth = np.array(s_data.get("groundTruth", [0] * len(emg_sig)), dtype=np.int32)
                
                gt_indices = s_data.get("groundTruthIndex", [0, 0])
                num_points = gt_indices[1] - gt_indices[0] if len(gt_indices) == 2 else 0
                
                # Extract sliding window spectrograms
                frames, frame_labels, _ = self.preprocessor.extract_frames_from_sample(
                    emg_sig, ground_truth, num_points, gesture_name
                )
                
                if len(frames) == 0:
                    continue
                    
                # Convert to tensors
                seq_tensor = torch.tensor(np.stack(frames), dtype=torch.float32) # (T, 8, 13, 24)
                target_ids = torch.tensor([Config.CLASS_TO_IDX.get(lbl, Config.CLASS_TO_IDX['noGesture']) 
                                           for lbl in frame_labels], dtype=torch.long)
                                           
                self.samples.append({
                    'sequence': seq_tensor,
                    'targets': target_ids,
                    'true_gesture': gesture_name,
                    'user_id': user_id,
                    'sample_id': s_key
                })
                
            if (u_idx + 1) % 10 == 0 or (u_idx + 1) == len(selected_dirs):
                print(f"Processed [{u_idx + 1}/{len(selected_dirs)}] users ({len(self.samples)} sequences so far)")
        return selected_dirs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        seq = item['sequence']
        if self.augment and self.augmentor is not None:
            seq = self.augmentor.apply(seq)
        return seq, item['targets'], item['true_gesture'], item['user_id']


def sequence_collate_fn(batch):
    """
    Pads variable length sequences in the mini-batch to the maximum sequence length.
    Returns:
        padded_sequences: (Batch, MaxTimeSteps, 8, 13, 24)
        padded_targets: (Batch, MaxTimeSteps)
        lengths: (Batch,)
        true_gestures: list of str
        user_ids: list of str
    """
    sequences, targets, true_gestures, user_ids = zip(*batch)
    lengths = [seq.shape[0] for seq in sequences]
    max_len = max(lengths)
    
    B = len(sequences)
    C, F_dim, TB = sequences[0].shape[1:]
    
    padded_seqs = torch.zeros((B, max_len, C, F_dim, TB), dtype=torch.float32)
    padded_targets = torch.full((B, max_len), fill_value=Config.CLASS_TO_IDX['noGesture'], dtype=torch.long)
    
    for i, (seq, tgt) in enumerate(zip(sequences, targets)):
        padded_seqs[i, :seq.shape[0]] = seq
        padded_targets[i, :tgt.shape[0]] = tgt
        
    return padded_seqs, padded_targets, torch.tensor(lengths, dtype=torch.long), true_gestures, user_ids


def get_dataloaders(batch_size: int = Config.BATCH_SIZE, 
                    max_users: int = None, 
                    use_cache: bool = True,
                    augment: bool = Config.USE_AUGMENTATION):
    """
    Creates and returns training and validation DataLoaders with optional augmentation.
    Audits participant sets, validates strict disjointness with an assertion,
    and displays a partition verification report to the console.
    """
    train_dataset = EMGSequenceDataset(data_split='train', max_users=max_users, use_cache=use_cache, augment=augment)
    val_dataset = EMGSequenceDataset(data_split='val', max_users=max_users, use_cache=use_cache, augment=False)
    
    # Extract actual loaded participant IDs from sample payloads
    train_users = sorted(list(set(s['user_id'] for s in train_dataset.samples)))
    val_users = sorted(list(set(s['user_id'] for s in val_dataset.samples)))
    overlap = set(train_users).intersection(set(val_users))
    
    # Enforce strict participant-independent disjointness
    assert len(overlap) == 0, f"DATA LEAKAGE ERROR: Overlapping participants detected across splits: {overlap}"
    
    # Console Verification Report
    print(f"\n{'='*75}")
    print(f"  DATASET PARTITION & CACHE VERIFICATION REPORT")
    print(f"{'='*75}")
    req_desc = f"{max_users} users per split" if (max_users is not None and max_users > 0) else "All Available (244 Train / 62 Val)"
    print(f"  Requested User Limit:   {req_desc}")
    print(f"  Training Split:         {len(train_users)} participants | {len(train_dataset)} sequences")
    print(f"    - Participant IDs:    {train_users[:6]}{' ...' if len(train_users) > 6 else ''}")
    print(f"    - Cache Source:       {os.path.basename(train_dataset.cache_file)}")
    print(f"  Validation Split:       {len(val_users)} participants | {len(val_dataset)} sequences")
    print(f"    - Participant IDs:    {val_users[:6]}{' ...' if len(val_users) > 6 else ''}")
    print(f"    - Cache Source:       {os.path.basename(val_dataset.cache_file)}")
    print(f"  Participant Overlap:    {len(overlap)} (Strictly Disjoint) -> PASS")
    print(f"{'='*75}\n")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sequence_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sequence_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader
