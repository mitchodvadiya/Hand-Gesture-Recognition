import os
import sys
import torch
from collections import Counter

def inspect_cache(cache_filename):
    cache_path = os.path.join("data_cache", cache_filename)
    if not os.path.exists(cache_path):
        print(f"Error: {cache_path} not found!")
        return

    print(f"\n{'='*65}")
    print(f"   INSPECTING CACHE FILE: {cache_filename}")
    print(f"   File size: {os.path.getsize(cache_path) / (1024*1024):.2f} MB")
    print(f"{'='*65}")

    data = torch.load(cache_path, weights_only=False)
    samples = data.get('samples', [])
    print(f"Total Sequences Stored: {len(samples)}")

    if len(samples) == 0:
        print("Cache is empty.")
        return

    # Participant breakdown
    users = sorted(list(set(s['user_id'] for s in samples)))
    print(f"Participants ({len(users)}): {users}")

    # Gesture trial breakdown
    gestures = [s['true_gesture'] for s in samples]
    gesture_counts = Counter(gestures)
    print("\nSequence Count Per Gesture:")
    for g, count in gesture_counts.items():
        print(f"  - {g:<12}: {count:>4} sequences")

    # Inspect first sequence
    s0 = samples[0]
    specs = s0.get('sequence', s0.get('spectrograms')) # (T, Channels=8, Freq=13, TimeSteps=24)
    targets = s0['targets']     # (T,)
    print("\nSample Data Dimensions:")
    print(f"  - Available Dictionary Keys: {list(s0.keys())}")
    print(f"  - Spectrogram Tensor Shape:  {tuple(specs.shape)} (TimeFrames, Channels, FreqBins, TimeBins)")
    print(f"  - Frame Targets Shape:      {tuple(targets.shape)}")
    print(f"  - Target Label Values:      {targets.unique().tolist()}")
    print(f"  - First Sample User:        {s0['user_id']}")
    print(f"  - First Sample Gesture:     {s0['true_gesture']}")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "train_seq_cache_u1.pt"
    inspect_cache(target)
