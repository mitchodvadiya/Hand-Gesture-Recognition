import os
import glob
import torch
import numpy as np
from collections import Counter
from config import Config

def analyze_all_caches():
    cache_dir = Config.CACHE_DIR
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    
    print("=" * 80)
    print("        COMPREHENSIVE CACHE AUDIT & SEQUENCE INTEGRITY REPORT")
    print("=" * 80)
    
    cache_manifests = {}
    
    for cf in cache_files:
        basename = os.path.basename(cf)
        size_mb = os.path.getsize(cf) / (1024 * 1024)
        data = torch.load(cf, weights_only=False)
        samples = data.get('samples', [])
        
        users = sorted(list(set(s['user_id'] for s in samples)))
        gestures = [s['true_gesture'] for s in samples]
        g_counts = Counter(gestures)
        
        # Frame-level targets inspection
        all_targets = []
        corrupted_sequences = 0
        target_shapes = set()
        
        for s in samples:
            tgts = s.get('targets')
            seq = s.get('sequence', s.get('spectrograms'))
            if tgts is None or seq is None:
                corrupted_sequences += 1
                continue
            all_targets.extend(tgts.tolist())
            target_shapes.add(tuple(tgts.shape))
            
        unique_target_ids = sorted(list(set(all_targets)))
        frame_target_counts = Counter(all_targets)
        total_frames = len(all_targets)
        
        cache_manifests[basename] = {
            "size_mb": round(size_mb, 2),
            "num_sequences": len(samples),
            "num_participants": len(users),
            "participants": users,
            "gesture_counts": dict(g_counts),
            "unique_target_ids": unique_target_ids,
            "corrupted_sequences": corrupted_sequences,
            "frame_target_counts": {Config.IDX_TO_CLASS.get(k, k): v for k, v in frame_target_counts.items()},
            "frame_proportions": {Config.IDX_TO_CLASS.get(k, k): round(100.0 * v / max(1, total_frames), 2) for k, v in frame_target_counts.items()}
        }
        
        print(f"\nCache File: {basename} ({size_mb:.2f} MB)")
        print(f"  - Total Sequences:      {len(samples)}")
        print(f"  - Unique Participants:  {len(users)} -> {users[:5]}{' ...' if len(users) > 5 else ''}")
        print(f"  - Gesture Trial Counts: {dict(g_counts)}")
        print(f"  - Target Shapes Found:  {target_shapes}")
        print(f"  - Unique Target IDs:    {unique_target_ids} (Config range: 0..{Config.NUM_CLASSES-1})")
        print(f"  - Corrupted Sequences:  {corrupted_sequences}")
        print(f"  - Frame-Level Distribution (Total: {total_frames:,} frames):")
        for cls_name, pct in cache_manifests[basename]["frame_proportions"].items():
            cnt = cache_manifests[basename]["frame_target_counts"][cls_name]
            print(f"      * {cls_name:<12}: {cnt:>7,} frames ({pct:>5.2f}%)")
            
    print("\n" + "=" * 80)
    print("        PARTICIPANT OVERLAP & DISJOINTNESS CROSS-CHECK")
    print("=" * 80)
    
    # 1. Compare train_seq_cache_u1 vs val_seq_cache_u1 (if both exist) or val_seq_cache_u5
    train_u1 = set(cache_manifests.get("train_seq_cache_u1.pt", {}).get("participants", []))
    val_u5 = set(cache_manifests.get("val_seq_cache_u5.pt", {}).get("participants", []))
    train_u5 = set(cache_manifests.get("train_seq_cache_u5.pt", {}).get("participants", []))
    val_all = set(cache_manifests.get("val_seq_cache_all.pt", {}).get("participants", []))
    train_old = set(cache_manifests.get("train_seq_cache.pt", {}).get("participants", []))
    val_old = set(cache_manifests.get("val_seq_cache.pt", {}).get("participants", []))
    
    print(f"Train (u5) vs Val (u5) Overlap: {len(train_u5.intersection(val_u5))} -> {train_u5.intersection(val_u5)}")
    print(f"Train (u1) vs Val (u5) Overlap: {len(train_u1.intersection(val_u5))} -> {train_u1.intersection(val_u5)}")
    print(f"Train (old) vs Val (old) Overlap: {len(train_old.intersection(val_old))} -> {train_old.intersection(val_old)}")
    
    # Theoretical split check from user directories
    user_dirs = sorted([d for d in glob.glob(os.path.join(Config.TRAINING_DIR, "user*")) if os.path.isdir(d)])
    np.random.seed(9)
    shuffled_idx = np.random.permutation(len(user_dirs))
    split_idx = int(0.8 * len(user_dirs))
    
    theory_train_users = [os.path.basename(user_dirs[i]) for i in shuffled_idx[:split_idx]]
    theory_val_users = [os.path.basename(user_dirs[i]) for i in shuffled_idx[split_idx:]]
    
    print(f"\nTheoretical Split (from trainingJSON, 306 users, seed=9, 80/20):")
    print(f"  - Total Users: {len(user_dirs)}")
    print(f"  - Theoretical Train Pool: {len(theory_train_users)} users")
    print(f"  - Theoretical Val Pool:   {len(theory_val_users)} users")
    print(f"  - Set Overlap:            {len(set(theory_train_users).intersection(set(theory_val_users)))} users")
    print(f"  - First 5 Train Users:    {theory_train_users[:5]}")
    print(f"  - First 5 Val Users:      {theory_val_users[:5]}")
    
    print("=" * 80)

if __name__ == "__main__":
    analyze_all_caches()
