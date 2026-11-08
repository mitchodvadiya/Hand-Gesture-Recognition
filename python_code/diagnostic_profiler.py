import os
import time
import torch
import numpy as np
from config import Config
from dataset import get_dataloaders
from models import BaseInceptionCNN, EnhancedCNN, BaseInceptionCNNLSTM, EnhancedHGRModel
from train import FocalLoss
from postprocess import HGRPostProcessor
from metrics import compute_classification_metrics

def get_process_memory_mb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

def run_diagnostic_profile(max_users: int = 1, batch_size: int = 16, num_batches: int = 3):
    print("=" * 80)
    print("          LIGHTWEIGHT COMPUTATIONAL & MEMORY DIAGNOSTIC PROFILER")
    print("=" * 80)
    device = torch.device("cpu")
    print(f"Target Device:     {device}")
    print(f"Initial Memory:    {get_process_memory_mb():.2f} MB")
    print(f"Test Subset:       {max_users} user(s), Batch Size: {batch_size}, Batches to Profile: {num_batches}")
    print("-" * 80)

    # 1. DataLoader & Augmentation Profiling
    print("\n[Phase 1] Profiling DataLoader & Augmentation...")
    t0 = time.perf_counter()
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, max_users=max_users, use_cache=True, augment=True)
    t_load_data = time.perf_counter() - t0
    print(f"  - Dataset & DataLoader init: {t_load_data:.3f} s (Memory: {get_process_memory_mb():.2f} MB)")

    # Time individual batch retrieval (includes EMGAugmentor)
    batch_fetch_times = []
    batches_data = []
    t_start_fetch = time.perf_counter()
    for b_idx, batch in enumerate(train_loader):
        t_batch = time.perf_counter() - t_start_fetch
        batch_fetch_times.append(t_batch)
        batches_data.append(batch)
        if b_idx + 1 >= num_batches:
            break
        t_start_fetch = time.perf_counter()

    avg_fetch_ms = np.mean(batch_fetch_times) * 1000.0
    print(f"  - Average Batch Retrieval Time (with Augmentation): {avg_fetch_ms:.2f} ms/batch")

    # 2. Model Execution Profiling (Forward, Backward, Optimizer)
    models_to_test = [
        ("BaseInceptionCNN", BaseInceptionCNN(num_classes=Config.NUM_CLASSES)),
        ("EnhancedCNN", EnhancedCNN(num_classes=Config.NUM_CLASSES)),
        ("BaseInceptionCNNLSTM", BaseInceptionCNNLSTM(num_classes=Config.NUM_CLASSES, hidden_dim=Config.LSTM_HIDDEN_DIM)),
        ("EnhancedHGRModel", EnhancedHGRModel(num_classes=Config.NUM_CLASSES, hidden_dim=Config.LSTM_HIDDEN_DIM))
    ]

    print("\n[Phase 2] Profiling Forward, Backward, and Optimizer across Architectures...")
    print(f"  {'Model Name':<22} | {'Params':<10} | {'Forward (ms)':<14} | {'Backward (ms)':<14} | {'Step (ms)':<10} | {'Mem (MB)':<10}")
    print("  " + "-" * 88)

    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_focal = FocalLoss(gamma=2.0)

    profile_results = {}

    for name, model in models_to_test:
        model.to(device)
        model.train()
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = criterion_focal if "Enhanced" in name else criterion_ce

        fwd_times, bwd_times, opt_times = [], [], []

        for seqs, targets, lengths, _, _ in batches_data:
            B, T = seqs.shape[:2]
            
            # Forward timing
            t_fwd_start = time.perf_counter()
            logits = model(seqs)
            t_fwd = (time.perf_counter() - t_fwd_start) * 1000.0
            fwd_times.append(t_fwd)

            # Flatten for loss
            logits_flat = logits.view(B * T, -1)
            targets_flat = targets.view(B * T)
            mask = torch.zeros(B, T, dtype=torch.bool)
            for i, length in enumerate(lengths):
                mask[i, :length] = True
            mask_flat = mask.view(B * T)
            loss = criterion(logits_flat[mask_flat], targets_flat[mask_flat])

            # Backward timing
            optimizer.zero_grad()
            t_bwd_start = time.perf_counter()
            loss.backward()
            t_bwd = (time.perf_counter() - t_bwd_start) * 1000.0
            bwd_times.append(t_bwd)

            # Optimizer step timing
            t_opt_start = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_opt = (time.perf_counter() - t_opt_start) * 1000.0
            opt_times.append(t_opt)

        avg_fwd = float(np.mean(fwd_times))
        avg_bwd = float(np.mean(bwd_times))
        avg_opt = float(np.mean(opt_times))
        cur_mem = get_process_memory_mb()

        profile_results[name] = {
            "params": param_count,
            "forward_ms": round(avg_fwd, 2),
            "backward_ms": round(avg_bwd, 2),
            "opt_ms": round(avg_opt, 2),
            "batch_total_ms": round(avg_fwd + avg_bwd + avg_opt, 2),
            "memory_mb": round(cur_mem, 2)
        }

        print(f"  {name:<22} | {param_count:>10,} | {avg_fwd:>12.2f} ms | {avg_bwd:>12.2f} ms | {avg_opt:>8.2f} ms | {cur_mem:>8.1f} MB")

    # 3. Checkpoint Saving Profiling
    print("\n[Phase 3] Profiling Checkpoint Serialization...")
    dummy_ckpt = {
        'epoch': 1,
        'model_state_dict': models_to_test[0][1].state_dict(),
        'val_metrics': {'accuracy': 70.0}
    }
    dummy_path = os.path.join(Config.CHECKPOINT_DIR, "diagnostic_dummy_ckpt.pth")
    t_save_start = time.perf_counter()
    torch.save(dummy_ckpt, dummy_path)
    t_save = (time.perf_counter() - t_save_start) * 1000.0
    file_kb = os.path.getsize(dummy_path) / 1024.0
    if os.path.exists(dummy_path):
        os.remove(dummy_path)
    print(f"  - Checkpoint Serialization Time: {t_save:.2f} ms (File Size: {file_kb:.1f} KB)")

    # 4. Projected Runtime Estimates for 5-User Cohort (750 sequences / 12 batches)
    print("\n[Phase 4] Projected Realistic Runtimes on this CPU (5-User Cohort, 12 Batches/Epoch):")
    for name, stats in profile_results.items():
        batch_total_s = (stats["batch_total_ms"] + avg_fetch_ms) / 1000.0
        # 12 batches per epoch
        epoch_s = batch_total_s * 12.0
        epochs_15_min = (epoch_s * 15.0) / 60.0
        print(f"  - {name:<22}: ~{epoch_s:.1f} s/epoch | 15 Epochs: ~{epochs_15_min:.1f} minutes")

    print("\n" + "=" * 80)
    print("                     DIAGNOSTIC PROFILING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_diagnostic_profile(max_users=1, batch_size=16, num_batches=3)
