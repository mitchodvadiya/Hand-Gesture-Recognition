import os
import torch
from dataset import get_dataloaders, EMGSequenceDataset
from config import Config

def test_pipeline_integrity():
    print("=" * 75)
    print("      RUNNING PIPELINE INTEGRITY & CACHE VALIDATION TESTS")
    print("=" * 75)
    
    # -------------------------------------------------------------
    # Test 1: 1-User Configuration Loads Exactly 1 User
    # -------------------------------------------------------------
    print("\n[Test 1] Testing 1-User DataLoader Loading...")
    train_loader_1, val_loader_1 = get_dataloaders(batch_size=16, max_users=1, use_cache=True, augment=False)
    
    train_users_1 = set(s['user_id'] for s in train_loader_1.dataset.samples)
    val_users_1 = set(s['user_id'] for s in val_loader_1.dataset.samples)
    
    print(f"  Loaded Train Users ({len(train_users_1)}): {train_users_1}")
    print(f"  Loaded Val Users ({len(val_users_1)}):   {val_users_1}")
    
    assert len(train_users_1) == 1, f"Expected 1 train user, got {len(train_users_1)}"
    assert len(val_users_1) == 1, f"Expected 1 val user, got {len(val_users_1)}"
    assert train_users_1.isdisjoint(val_users_1), "Train and Val users must be disjoint!"
    print("  -> TEST 1 PASSED: 1-user split correctly loaded 1 participant per split with 0 overlap.")
    
    # -------------------------------------------------------------
    # Test 2: Incompatible Cache Rejection
    # -------------------------------------------------------------
    print("\n[Test 2] Testing Stale / Incompatible Cache Rejection...")
    # Verify that requesting max_users=1 cannot silently load the 5-user cache
    ds_u1 = EMGSequenceDataset(data_split='train', max_users=1, use_cache=True)
    assert len(set(s['user_id'] for s in ds_u1.samples)) == 1, "Cache validation failed to enforce max_users=1"
    assert os.path.basename(ds_u1.cache_file) == "train_seq_cache_u1.pt", "Wrong cache filename generated"
    print("  -> TEST 2 PASSED: 1-user configuration strictly targets '_u1.pt' cache.")
    
    # -------------------------------------------------------------
    # Test 3: 5-User Configuration Loads Exactly 5 Users
    # -------------------------------------------------------------
    print("\n[Test 3] Testing 5-User DataLoader Loading...")
    train_loader_5, val_loader_5 = get_dataloaders(batch_size=32, max_users=5, use_cache=True, augment=False)
    
    train_users_5 = set(s['user_id'] for s in train_loader_5.dataset.samples)
    val_users_5 = set(s['user_id'] for s in val_loader_5.dataset.samples)
    
    print(f"  Loaded Train Users ({len(train_users_5)}): {sorted(list(train_users_5))}")
    print(f"  Loaded Val Users ({len(val_users_5)}):   {sorted(list(val_users_5))}")
    
    assert len(train_users_5) == 5, f"Expected 5 train users, got {len(train_users_5)}"
    assert len(val_users_5) == 5, f"Expected 5 val users, got {len(val_users_5)}"
    assert train_users_5.isdisjoint(val_users_5), "Train and Val users must be disjoint!"
    print("  -> TEST 3 PASSED: 5-user split correctly loaded 5 distinct participants per split with 0 overlap.")
    
    # -------------------------------------------------------------
    # Test 4: Verify Original Result Files are Untouched
    # -------------------------------------------------------------
    print("\n[Test 4] Verifying Historical Artifacts Preservation...")
    expected_files = [
        os.path.join(Config.RESULTS_DIR, "VALIDATION_REPORT.md"),
        os.path.join(Config.RESULTS_DIR, "cnn_base_metrics.csv"),
        os.path.join(Config.RESULTS_DIR, "cnn_base_confusion_matrix.csv")
    ]
    for ef in expected_files:
        assert os.path.exists(ef), f"Historical artifact missing: {ef}"
        assert os.path.getsize(ef) > 0, f"Historical artifact corrupted: {ef}"
        print(f"  - Preserved: {os.path.basename(ef)} ({os.path.getsize(ef)} bytes)")
    print("  -> TEST 4 PASSED: All historical result artifacts intact.")
    
    print("\n" + "=" * 75)
    print("       ALL PIPELINE INTEGRITY TESTS SUCCESSFULLY PASSED (4/4)")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    test_pipeline_integrity()
