import os
import torch
import numpy as np
from config import Config
from models import BaseInceptionCNN
from postprocess import HGRPostProcessor
from metrics import compute_classification_metrics

def diagnose():
    print("=" * 75)
    print("      DIAGNOSING CHECKPOINT ACCURACY & OPEN VS WAVEOUT CONFUSION")
    print("=" * 75)

    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model_cnn_base_crossentropy.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("\n1. CHECKPOINT STORED KEYS:")
    for k in ckpt.keys():
        if k != "model_state_dict" and k != "optimizer_state_dict":
            print(f"   - {k}: {ckpt[k]}")

    val_metrics = ckpt.get("val_metrics", {})
    print(f"\n2. CHECKPOINT STORED METRICS:")
    print(f"   Accuracy: {val_metrics.get('accuracy')}%")
    print(f"   Macro F1: {val_metrics.get('macro_f1')}%")
    print(f"   Total Samples: {val_metrics.get('total_samples')}")
    print(f"   Per-class support: {[v['support'] for v in val_metrics.get('per_class', {}).values()]}")

    # Load 5-user validation cache
    cache_u5_path = os.path.join(Config.CACHE_DIR, "val_seq_cache_u5.pt")
    if os.path.exists(cache_u5_path):
        cache_u5 = torch.load(cache_u5_path, weights_only=False)
        samples_u5 = cache_u5['samples']
        print(f"\n3. EVALUATING ON val_seq_cache_u5.pt ({len(samples_u5)} sequences)...")
        
        model = BaseInceptionCNN(num_classes=Config.NUM_CLASSES)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        postprocessor = HGRPostProcessor(mode=Config.POSTPROCESS_MODE)

        y_true, y_raw, y_post = [], [], []
        with torch.no_grad():
            for s in samples_u5:
                seq = s['sequence'].unsqueeze(0) # (1, T, 8, 13, 24)
                logits = model(seq) # (1, T, 6)
                preds = torch.argmax(logits, dim=-1).squeeze(0).tolist()
                pred_names = [Config.IDX_TO_CLASS[idx] for idx in preds]
                
                raw_pred = postprocessor.process(pred_names, apply_filter=False)
                post_pred = postprocessor.process(pred_names, apply_filter=True)
                
                y_true.append(s['true_gesture'])
                y_raw.append(raw_pred)
                y_post.append(post_pred)

        m_raw = compute_classification_metrics(y_true, y_raw)
        m_post = compute_classification_metrics(y_true, y_post)
        print(f"   -> Raw Accuracy on u5:        {m_raw['accuracy']:.2f}%")
        print(f"   -> Post-processed Accuracy on u5: {m_post['accuracy']:.2f}%")
        print(f"   -> Macro F1 on u5:            {m_post['macro_f1']:.2f}%")
        print(f"   -> Matches Checkpoint Accuracy (62.4%) exactly? {'YES' if abs(m_post['accuracy'] - ckpt.get('val_post_acc', 0)) < 0.01 else 'NO'}")

    # Participant overlap check
    print("\n4. PARTICIPANT ANALYSIS:")
    train_u5_path = os.path.join(Config.CACHE_DIR, "train_seq_cache_u5.pt")
    if os.path.exists(train_u5_path):
        train_u5 = torch.load(train_u5_path, weights_only=False)
        train_users = sorted(list(set(s['user_id'] for s in train_u5['samples'])))
        print(f"   Train Participants in u5 ({len(train_users)}): {train_users}")
        
    val_all_path = os.path.join(Config.CACHE_DIR, "val_seq_cache_all.pt")
    if os.path.exists(val_all_path):
        val_all = torch.load(val_all_path, weights_only=False)
        val_users = sorted(list(set(s['user_id'] for s in val_all['samples'])))
        print(f"   Val Participants in val_all ({len(val_users)}): {val_users[:8]} ... {val_users[-4:]}")
        if os.path.exists(train_u5_path):
            overlap = set(train_users).intersection(set(val_users))
            print(f"   Overlap between train_u5 and val_all: {overlap} (count: {len(overlap)})")

    # Analyze open vs waveOut features
    print("\n5. OPEN VS WAVEOUT INSPECTION:")
    # Check sample spectrograms and targets
    if os.path.exists(cache_u5_path):
        open_samples = [s for s in samples_u5 if s['true_gesture'] == 'open']
        waveout_samples = [s for s in samples_u5 if s['true_gesture'] == 'waveOut']
        print(f"   Open sequences in u5: {len(open_samples)}")
        print(f"   WaveOut sequences in u5: {len(waveout_samples)}")
        if open_samples:
            s0 = open_samples[0]
            print(f"   Sample open: user={s0['user_id']}, sample_id={s0['sample_id']}, targets={s0['targets'].tolist()[:15]}...")
        if waveout_samples:
            w0 = waveout_samples[0]
            print(f"   Sample waveOut: user={w0['user_id']}, sample_id={w0['sample_id']}, targets={w0['targets'].tolist()[:15]}...")

if __name__ == "__main__":
    diagnose()
