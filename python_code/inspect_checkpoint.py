import os
import sys
import torch

def inspect_checkpoint(checkpoint_path: str):
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)

    print("=" * 75)
    print("                 INSPECTING PYTORCH MODEL CHECKPOINT")
    print("=" * 75)
    print(f"File Path: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Checkpoint file not found: {checkpoint_path}")
        print("=" * 75)
        return

    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"File Size: {file_size_mb:.2f} MB")

    try:
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"\n[ERROR] Failed to load checkpoint: {e}")
        return

    if not isinstance(data, dict):
        print(f"Loaded object type: {type(data)}")
        return

    print("\n--- Checkpoint Metadata ---")
    print(f"  Model Architecture : {data.get('model_type', 'N/A').upper()}")
    print(f"  Loss Function      : {data.get('loss_type', 'N/A').upper()}")
    print(f"  Saved at Epoch     : {data.get('epoch', 'N/A')}")
    print(f"  Best Val Accuracy  : {data.get('val_post_acc', 'N/A')}%")

    val_metrics = data.get("val_metrics", {})
    if val_metrics:
        print(f"  Macro Precision    : {val_metrics.get('macro_precision', 'N/A')}%")
        print(f"  Macro Recall       : {val_metrics.get('macro_recall', 'N/A')}%")
        print(f"  Macro F1-Score     : {val_metrics.get('macro_f1', 'N/A')}%")

        if "per_class" in val_metrics:
            print("\n--- Validation Per-Class Performance ---")
            header = f"  {'Gesture':<12} | {'Precision (%)':<14} | {'Recall (%)':<12} | {'F1-Score (%)':<13} | {'Support':<8}"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for cls_name, vals in val_metrics["per_class"].items():
                print(f"  {cls_name:<12} | {vals['precision']:<14.2f} | {vals['recall']:<12.2f} | {vals['f1_score']:<13.2f} | {vals['support']:<8}")

    # Inspect model weights
    state_dict = data.get("model_state_dict", {})
    if state_dict:
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"\n--- Network Structure & Weights ({len(state_dict)} tensors, {total_params:,} parameters) ---")
        for idx, (name, param) in enumerate(list(state_dict.items())[:10]):
            print(f"  [{idx+1:02d}] {name:<45} : shape {list(param.shape)}")
        if len(state_dict) > 10:
            print(f"  ... and {len(state_dict) - 10} more layer tensors.")

    print("\n" + "=" * 75)
    print("                     CHECKPOINT INSPECTION COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    default_ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model_cnn_base_crossentropy.pth")
    target = sys.argv[1] if len(sys.argv) > 1 else default_ckpt
    inspect_checkpoint(target)
