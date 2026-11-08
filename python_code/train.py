import os
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from config import Config
from dataset import get_dataloaders
from models import BaseInceptionCNN, EnhancedCNN, BaseInceptionCNNLSTM, EnhancedHGRModel
from postprocess import HGRPostProcessor
from metrics import compute_classification_metrics, format_confusion_matrix, format_metrics_table

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional Label Smoothing:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy background/resting ('noGesture') frames and forces
    the model to focus on hard, transient gesture contractions.
    """
    def __init__(self, gamma: float = Config.FOCAL_GAMMA, alpha: torch.Tensor = None, 
                 label_smoothing: float = Config.LABEL_SMOOTHING, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1) # (N, C)
        probs = torch.exp(log_probs)              # (N, C)

        # Label smoothing target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, fill_value=self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - probs, self.gamma)

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            focal_weight = focal_weight * alpha.unsqueeze(0)

        loss = -smooth_targets * focal_weight * log_probs
        loss = loss.sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def parse_args():
    parser = argparse.ArgumentParser(description="Train Hand Gesture Recognition Model (EMG-EPN-612)")
    parser.add_argument("--model", type=str, default="enhanced",
                        choices=["cnn_base", "cnn_enhanced", "base", "enhanced", "cnn_lstm_base", "cnn_lstm_enhanced"],
                        help="Model architecture: 'cnn_base', 'cnn_enhanced', 'base' (CNN-LSTM), or 'enhanced' (CNN-LSTM + Attention)")
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--max_users", type=int, default=None, help="Subset of users for fast debugging (e.g. 10 or 20), None for all")
    parser.add_argument("--no_cache", action="store_true", help="Force recomputation of dataset cache")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--loss", type=str, default="focal" if Config.USE_FOCAL_LOSS else "crossentropy",
                        choices=["focal", "crossentropy"], help="Loss function: 'focal' (Class-Weighted Focal Loss) or 'crossentropy'")
    parser.add_argument("--focal_gamma", type=float, default=Config.FOCAL_GAMMA,
                        help="Focal Loss gamma parameter")
    parser.add_argument("--label_smoothing", type=float, default=Config.LABEL_SMOOTHING,
                        help="Label smoothing factor (e.g. 0.05)")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable data augmentation (scaling & noise)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional tag for experiment naming and logging")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct_frames = 0
    total_frames = 0
    
    for batch_idx, (seqs, targets, lengths, _, _) in enumerate(loader):
        seqs = seqs.to(device)       # (B, T, 8, 13, 24)
        targets = targets.to(device) # (B, T)
        
        optimizer.zero_grad()
        logits = model(seqs)        # (B, T, NumClasses)
        
        # Flatten across time steps, ignoring padding
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        targets_flat = targets.view(B * T)
        
        # Create mask for valid sequence lengths
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            mask[i, :length] = True
        mask_flat = mask.view(B * T)
        
        loss = criterion(logits_flat[mask_flat], targets_flat[mask_flat])
        loss.backward()
        
        # Gradient clipping matching MATLAB
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * targets_flat[mask_flat].size(0)
        
        preds = torch.argmax(logits_flat[mask_flat], dim=-1)
        correct_frames += (preds == targets_flat[mask_flat]).sum().item()
        total_frames += targets_flat[mask_flat].size(0)
        
    avg_loss = total_loss / max(1, total_frames)
    frame_acc = 100.0 * correct_frames / max(1, total_frames)
    return avg_loss, frame_acc


def evaluate_model(model, loader, criterion, device, postprocessor):
    model.eval()
    total_loss = 0.0
    correct_frames = 0
    total_frames = 0
    
    y_true_seq = []
    y_pred_raw_seq = []
    y_pred_post_seq = []
    
    with torch.no_grad():
        for seqs, targets, lengths, true_gestures, _ in loader:
            seqs = seqs.to(device)
            targets = targets.to(device)
            
            logits = model(seqs)
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            
            mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            for i, length in enumerate(lengths):
                mask[i, :length] = True
            mask_flat = mask.view(B * T)
            
            loss = criterion(logits_flat[mask_flat], targets_flat[mask_flat])
            total_loss += loss.item() * targets_flat[mask_flat].size(0)
            
            preds_flat = torch.argmax(logits_flat[mask_flat], dim=-1)
            correct_frames += (preds_flat == targets_flat[mask_flat]).sum().item()
            total_frames += targets_flat[mask_flat].size(0)
            
            # Sequence classification with & without post-processing
            preds = torch.argmax(logits, dim=-1) # (B, T)
            for i in range(B):
                valid_len = lengths[i].item()
                pred_indices = preds[i, :valid_len].cpu().numpy().tolist()
                pred_label_names = [Config.IDX_TO_CLASS[idx] for idx in pred_indices]
                actual_gesture = true_gestures[i]
                
                raw_pred = postprocessor.process(pred_label_names, apply_filter=False)
                post_pred = postprocessor.process(pred_label_names, apply_filter=True)
                
                y_true_seq.append(actual_gesture)
                y_pred_raw_seq.append(raw_pred)
                y_pred_post_seq.append(post_pred)
                
    avg_loss = total_loss / max(1, total_frames)
    frame_acc = 100.0 * correct_frames / max(1, total_frames)
    
    # Calculate full metrics using validation sequence predictions
    post_metrics = compute_classification_metrics(y_true_seq, y_pred_post_seq)
    raw_metrics = compute_classification_metrics(y_true_seq, y_pred_raw_seq)
    
    return avg_loss, frame_acc, raw_metrics, post_metrics


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else Config.DEVICE
    timestamp = int(time.time())
    exp_tag = f"_{args.tag}" if args.tag else ""
    exp_name = f"exp_{args.model}_{args.loss}{exp_tag}_{timestamp}"
    
    print(f"==================================================")
    print(f"        HGR EXPERIMENT: {exp_name}")
    print(f"==================================================")
    print(f"Model Architecture: {args.model.upper()}")
    print(f"Target Device:      {device}")
    print(f"Epochs:             {args.epochs}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Learning Rate:      {args.lr}")
    print(f"Data Split:         Participant-Independent (80% Train, 20% Val, Seed=9)")
    print(f"Data Augmentation:  {'ENABLED (Amplitude Scale, Sensor Noise)' if not args.no_augment else 'DISABLED'}")
    print(f"Loss Function:      {args.loss.upper()} (gamma={args.focal_gamma}, smoothing={args.label_smoothing})")
    print(f"==================================================")
    
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # 1. Load Data
    print("\n--- Loading Dataset ---")
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        max_users=args.max_users,
        use_cache=not args.no_cache,
        augment=not args.no_augment
    )
    
    # 2. Verify Training Data Class Distribution & Compute Class Weights strictly on Training Data
    print("\n--- Analyzing Training Data Class Distribution ---")
    train_targets = []
    for s in train_loader.dataset.samples:
        train_targets.extend(s['targets'].tolist())
        
    class_counts = torch.bincount(torch.tensor(train_targets, dtype=torch.long), minlength=Config.NUM_CLASSES).float()
    total_train_frames = class_counts.sum().item()
    
    print(f"Total Training Frames: {int(total_train_frames):,}")
    for idx, cls_name in Config.IDX_TO_CLASS.items():
        cnt = int(class_counts[idx].item())
        pct = (100.0 * cnt / total_train_frames) if total_train_frames > 0 else 0.0
        print(f"  [{idx}] {cls_name:<12}: {cnt:>8,} frames ({pct:5.2f}%)")
        
    # Calculate inverse-frequency balanced class weights strictly on training set
    raw_weights = total_train_frames / (Config.NUM_CLASSES * torch.clamp(class_counts, min=1.0))
    class_weights = raw_weights / raw_weights.mean()
    class_weight_dict = {Config.IDX_TO_CLASS[i]: round(class_weights[i].item(), 3) for i in range(Config.NUM_CLASSES)}
    print(f"Calculated Training-Set Class Weights:\n  {class_weight_dict}")
    
    # 3. Instantiate Network
    print("\n--- Initializing Network ---")
    if args.model in ["cnn_base", "base_cnn"]:
        model = BaseInceptionCNN(num_classes=Config.NUM_CLASSES)
    elif args.model in ["cnn_enhanced", "enhanced_cnn"]:
        model = EnhancedCNN(num_classes=Config.NUM_CLASSES)
    elif args.model in ["base", "cnn_lstm_base", "base_lstm"]:
        model = BaseInceptionCNNLSTM(num_classes=Config.NUM_CLASSES, hidden_dim=Config.LSTM_HIDDEN_DIM)
    else:
        model = EnhancedHGRModel(num_classes=Config.NUM_CLASSES, hidden_dim=Config.LSTM_HIDDEN_DIM)
        
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Learnable Parameters: {total_params:,}")
    
    # 4. Loss Function & Optimizer
    if args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=class_weights.to(device), label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    postprocessor = HGRPostProcessor(mode=Config.POSTPROCESS_MODE)
    
    best_val_acc = 0.0
    best_metrics = None
    best_checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"best_model_{args.model}_{args.loss}{exp_tag}.pth")
    epoch_history = []
    
    # 5. Training Loop
    print("\n--- Beginning Training Pipeline ---")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, raw_metrics, post_metrics = evaluate_model(model, val_loader, criterion, device, postprocessor)
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        val_post_seq = post_metrics["accuracy"]
        val_macro_f1 = post_metrics["macro_f1"]
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | "
              f"Val Loss: {val_loss:.4f} | "
              f"Post Acc: {val_post_seq:.2f}% | "
              f"Macro F1: {val_macro_f1:.2f}%")
              
        epoch_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_frame_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_raw_accuracy": raw_metrics["accuracy"],
            "val_post_accuracy": post_metrics["accuracy"],
            "val_macro_precision": post_metrics["macro_precision"],
            "val_macro_recall": post_metrics["macro_recall"],
            "val_macro_f1": post_metrics["macro_f1"],
            "epoch_duration_sec": round(epoch_time, 2)
        })
        
        # Checkpoint best model based on validation post-processed recognition accuracy
        if val_post_seq > best_val_acc:
            best_val_acc = val_post_seq
            best_metrics = post_metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_post_acc': val_post_seq,
                'val_metrics': post_metrics,
                'model_type': args.model,
                'loss_type': args.loss,
                'class_weights': class_weights.cpu().numpy().tolist(),
                'config': {
                    'num_classes': Config.NUM_CLASSES,
                    'hidden_dim': Config.LSTM_HIDDEN_DIM
                }
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path} (Val Acc: {val_post_seq:.2f}%, Macro F1: {val_macro_f1:.2f}%)")

    total_duration = time.time() - start_time
    
    # 6. Save Complete Experiment Configuration & Metrics
    exp_record = {
        "experiment_name": exp_name,
        "timestamp": timestamp,
        "config": {
            "model": args.model,
            "loss": args.loss,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_users": args.max_users,
            "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
            "label_smoothing": args.label_smoothing,
            "augmentation_enabled": not args.no_augment,
            "class_weights": class_weight_dict
        },
        "training_time_min": round(total_duration / 60.0, 2),
        "best_checkpoint": best_checkpoint_path,
        "best_validation_metrics": best_metrics,
        "epoch_history": epoch_history
    }
    
    exp_file = os.path.join(Config.RESULTS_DIR, f"{exp_name}.json")
    with open(exp_file, 'w', encoding='utf-8') as f:
        json.dump(exp_record, f, indent=4)
        
    latest_file = os.path.join(Config.RESULTS_DIR, f"latest_{args.model}_{args.loss}.json")
    with open(latest_file, 'w', encoding='utf-8') as f:
        json.dump(exp_record, f, indent=4)
        
    print("\n" + "="*55)
    print("              EXPERIMENT RESULTS SUMMARY")
    print("="*55)
    print(f"Experiment File:  {exp_file}")
    print(f"Best Checkpoint:  {best_checkpoint_path}")
    print(f"Training Time:    {total_duration/60:.2f} minutes")
    print(f"Best Post-Processed Accuracy: {best_val_acc:.2f}%\n")
    
    if best_metrics:
        print(format_metrics_table(best_metrics))
        print("\n" + format_confusion_matrix(best_metrics["confusion_matrix"]))
    print("="*55)

if __name__ == "__main__":
    main()

