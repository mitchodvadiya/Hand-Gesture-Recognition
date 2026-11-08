import os
import glob
import json
import time
import argparse
import torch
import numpy as np

from config import Config
from preprocessing import EMGPreprocessor
from models import BaseInceptionCNN, EnhancedCNN, BaseInceptionCNNLSTM, EnhancedHGRModel
from postprocess import HGRPostProcessor
from dataset import load_user_json, parse_emg_channels

def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses.json on Testing Set for EMG-EPN-612 Evaluator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--output", type=str, default=None, help="Output responses.json file path")
    parser.add_argument("--max_users", type=int, default=None, help="Limit number of test users (for quick verification)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = Config.DEVICE
    print(f"=== Generating Testing Benchmark Predictions ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_type = checkpoint.get("model_type", "enhanced")
    
    if model_type in ["cnn_base", "base_cnn"]:
        model = BaseInceptionCNN(num_classes=Config.NUM_CLASSES)
    elif model_type in ["cnn_enhanced", "enhanced_cnn"]:
        model = EnhancedCNN(num_classes=Config.NUM_CLASSES)
    elif model_type in ["base", "cnn_lstm_base", "base_lstm"]:
        model = BaseInceptionCNNLSTM(num_classes=Config.NUM_CLASSES, hidden_dim=Config.LSTM_HIDDEN_DIM)
    else:
        model = EnhancedHGRModel(num_classes=Config.NUM_CLASSES, hidden_dim=Config.LSTM_HIDDEN_DIM)

        
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    preprocessor = EMGPreprocessor()
    postprocessor = HGRPostProcessor(mode=Config.POSTPROCESS_MODE)
    
    test_dirs = sorted([d for d in glob.glob(os.path.join(Config.TESTING_DIR, "user*")) if os.path.isdir(d)])
    if args.max_users is not None and args.max_users > 0:
        test_dirs = test_dirs[:args.max_users]
        
    print(f"Found {len(test_dirs)} test users in {Config.TESTING_DIR}")
    
    responses = {"testing": {}}
    t_start = time.time()
    
    for u_idx, u_dir in enumerate(test_dirs):
        user_name = os.path.basename(u_dir)
        user_data = load_user_json(u_dir)
        if user_data is None:
            continue
            
        testing_samples = user_data.get("testingSamples", {})
        
        user_classes = []
        user_vector_labels = []
        user_vector_timepoints = []
        user_vector_proctimes = []
        
        # Sort sample keys numerically or alphabetically (idx_1 to idx_150)
        sorted_keys = sorted(testing_samples.keys(), key=lambda k: int(k.split('_')[-1]) if '_' in k else 0)
        
        for s_key in sorted_keys:
            s_data = testing_samples[s_key]
            emg_sig = parse_emg_channels(s_data["emg"])
            
            t_sample_start = time.time()
            preprocessed = preprocessor.preprocess_signal(emg_sig)
            
            # Extract sliding frames (300 window, step 30)
            sig_len = len(preprocessed)
            num_windows = (sig_len - Config.FRAME_WINDOW) // Config.WINDOW_STEP + 1
            
            frames = []
            timepoints = []
            for w in range(num_windows):
                start = w * Config.WINDOW_STEP
                end = start + Config.FRAME_WINDOW
                timestamp = start + Config.FRAME_WINDOW // 2
                spec = preprocessor.compute_spectrogram(preprocessed[start:end, :])
                frames.append(spec)
                timepoints.append(int(timestamp))
                
            if len(frames) == 0:
                user_classes.append("noGesture")
                user_vector_labels.append([])
                user_vector_timepoints.append([])
                user_vector_proctimes.append([0.0])
                continue
                
            # Run inference
            seq_tensor = torch.tensor(np.stack(frames), dtype=torch.float32).unsqueeze(0).to(device) # (1, T, 8, 13, 24)
            with torch.no_grad():
                logits = model(seq_tensor) # (1, T, NumClasses)
                preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy().tolist()
                
            pred_labels = [Config.IDX_TO_CLASS[idx] for idx in preds]
            
            # Post-processing
            final_gesture = postprocessor.process(pred_labels, apply_filter=True)
            elapsed_time = (time.time() - t_sample_start) / max(1, len(frames))
            
            user_classes.append(final_gesture)
            user_vector_labels.append(pred_labels)
            user_vector_timepoints.append(timepoints)
            user_vector_proctimes.append([round(elapsed_time, 4)] * len(frames))
            
        responses["testing"][user_name] = {
            "class": user_classes,
            "vectorOfLabels": user_vector_labels,
            "vectorOfTimePoints": user_vector_timepoints,
            "vectorOfProcessingTime": user_vector_proctimes
        }
        
        if (u_idx + 1) % 5 == 0 or (u_idx + 1) == len(test_dirs):
            print(f"Processed [{u_idx + 1}/{len(test_dirs)}] test users ({time.time() - t_start:.1f}s)")
            
    # Save output JSON
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    out_path = args.output or os.path.join(Config.RESULTS_DIR, "responses.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=2)
        
    print(f"\nSuccessfully generated benchmark responses file:")
    print(f"-> {out_path}")
    print(f"Total Test Users processed: {len(responses['testing'])}")

if __name__ == "__main__":
    main()
