import os
import unittest
import torch
import numpy as np

from config import Config
from models import BaseInceptionCNN, EnhancedCNN, BaseInceptionCNNLSTM, EnhancedHGRModel
from dataset import EMGAugmentor, sequence_collate_fn
from train import FocalLoss
from metrics import compute_classification_metrics, save_metrics_to_csv, save_confusion_matrix_to_csv

class TestOption2Implementation(unittest.TestCase):

    def test_01_baseline_models_preserved(self):
        """Verify Base CNN, Enhanced CNN, Base CNN-LSTM, and Enhanced CNN-LSTM are operational."""
        B, T, C, F_dim, TB = 2, 4, 8, 13, 24
        mock_input = torch.randn(B, T, C, F_dim, TB)
        
        # 1. Test Base CNN
        base_cnn = BaseInceptionCNN(num_classes=Config.NUM_CLASSES)
        cnn_out = base_cnn(mock_input)
        self.assertEqual(cnn_out.shape, (B, T, Config.NUM_CLASSES))
        
        # 2. Test Enhanced CNN
        enh_cnn = EnhancedCNN(num_classes=Config.NUM_CLASSES)
        enh_cnn_out = enh_cnn(mock_input)
        self.assertEqual(enh_cnn_out.shape, (B, T, Config.NUM_CLASSES))
        
        # 3. Test Base Inception CNN-LSTM
        base_lstm = BaseInceptionCNNLSTM(num_classes=Config.NUM_CLASSES, hidden_dim=Config.LSTM_HIDDEN_DIM)
        lstm_out = base_lstm(mock_input)
        self.assertEqual(lstm_out.shape, (B, T, Config.NUM_CLASSES))
        
        # 4. Test Enhanced CNN-LSTM Model
        enhanced = EnhancedHGRModel(num_classes=Config.NUM_CLASSES, hidden_dim=Config.LSTM_HIDDEN_DIM)
        enh_out = enhanced(mock_input)
        self.assertEqual(enh_out.shape, (B, T, Config.NUM_CLASSES))


    def test_02_augmentation_and_channel_integrity(self):
        """Verify data augmentation behavior and channel ordering."""
        T, C, F_dim, TB = 5, 8, 13, 24
        mock_seq = torch.ones((T, C, F_dim, TB))
        
        # Default verified setup: Circular shift is disabled to preserve physical muscle mapping
        augmentor_safe = EMGAugmentor(enable_circular_shift=False, scale_prob=0.0, noise_prob=0.0)
        out_safe = augmentor_safe.apply(mock_seq)
        self.assertTrue(torch.allclose(mock_seq, out_safe))
        
        # Amplitude scaling test: preserves channels and spatial topology
        augmentor_scale = EMGAugmentor(enable_circular_shift=False, scale_prob=1.0, noise_prob=0.0)
        out_scaled = augmentor_scale.apply(mock_seq)
        self.assertEqual(out_scaled.shape, mock_seq.shape)
        self.assertTrue(torch.all(out_scaled > 0)) # Non-negative spectrograms

    def test_03_class_weights_calculation(self):
        """Verify class weights are calculated using training labels only and balance inverse frequencies."""
        # Mock training target labels with acute class imbalance (noGesture dominating)
        # Class 5 (noGesture): 100 frames, Gestures 0-4: 10 frames each
        mock_targets = [5] * 100 + [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10
        target_tensor = torch.tensor(mock_targets, dtype=torch.long)
        
        class_counts = torch.bincount(target_tensor, minlength=Config.NUM_CLASSES).float()
        total_frames = class_counts.sum()
        
        raw_weights = total_frames / (Config.NUM_CLASSES * torch.clamp(class_counts, min=1.0))
        class_weights = raw_weights / raw_weights.mean()
        
        # Minority classes (0-4) must have higher weight than dominant class 5 (noGesture)
        for c in range(5):
            self.assertGreater(class_weights[c].item(), class_weights[5].item())

    def test_04_focal_loss_correctness(self):
        """Verify Focal Loss down-weights easy examples and respects class weighting."""
        num_classes = 6
        gamma = 2.0
        
        # Case 1: Easy prediction (model is very confident in true class)
        easy_logits = torch.tensor([[10.0, -5.0, -5.0, -5.0, -5.0, -5.0]])
        target_easy = torch.tensor([0])
        
        # Case 2: Hard prediction (model is unconfident / misclassified)
        hard_logits = torch.tensor([[-5.0, 10.0, -5.0, -5.0, -5.0, -5.0]])
        target_hard = torch.tensor([0])
        
        focal_loss_fn = FocalLoss(gamma=gamma, label_smoothing=0.0)
        ce_loss_fn = torch.nn.CrossEntropyLoss()
        
        loss_easy_focal = focal_loss_fn(easy_logits, target_easy)
        loss_hard_focal = focal_loss_fn(hard_logits, target_hard)
        
        # Focal loss on easy example must be near zero due to (1 - p_t)^gamma modulating factor
        self.assertLess(loss_easy_focal.item(), 0.01)
        self.assertGreater(loss_hard_focal.item(), loss_easy_focal.item())

    def test_05_metrics_and_csv_export(self):
        """Verify comprehensive metrics calculation, confusion matrix, and CSV exports."""
        y_true = ["fist", "fist", "open", "pinch", "noGesture", "waveIn"]
        y_pred = ["fist", "open", "open", "pinch", "noGesture", "waveIn"]
        
        metrics = compute_classification_metrics(y_true, y_pred, Config.ALL_CLASSES)
        
        self.assertEqual(metrics["total_samples"], 6)
        self.assertEqual(metrics["total_correct"], 5) # 5 out of 6 correct
        self.assertAlmostEqual(metrics["accuracy"], (5/6)*100.0, places=1)
        self.assertIn("macro_precision", metrics)
        self.assertIn("macro_recall", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertEqual(len(metrics["confusion_matrix"]), len(Config.ALL_CLASSES))
        
        # Test CSV export
        test_csv_metrics = os.path.join(Config.RESULTS_DIR, "test_smoke_metrics.csv")
        test_csv_cm = os.path.join(Config.RESULTS_DIR, "test_smoke_cm.csv")
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        
        save_metrics_to_csv(metrics, test_csv_metrics)
        save_confusion_matrix_to_csv(metrics["confusion_matrix"], Config.ALL_CLASSES, test_csv_cm)
        
        self.assertTrue(os.path.exists(test_csv_metrics))
        self.assertTrue(os.path.exists(test_csv_cm))
        
        # Clean up test files
        if os.path.exists(test_csv_metrics):
            os.remove(test_csv_metrics)
        if os.path.exists(test_csv_cm):
            os.remove(test_csv_cm)

    def test_06_training_pipeline_smoke_step(self):
        """Smoke test verifying full forward, loss, backward, and optimization step end-to-end."""
        device = torch.device("cpu")
        model = EnhancedHGRModel(num_classes=Config.NUM_CLASSES, hidden_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = FocalLoss(gamma=2.0)
        
        # Batch: 2 sequences, each of length 3
        mock_seqs = torch.randn(2, 3, 8, 13, 24)
        mock_targets = torch.randint(0, Config.NUM_CLASSES, (2, 3))
        
        optimizer.zero_grad()
        logits = model(mock_seqs) # (2, 3, 6)
        
        loss = criterion(logits.view(-1, Config.NUM_CLASSES), mock_targets.view(-1))
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

if __name__ == "__main__":
    unittest.main()
