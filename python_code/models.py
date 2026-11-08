import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

# =====================================================================
# 1. BASE PAPER INCEPTION CNN ARCHITECTURE (Exact Reproduction)
# =====================================================================

class InceptionBlock(nn.Module):
    """
    Standard Inception module matching MATLAB modelCreation.m:
    Branch 1: 1x1 conv (18 filters)
    Branch 2: 1x1 reduce (16 filters) -> 3x3 conv (18 filters, pad 1)
    Branch 3: 1x1 reduce (16 filters) -> 5x5 conv (18 filters, pad 2)
    Branch 4: 3x3 maxpool (stride 1, pad 1) -> 1x1 proj (18 filters)
    Concat channels: 18 * 4 = 72
    """
    def __init__(self, in_channels: int, out_per_branch: int = 18, reduce_channels: int = 16):
        super().__init__()
        # Branch 1
        self.b1 = nn.Conv2d(in_channels, out_per_branch, kernel_size=1)
        
        # Branch 2 (3x3)
        self.b2_reduce = nn.Conv2d(in_channels, reduce_channels, kernel_size=1)
        self.b2_conv = nn.Conv2d(reduce_channels, out_per_branch, kernel_size=3, padding=1)
        
        # Branch 3 (5x5)
        self.b3_reduce = nn.Conv2d(in_channels, reduce_channels, kernel_size=1)
        self.b3_conv = nn.Conv2d(reduce_channels, out_per_branch, kernel_size=5, padding=2)
        
        # Branch 4 (Pool)
        self.b4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_proj = nn.Conv2d(in_channels, out_per_branch, kernel_size=1)

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2_conv(F.relu(self.b2_reduce(x)))
        b3 = self.b3_conv(F.relu(self.b3_reduce(x)))
        b4 = self.b4_proj(self.b4_pool(x))
        out = torch.cat([b2, b3, b4, b1], dim=1) # 72 channels
        return F.relu(out)


class InceptionCNNBackbone(nn.Module):
    """
    Deep 6-module Inception network with residual addition skips:
    1a -> 1b -> 1c (+ 1a) -> 1d -> 1e (+ 1c) -> 1f
    """
    def __init__(self, in_channels: int = 8):
        super().__init__()
        self.inc_1a = InceptionBlock(in_channels)
        self.inc_1b = InceptionBlock(72)
        self.inc_1c = InceptionBlock(72)
        self.inc_1d = InceptionBlock(72)
        self.inc_1e = InceptionBlock(72)
        self.inc_1f = InceptionBlock(72)
        self.norm = nn.BatchNorm2d(72)

    def forward(self, x):
        # Module 1a
        out_1a = self.inc_1a(x)
        
        # Module 1b
        out_1b = self.inc_1b(out_1a)
        
        # Module 1c + Residual connection from 1a
        out_1c = F.relu(self.inc_1c(out_1b) + out_1a)
        
        # Module 1d
        out_1d = self.inc_1d(out_1c)
        
        # Module 1e + Residual connection from 1c
        out_1e = F.relu(self.inc_1e(out_1d) + out_1c)
        
        # Module 1f
        out_1f = self.norm(self.inc_1f(out_1e))
        return out_1f


class BaseInceptionCNN(nn.Module):
    """Base Paper CNN Model for single frame or sequence frame classification"""
    def __init__(self, num_classes: int = Config.NUM_CLASSES):
        super().__init__()
        self.backbone = InceptionCNNBackbone(in_channels=8)
        self.fc = nn.Linear(72 * 13 * 24, num_classes)

    def forward(self, x):
        if x.dim() == 5:
            B, T, C, F_dim, TB = x.shape
            x_reshaped = x.view(B * T, C, F_dim, TB)
            feat = self.backbone(x_reshaped)
            logits = self.fc(feat.view(B * T, -1))
            return logits.view(B, T, -1)
        else:
            feat = self.backbone(x)
            logits = self.fc(feat.view(feat.size(0), -1))
            return logits


class EnhancedCNN(nn.Module):
    """Enhanced CNN Model with Channel Attention and Dense Feature Projection (no LSTM)"""
    def __init__(self, num_classes: int = Config.NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.channel_attn = ChannelAttention(channels=8)
        self.backbone = InceptionCNNBackbone(in_channels=8)
        raw_dim = 72 * 13 * 24
        self.classifier = nn.Sequential(
            nn.Linear(raw_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.dim() == 5:
            B, T, C, F_dim, TB = x.shape
            x_reshaped = x.view(B * T, C, F_dim, TB)
            weighted_x = self.channel_attn(x_reshaped)
            cnn_feats = self.backbone(weighted_x)
            logits = self.classifier(cnn_feats.view(B * T, -1))
            return logits.view(B, T, -1)
        else:
            weighted_x = self.channel_attn(x)
            cnn_feats = self.backbone(weighted_x)
            logits = self.classifier(cnn_feats.view(cnn_feats.size(0), -1))
            return logits



class BaseInceptionCNNLSTM(nn.Module):
    """Base Paper CNN-LSTM Model for sequence of spectrogram frames"""
    def __init__(self, num_classes: int = Config.NUM_CLASSES, hidden_dim: int = Config.LSTM_HIDDEN_DIM):
        super().__init__()
        self.backbone = InceptionCNNBackbone(in_channels=8)
        self.feat_dim = 72 * 13 * 24
        
        # Unidirectional LSTM matching paper (128 hidden units)
        self.lstm = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (Batch, TimeSteps, Channels=8, Freq=13, TimeBins=24)
        B, T, C, F_dim, TB = x.shape
        x_reshaped = x.view(B * T, C, F_dim, TB)
        
        # Extract features per frame
        cnn_feats = self.backbone(x_reshaped)
        cnn_flat = cnn_feats.view(B, T, -1)
        
        # Sequence modeling with LSTM
        lstm_out, _ = self.lstm(cnn_flat)
        logits = self.fc(lstm_out) # (B, T, num_classes)
        return logits


# =====================================================================
# 2. NOVEL ENHANCED HGR MODEL (To Outperform Base Paper > 90.55%)
# =====================================================================

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention across 8 EMG electrode sensors.
    Dynamically re-weights informative muscle electrode channels.
    """
    def __init__(self, channels: int = 8, reduction: int = 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class TemporalSelfAttention(nn.Module):
    """
    Multi-Head Temporal Self-Attention over sequence time steps.
    Focuses on muscle burst transitions rather than passive resting frames.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, T, hidden_dim)
        attn_out, _ = self.mha(x, x, x)
        return self.norm(x + attn_out)


class EnhancedHGRModel(nn.Module):
    """
    Advanced Spatial-Temporal Hand Gesture Recognition Network:
    1. Electrode Channel Attention on the 8 EMG channels
    2. Deep Inception Backbone with Residual Skip Paths
    3. Dimensionality reduction projection
    4. Bidirectional LSTM (BiLSTM) capturing past and future muscle activation states
    5. Multi-Head Temporal Self-Attention
    6. Classification head with Dropout regularization
    """
    def __init__(self, num_classes: int = Config.NUM_CLASSES, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        # 1. Electrode Channel Attention
        self.channel_attn = ChannelAttention(channels=8)
        
        # 2. Inception Backbone
        self.backbone = InceptionCNNBackbone(in_channels=8)
        
        # 3. Dense feature projection
        raw_dim = 72 * 13 * 24
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 4. Bidirectional LSTM (BiLSTM)
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # 5. Temporal Self-Attention
        self.temporal_attn = TemporalSelfAttention(hidden_dim=hidden_dim, num_heads=4)
        
        # 6. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, 8, 13, 24)
        B, T, C, F_dim, TB = x.shape
        x_reshaped = x.view(B * T, C, F_dim, TB)
        
        # Electrode attention + CNN feature extraction
        weighted_x = self.channel_attn(x_reshaped)
        cnn_feats = self.backbone(weighted_x)
        cnn_flat = cnn_feats.view(B * T, -1)
        
        # Project dimension
        proj_feats = self.proj(cnn_flat).view(B, T, -1)
        
        # BiLSTM sequence representation
        lstm_out, _ = self.bilstm(proj_feats)
        
        # Temporal Self-Attention
        attn_out = self.temporal_attn(lstm_out)
        
        # Output frame-level predictions
        logits = self.classifier(attn_out) # (B, T, num_classes)
        return logits
