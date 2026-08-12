import torch
import torch.nn as nn

class SmcMultiTaskNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Head 1: tp_probability (sigmoid)
        self.head_tp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Head 2: direction_logits (softmax over 3 classes: bearish/neutral/bullish)
        # 0=bearish, 1=neutral, 2=bullish
        self.head_dir = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # Head 3: risk_score (sigmoid)
        self.head_risk = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.shared(x)
        
        tp_prob = self.head_tp(features)
        direction = self.head_dir(features)
        risk = self.head_risk(features)
        
        return tp_prob, direction, risk
