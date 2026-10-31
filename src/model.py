import torch
import torch.nn as nn


class VectorNet(nn.Module):
    """
    Lightweight acoustic classifier for mosquito vector identification.
    Optimized for low-latency inference on embedded/edge hardware.
    """

    def __init__(self, in_features=48, num_classes=6, hidden_dims=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = in_features

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
