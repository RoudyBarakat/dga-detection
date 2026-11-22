"""MLP on the hand-crafted features (Model 3).

The same 74 features as the logistic regression (Model 1), but with hidden layers:

    x (features) -> standardize -> [Linear -> ReLU -> Dropout] x len(hidden) -> Linear -> logits

Without the ReLU the layers would collapse into one linear map, i.e. into LR again; with
it, each hidden unit is a small detector that switches on in one region of the feature
space, and the next layer combines them, so the boundary between classes can bend.

Like CharCNN, the model returns {task: logits} (one output per task), so it is trained
with fit() of src/models/train.py, the same loop as the CNN. The standardization (mean
and standard deviation of every feature on the training set) is stored in the model, so
the model takes the raw features.
"""
import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, mean, std, outputs, hidden=(128, 64), dropout=0.1):
        """mean, std: of every feature on the training set. outputs: {task: n_classes}."""
        super().__init__()
        std = np.where(np.asarray(std) > 0, std, 1.0)   # a constant feature stays 0
        self.register_buffer("mean", torch.as_tensor(np.asarray(mean), dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        layers, n_in = [], len(mean)
        for n_out in hidden:
            layers += [nn.Linear(n_in, n_out), nn.ReLU(), nn.Dropout(dropout)]
            n_in = n_out
        self.body = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({task: nn.Linear(n_in, n) for task, n in outputs.items()})

    def forward(self, x):
        h = self.body((x - self.mean) / self.std)
        return {task: head(h) for task, head in self.heads.items()}
