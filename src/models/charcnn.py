"""Character-level CNN of Gregorio et al. (2024, Applied Sciences 14:7244), Figure 1.

    embedding 256 -> 60
    conv 500 x 7, conv 250 x 6, max-pool 6
    conv 60 x 5,  conv 60 x 5,  max-pool 4
    conv 30 x 4,  conv 30 x 3,  max-pool 3
    flatten (93 x 30 = 2790), dense 15, output layer(s)

Every convolution is followed by a ReLU. The figure's flatten size, 2100 = 70 x 30,
means that the sequence keeps its input length throughout, so convolutions and
max-pooling use stride 1 and 'same' padding (as Keras does with padding="same").
The input here is 93 characters instead of the paper's 70 (see encoding.py), so the
flattened size is 93 x 30 = 2790.

The paper has one output layer (softmax over n classes). Here there is one per task
in `outputs`: {"label": 2} (binary), {"family": 104} (family-first) or both
(multi-task, shared encoder). The model returns logits; the softmax is in the loss.
"""
import torch.nn.functional as F
from torch import nn

from src.models.encoding import MAX_LEN, VOCAB


class SamePool(nn.Module):
    """Max-pooling with stride 1 and 'same' padding (the extra padding on the right, as in Keras)."""

    def __init__(self, size):
        super().__init__()
        self.pad = ((size - 1) // 2, size // 2)
        self.pool = nn.MaxPool1d(size, stride=1)

    def forward(self, x):
        return self.pool(F.pad(x, self.pad, value=float("-inf")))


def conv(c_in, c_out, size):
    return [nn.Conv1d(c_in, c_out, size, padding="same"), nn.ReLU()]


class CharCNN(nn.Module):
    def __init__(self, outputs, max_len=MAX_LEN, embed_dim=60, hidden=15):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, embed_dim)
        self.convs = nn.Sequential(
            *conv(embed_dim, 500, 7), *conv(500, 250, 6), SamePool(6),
            *conv(250, 60, 5), *conv(60, 60, 5), SamePool(4),
            *conv(60, 30, 4), *conv(30, 30, 3), SamePool(3))
        self.dense = nn.Sequential(nn.Flatten(), nn.Linear(max_len * 30, hidden), nn.ReLU())
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, n) for task, n in outputs.items()})

    def forward(self, x):
        h = self.embedding(x.long()).transpose(1, 2)   # (batch, embed_dim, max_len)
        h = self.dense(self.convs(h))
        return {task: head(h) for task, head in self.heads.items()}
