"""Revised character-level CNN (Model 2b): the CNN of Gregorio et al. (2024) in
charcnn.py, with four changes, each fixing a weakness of the trained Model 2:

1. Global pooling instead of flatten. The paper's pools have stride 1, so the sequence
   keeps its 93 positions and the dense layer has one weight per position: a pattern
   is learned again at every place it can appear. Here each filter gives two numbers
   for the whole domain, its maximum (is the pattern anywhere?) and its mean (how much
   of it is there?), wherever the pattern is. The pools are removed. Pooling loses the
   domain length, which Model 2 read from where the padding starts, so log(length) is
   given to the dense layer.
2. The same width everywhere. The paper goes 500, 250, 60, 60, 30, 30 filters, then
   15 units: in the trained binary model 6 of the 15 units never activate. Here every
   convolution has 128 filters and the dense layer 128 units.
3. Dropout and batch normalization. Model 2 has only the L1/L2 penalties, and its
   validation loss rises after epoch 9 to 13. Here dropout is applied before and after
   the dense layer, and batch normalization after every convolution.
4. The padding is ignored. Model 2 reads the padding (83% of the 93 positions) as
   characters. Here padding has no embedding, stays at zero through the convolutions,
   and is left out of the pooling and of the batch-norm statistics. Each batch is cut
   to its longest domain, so the output does not depend on the padding and the
   convolutions only run on the real characters.

    embedding 256 -> 60, index 0 = padding (kept at zero)
    conv 128 x 7, conv 128 x 6, conv 128 x 5, conv 128 x 5, conv 128 x 4, conv 128 x 3
        each followed by batch normalization (over the real characters) and a ReLU
    max and mean of each filter over the real characters (2 x 128), log(length) -> 257
    dropout, dense 128 + ReLU, dropout, output layer(s)

The kernel sizes are those of the paper. As in charcnn.py, there is one output layer
per task in `outputs` ({"label": 2}, {"family": 104} or both), and the model returns
logits; the softmax is in the loss.
"""
import torch
from torch import nn

from src.models.encoding import VOCAB

KERNEL_SIZES = (7, 6, 5, 5, 4, 3)   # the paper's six convolutions


class MaskedBatchNorm(nn.BatchNorm1d):
    """Batch normalization whose statistics are computed over the real characters only.

    With the padding counted, the mean and variance would depend on how much padding a
    batch has. The padding positions come out as zero.
    """

    def forward(self, h, mask):   # h: (batch, channels, length), mask: (batch, 1, length)
        if self.training:
            n = mask.sum()
            mean = (h * mask).sum((0, 2)) / n
            var = ((h - mean[:, None]) ** 2 * mask).sum((0, 2)) / n
            with torch.no_grad():   # running statistics, used for prediction (as nn.BatchNorm1d)
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var * n / (n - 1), self.momentum)
                self.num_batches_tracked += 1
        else:
            mean, var = self.running_mean, self.running_var
        h = (h - mean[:, None]) / torch.sqrt(var[:, None] + self.eps)
        return (h * self.weight[:, None] + self.bias[:, None]) * mask


class ConvBlock(nn.Module):
    """Convolution, batch normalization over the real characters, ReLU."""

    def __init__(self, c_in, c_out, size):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, size, padding="same")
        self.norm = MaskedBatchNorm(c_out)

    def forward(self, h, mask):
        return torch.relu(self.norm(self.conv(h), mask))


class CharCNN2(nn.Module):
    def __init__(self, outputs, width=128, hidden=128, dropout=0.5, embed_dim=60):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, embed_dim, padding_idx=0)
        sizes = [embed_dim] + [width] * len(KERNEL_SIZES)
        self.convs = nn.ModuleList(ConvBlock(c_in, c_out, k)
                                   for c_in, c_out, k in zip(sizes, sizes[1:], KERNEL_SIZES))
        self.dense = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * width + 1, hidden),
                                   nn.ReLU(), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, n) for task, n in outputs.items()})

    def forward(self, x):
        x = x.long()
        length = (x != 0).sum(1, keepdim=True)              # (batch, 1): real characters
        x = x[:, :length.max()]                            # cut the batch to its longest domain
        mask = (x != 0).unsqueeze(1).float()               # (batch, 1, length): 1 = character
        h = self.embedding(x).transpose(1, 2)              # (batch, embed_dim, length)
        for block in self.convs:
            h = block(h, mask)                             # the padding stays at zero
        h_max = h.amax(2)          # ReLU outputs are >= 0 and the padding is 0: max over characters
        h_mean = h.sum(2) / length
        h = torch.cat([h_max, h_mean, length.float().log()], 1)
        h = self.dense(h)
        return {task: head(h) for task, head in self.heads.items()}
