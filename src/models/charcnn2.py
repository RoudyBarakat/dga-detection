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


class _MaskedBatchNormReLU(torch.autograd.Function):
    """relu(batch normalization over the real characters) * mask, in training.

    On the GPU this layer is limited by memory traffic, not by arithmetic: every
    elementwise operation reads and writes the whole (batch, channels, length) tensor.
    Written with plain tensor operations, autograd runs about 12 such passes forward and
    twice as many backward; this function does the same computation in fewer passes
    (one pass for both statistics, the affine folded into one scale and shift, a
    hand-written backward).
    """

    @staticmethod
    def forward(ctx, h, mask, weight, bias, eps):
        n = mask.sum()                                 # real characters in the batch
        hm = h * mask                                  # the padding set to zero
        var_all, mean_all = torch.var_mean(hm, dim=(0, 2), correction=0)   # one pass, all positions
        k = hm.numel() / hm.shape[1] / n               # positions / real characters
        mean = mean_all * k                            # the statistics of the real characters
        var = (k * var_all - k * (k - 1) * mean_all.square()).clamp_min(0)
        inv = torch.rsqrt(var + eps)
        scale = weight * inv
        y = torch.addcmul((bias - mean * scale)[:, None], h, scale[:, None]).mul_(mask).relu_()
        ctx.save_for_backward(h, mask, y, mean, inv, weight, n)
        ctx.mark_non_differentiable(mean, var)
        return y, mean, var

    @staticmethod
    def backward(ctx, grad_y, _grad_mean, _grad_var):
        h, mask, y, mean, inv, weight, n = ctx.saved_tensors
        g = torch.ops.aten.threshold_backward(grad_y, y, 0)   # 0 where the ReLU is off and at padding
        grad_bias = g.sum((0, 2))
        grad_weight = ((g * h).sum((0, 2)) - mean * grad_bias) * inv
        # Gradient through the batch statistics: grad_h = (a g + b h + c) * mask.
        a = weight * inv
        b = -a * inv * grad_weight / n
        c = -a * grad_bias / n - b * mean
        grad_h = torch.addcmul(c[:, None], h, b[:, None]).addcmul_(g, a[:, None]).mul_(mask)
        return grad_h, None, grad_weight, grad_bias, None


class MaskedBatchNormReLU(nn.BatchNorm1d):
    """Batch normalization whose statistics are computed over the real characters only,
    followed by a ReLU.

    With the padding counted, the mean and variance would depend on how much padding a
    batch has. The padding positions come out as zero. Same parameters and running
    statistics as nn.BatchNorm1d, used for prediction in the same way.
    """

    def forward(self, h, mask):   # h: (batch, channels, length), mask: (batch, 1, length)
        if self.training:
            y, mean, var = _MaskedBatchNormReLU.apply(h, mask, self.weight, self.bias, self.eps)
            with torch.no_grad():
                n = mask.sum()
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var * n / (n - 1), self.momentum)
                self.num_batches_tracked += 1
            return y
        scale = self.weight * torch.rsqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale
        return torch.addcmul(shift[:, None], h, scale[:, None]).mul_(mask).relu_()


class ConvBlock(nn.Module):
    """Convolution, then batch normalization over the real characters and a ReLU."""

    def __init__(self, c_in, c_out, size):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, size, padding="same")
        self.norm = MaskedBatchNormReLU(c_out)

    def forward(self, h, mask):
        return self.norm(self.conv(h), mask)


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
        # Cut the batch to its longest domain: the output is the same, only faster. Not
        # while torch.export traces the model (MLflow, tracking.log_model): it cannot
        # trace a cut whose size depends on the data.
        if not torch.compiler.is_compiling():
            x = x[:, :length.max()]
        mask = (x != 0).unsqueeze(1).float()               # (batch, 1, length): 1 = character
        h = self.embedding(x).transpose(1, 2)              # (batch, embed_dim, length)
        for block in self.convs:
            h = block(h, mask)                             # the padding stays at zero
        h_max = h.amax(2)          # ReLU outputs are >= 0 and the padding is 0: max over characters
        h_mean = h.sum(2) / length
        h = torch.cat([h_max, h_mean, length.float().log()], 1)
        h = self.dense(h)
        return {task: head(h) for task, head in self.heads.items()}
