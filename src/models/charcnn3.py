"""Character-level CNN that pools the name and the ending apart (Model 2c): the revised
CNN of charcnn2.py (Model 2b) with two changes at the end of the network. Both aim at
the errors the trained Model 2b still makes: most of its missed DGA domains and of its
false alarms are short names, where DGA and legitimate names look alike.

1. The name and the ending are pooled apart. Model 2b keeps, for each filter, its
   maximum and its mean over the whole domain, and the length of the whole domain: it
   does not know where the name stops and the ending (the public suffix: .com, .co.zw,
   ...) starts. In a short name the ending is a large part of the domain: in bmn.xyz
   the mean of every filter is half .xyz, and the length is 7, not 3. Here each filter
   gives four numbers, its maximum and mean over the name and over the ending, and the
   length of both parts is given.
2. The lengths are not dropped out. Model 2b applies dropout to every input of its
   dense layer, its one length number included: in training the length is 0 half of
   the time, so the model learns not to rely on it. Here dropout is applied to the
   pooled numbers only (rate `pool_dropout`), and the lengths always reach the dense
   layer; the dropout after the dense layer is that of Model 2b.

Everything else is Model 2b: the embedding, the six convolutions with batch
normalization over the real characters, and the padding left out (charcnn2.py).

    embedding 128 -> 60, index 0 = padding (kept at zero)
    conv 128 x 7, conv 128 x 6, conv 128 x 5, conv 128 x 5, conv 128 x 4, conv 128 x 3
        each followed by batch normalization (over the real characters) and a ReLU
    max and mean of each filter over the name (2 x 128) and over the ending (2 x 128)
    dropout (pool_dropout) on these 512 numbers, then log(1 + length) of both parts -> 514
    dense 128 + ReLU, dropout, output layer(s)

The input says which characters are the ending: encode_split() gives the ASCII code
of every character, plus ENDING (128) for the characters of the ending. The model
reads the character (code % 128) with the same embedding wherever it is, so the
convolutions read the full domain as Model 2b does, across the dot; the mark (code >=
128) is only used by the pooling. Use encode_split() for this model, and encode() for
Model 2 and Model 2b.
"""
import numpy as np
import torch
from torch import nn

from src.features.split import split_domain
from src.models.charcnn2 import KERNEL_SIZES, ConvBlock
from src.models.encoding import MAX_LEN, encode

ENDING = 128   # added to the code of every character of the ending (ASCII codes are < 128)


def encode_split(domains, max_len=MAX_LEN):
    """uint8 array of shape (n, max_len): as encode(), plus ENDING on the ending.

    The ending is the public suffix and the dot before it, as split_domain() splits the
    domain (src/features/split.py, the split of the hand-crafted features):

        bmn.xyz -> name 'bmn', ending '.xyz'
        fdr.org.br -> name 'fdr', ending '.org.br'
        0756293c.duckdns.org -> name '0756293c', ending '.duckdns.org'
        mail.bbc.co.uk -> name 'mail.bbc', ending '.co.uk'

    A domain without a known suffix has no ending: all its characters are the name.
    """
    out = encode(domains, max_len)
    suffix = np.array([len(split_domain(domain)[2]) for domain in domains])
    ending = np.where(suffix > 0, suffix + 1, 0)   # with the dot before the suffix
    length = (out != 0).sum(1)
    position = np.arange(max_len)
    out[(position >= (length - ending)[:, None]) & (position < length[:, None])] += ENDING
    return out


def name_length(x):
    """Number of characters of the name (before the ending) of each row of encode_split()."""
    x = np.asarray(x)
    return ((x != 0) & (x < ENDING)).sum(1)


class CharCNN3(nn.Module):
    def __init__(self, outputs, width=128, hidden=128, dropout=0.5, pool_dropout=0.2,
                 embed_dim=60):
        super().__init__()
        self.embedding = nn.Embedding(ENDING, embed_dim, padding_idx=0)
        sizes = [embed_dim] + [width] * len(KERNEL_SIZES)
        self.convs = nn.ModuleList(ConvBlock(c_in, c_out, k)
                                   for c_in, c_out, k in zip(sizes, sizes[1:], KERNEL_SIZES))
        self.pool_dropout = nn.Dropout(pool_dropout)
        self.dense = nn.Sequential(nn.Linear(4 * width + 2, hidden), nn.ReLU(),
                                   nn.Dropout(dropout))
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, n) for task, n in outputs.items()})

    def forward(self, x):
        x = x.long()
        length = (x != 0).sum(1, keepdim=True)              # (batch, 1): real characters
        # Cut the batch to its longest domain, as in CharCNN2 (not while torch.export
        # traces the model: it cannot trace a cut whose size depends on the data).
        if not torch.compiler.is_compiling():
            x = x[:, :length.max()]
        mask = (x != 0).unsqueeze(1).float()               # (batch, 1, length): 1 = character
        ending = (x >= ENDING).unsqueeze(1).float()        # 1 = character of the ending
        h = self.embedding(x % ENDING).transpose(1, 2)     # the same character, same embedding
        for block in self.convs:
            h = block(h, mask)                             # the padding stays at zero
        pooled, lengths = [], []
        for part in (mask - ending, ending):               # the name, then the ending
            n = part.sum(2)                                # (batch, 1): its characters
            h_part = h * part                              # ReLU outputs are >= 0: 0 elsewhere
            pooled += [h_part.amax(2), h_part.sum(2) / n.clamp(min=1)]
            lengths.append(n)
        h = torch.cat([self.pool_dropout(torch.cat(pooled, 1)),
                       torch.cat(lengths, 1).log1p()], 1)  # the lengths are never dropped
        h = self.dense(h)
        return {task: head(h) for task, head in self.heads.items()}
