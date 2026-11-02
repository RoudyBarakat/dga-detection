"""Train and apply the neural models (PyTorch).

fit() follows the training setup of Gregorio et al. (2024), Table 2: Adam, learning
rate 1e-4, batch size 250, cross-entropy plus L1 (1e-5) and L2 (1e-4) penalties on
the convolution and dense weights, 20 epochs, keeping the best epoch on validation.
Two differences: the loss is weighted with the training sample weights, and the best
epoch is the one with the highest validation macro-F1 on the DGA/legitimate decision
(the paper uses accuracy). With several tasks (multi-task), their losses are summed.
"""
import copy
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.metrics import binary_metrics


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)   # also seeds the GPU


@torch.no_grad()
def predict_proba(model, x, batch_size=4096):
    """Softmax probabilities of every task: {task: array of shape (n, classes)}."""
    device = next(model.parameters()).device
    model.eval()
    out = {task: [] for task in model.heads}
    for start in range(0, len(x), batch_size):
        logits = model(torch.as_tensor(x[start:start + batch_size]).to(device))
        for task, z in logits.items():
            out[task].append(z.softmax(1).cpu().numpy())
    return {task: np.concatenate(p) for task, p in out.items()}


def p_dga(proba, task, classes):
    """P(DGA) = 1 - P(legit) from one task's probabilities (label or family)."""
    return 1 - proba[task][:, list(classes[task]).index("legit")]


def penalty(model, l1, l2):
    """L1 + L2 of the convolution and dense weights, added to the loss as Keras does."""
    weights = [p for name, p in model.named_parameters()
               if name.endswith("weight") and not name.startswith("embedding")]
    return sum(l1 * w.abs().sum() + l2 * w.square().sum() for w in weights)


def fit(model, x_train, y_train, weights, x_val, y_val, classes, decide, epochs=20,
        lr=1e-4, batch_size=250, l1=1e-5, l2=1e-4, seed=0, device=None):
    """Train `model` and keep the weights of its best epoch on validation.

    y_train: {task: class codes}, one entry per output layer of the model.
    weights: sample weight of each training domain, or None.
    y_val: 1 for DGA, 0 for legitimate. decide: the task whose P(DGA) is scored.
    Returns one row per epoch: training loss and validation metrics.
    """
    device = device or pick_device()
    model.to(device)
    tasks = list(y_train)
    w = np.ones(len(x_train)) if weights is None else np.array(weights)
    data = TensorDataset(torch.as_tensor(x_train), torch.as_tensor(w, dtype=torch.float32),
                         *(torch.as_tensor(y_train[task]) for task in tasks))
    loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history, best_f1, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        total = torch.zeros((), device=device)
        for xb, wb, *yb in loader:
            xb, wb = xb.to(device), wb.to(device)
            logits = model(xb)
            loss = sum((F.cross_entropy(logits[task], y.to(device), reduction="none") * wb).mean()
                       for task, y in zip(tasks, yb))
            loss = loss + penalty(model, l1, l2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(xb)

        p = p_dga(predict_proba(model, x_val), decide, classes)
        row = {"epoch": epoch, "train_loss": total.item() / len(x_train),
               **binary_metrics(y_val, p), "seconds": round(time.time() - start)}
        history.append(row)
        if row["macro_f1"] > best_f1:
            best_f1, best_state = row["macro_f1"], copy.deepcopy(model.state_dict())
        print(f"epoch {epoch:2d}: loss {row['train_loss']:.4f}, "
              f"val macro-F1 {row['macro_f1']:.4f}, AUC {row['roc_auc']:.4f} "
              f"({row['seconds']} s)")

    model.load_state_dict(best_state)
    return pd.DataFrame(history).set_index("epoch")
