"""Train and apply the neural models (PyTorch).

fit() runs the training of Gregorio et al. (2024), Table 2: the optimizer (Adam in
the paper) minimizes the loss on shuffled mini-batches for a number of epochs, and the
best epoch on validation is kept. Here the best epoch is the one with the highest
validation macro-F1 on the DGA/legitimate decision (the paper uses accuracy). The
notebook defines the predictor, the loss and the optimizer, and passes them to fit().

After every epoch, fit() saves the training in a folder, logs it to MLflow
(tracking.py) and redraws the loss plot (plots.py). A training that was interrupted
continues from its last saved epoch when fit() is called again with the same folder:

    folder/epoch_01.pt, ...   the model weights after each epoch
    folder/history.csv        J_train, J_val and the validation metrics of every epoch
    folder/last.pt            everything needed to continue (weights, optimizer, shuffling)
    folder/best.pt            the weights of the kept epoch, once training has finished
    folder/mlflow_run_id.txt  the MLflow run of this training
"""
import random
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset

from src.models import tracking
from src.models.metrics import binary_metrics
from src.models.plots import LivePlot


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


class Split(NamedTuple):
    """One split of the data, as fit() uses it.

    x: the encoded domains. y: the class codes of the target, or {task: codes} for a
    model with several outputs. w: the sample weights (None: all 1). dga: 1 for DGA,
    0 for legitimate (needed for validation: the metrics are on that decision).
    """
    x: np.ndarray
    y: object
    w: np.ndarray | None = None
    dga: np.ndarray | None = None


def fit(model, loss_fn, optimizer, train, val, *, decide, classes, folder, epochs=20,
        batch_size=250, seed=0, params=None, device=None):
    """Train `model` and keep the weights of its best validation epoch.

    loss_fn(model, logits, y, w): the loss of one batch, minimized by `optimizer`.
    train, val: Split. decide: the output whose P(DGA) is scored on validation.
    folder: where the training is saved; if it holds an interrupted training, that
    training continues. params: the settings of the run (learning rate, penalties...),
    logged to MLflow and checked when a training continues (setup MLflow first with
    tracking.setup).
    Returns one row per epoch (J_train, J_val and the validation metrics) and the
    validation P(DGA) of the kept epoch; `model` keeps the weights of the kept epoch.
    """
    device = device or pick_device()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.to(device)
    x, w, y = _tensors(train)
    tasks = list(y) if isinstance(y, dict) else None
    data = TensorDataset(x, w, *(y.values() if tasks else [y]))
    # Shuffled mini-batches, each taken with one indexing step instead of one domain at a
    # time: the same batches as DataLoader(shuffle=True), about 10x less loading time.
    shuffle = torch.Generator().manual_seed(seed)
    batches = BatchSampler(RandomSampler(data, generator=shuffle), batch_size, drop_last=False)
    loader = DataLoader(data, sampler=batches, batch_size=None, generator=shuffle)

    settings = {**(params or {}), "batch_size": batch_size, "seed": seed}
    history, best_f1, best_p = [], -1.0, None
    last = _load_last(folder, settings)
    if last is not None:
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        shuffle.set_state(last["shuffle"])   # the next epochs get the batches they would have had
        history, best_f1, best_p = last["history"], last["best_f1"], last["best_p"]
        print(f"{len(history)} epoch(s) already saved in {folder.name}")
    if len(history) >= epochs and (folder / "best.pt").exists():   # finished: only reload
        model.load_state_dict(torch.load(folder / "best.pt", map_location=device,
                                         weights_only=True))
        return pd.DataFrame(history).set_index("epoch"), best_p

    plot = LivePlot(f"Loss, {folder.name}")
    plot.update(history)
    run_params = {**settings, "train_rows": len(x), "device": str(device),
                  "checkpoints": str(folder)}
    with tracking.start_run(folder.name, run_params, folder=folder):
        for epoch in range(len(history) + 1, epochs + 1):
            start = time.time()
            train_loss = train_epoch(model, loss_fn, optimizer, loader, tasks, device)
            val_loss, p = evaluate(model, loss_fn, val, decide, classes, device)
            metrics = binary_metrics(val.dga, p)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                            **metrics, "seconds": round(time.time() - start)})
            if metrics["macro_f1"] > best_f1:
                best_f1, best_p = metrics["macro_f1"], p

            _save_epoch(folder, epoch, model, {
                "settings": settings, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "shuffle": shuffle.get_state(),
                "history": history, "best_f1": best_f1, "best_p": best_p})
            tracking.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                                  **{f"val_{k}": v for k, v in metrics.items()}}, step=epoch)
            print(f"epoch {epoch:2d}: J_train {train_loss:.4f}, J_val {val_loss:.4f}, "
                  f"val macro-F1 {metrics['macro_f1']:.4f}, AUC {metrics['roc_auc']:.4f} "
                  f"({history[-1]['seconds']} s)")
            plot.update(history)

        # Keep the best epoch, and log it to MLflow with the history and the loss plot.
        history = pd.DataFrame(history).set_index("epoch")
        best = int(history["macro_f1"].idxmax())   # the first epoch with the highest macro-F1
        model.load_state_dict(torch.load(folder / f"epoch_{best:02d}.pt", map_location=device,
                                         weights_only=True))
        torch.save(model.state_dict(), folder / "best.pt")
        tracking.log_metrics({"kept_epoch": best, "kept_val_macro_f1": best_f1})
        tracking.log_files(folder / "history.csv")
        tracking.log_figure(plot.fig, "losses.png")
        tracking.log_model(model, input_example=val.x[:5])
    print(f"kept epoch {best} (val macro-F1 {best_f1:.4f})")
    return history, best_p


def train_epoch(model, loss_fn, optimizer, loader, tasks, device):
    """One pass of the optimizer over the shuffled training batches; returns J_train."""
    model.train()
    total, n = torch.zeros((), device=device), 0
    for xb, wb, *yb in loader:
        xb, wb = xb.to(device), wb.to(device)
        yb = {t: v.to(device) for t, v in zip(tasks, yb)} if tasks else yb[0].to(device)
        loss = loss_fn(model, model(xb), yb, wb)
        optimizer.zero_grad()
        loss.backward()    # gradient of the loss
        optimizer.step()   # one optimizer step
        total += loss.detach() * len(xb)
        n += len(xb)
    return total.item() / n


@torch.no_grad()
def evaluate(model, loss_fn, split, decide, classes, device, batch_size=4096):
    """The loss on the whole `split` (J_val) and the P(DGA) of each of its domains."""
    model.eval()
    x, w, y = _tensors(split)
    total, proba = 0.0, []
    for start in range(0, len(x), batch_size):
        rows = slice(start, start + batch_size)
        logits = model(x[rows].to(device))
        yb = ({t: v[rows].to(device) for t, v in y.items()} if isinstance(y, dict)
              else y[rows].to(device))
        total += loss_fn(model, logits, yb, w[rows].to(device)).item() * len(x[rows])
        proba.append(logits[decide].softmax(1).cpu().numpy())
    return total / len(x), p_dga({decide: np.concatenate(proba)}, decide, classes)


def load_history(folder):
    """The saved history of the training in `folder` (one row per epoch), or None."""
    path = Path(folder) / "history.csv"
    return pd.read_csv(path, index_col="epoch") if path.exists() else None


def _tensors(split):
    """x, w and y of `split` as tensors (y: one tensor, or {task: tensor})."""
    w = np.ones(len(split.x)) if split.w is None else split.w
    y = ({task: _tensor(codes) for task, codes in split.y.items()}
         if isinstance(split.y, dict) else _tensor(split.y))
    return _tensor(split.x), _tensor(w, np.float32), y


def _tensor(values, dtype=None):
    """`values` as a tensor, sharing its memory unless it is read-only (as pandas returns)."""
    values = np.asarray(values, dtype=dtype)
    return torch.as_tensor(values if values.flags.writeable else values.copy())


def _load_last(folder, settings):
    """The saved state of the training in `folder`, or None if it has not started."""
    if not (folder / "last.pt").exists():
        return None
    last = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
    if last["settings"] != settings:
        raise ValueError(f"{folder} holds a training with {last['settings']}, not "
                         f"{settings}: use another folder to start a new training")
    return last


def _save_epoch(folder, epoch, model, state):
    """Save the weights of `epoch`, the history so far, and the state to continue from."""
    torch.save(model.state_dict(), folder / f"epoch_{epoch:02d}.pt")
    pd.DataFrame(state["history"]).to_csv(folder / "history.csv", index=False)
    torch.save(state, folder / "last.tmp")
    (folder / "last.tmp").replace(folder / "last.pt")   # an interrupted save keeps the old one
