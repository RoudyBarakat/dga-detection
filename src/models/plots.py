"""Plots of the training, shared by the notebooks.

plot_losses() draws J_train and J_val per epoch on a matplotlib axis. LivePlot shows
that plot in the notebook output and redraws it after every epoch (fit() in train.py
uses it while a model trains).
"""
import io

import pandas as pd
from IPython.display import Image, display
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


def plot_losses(ax, history, title="Loss"):
    """J_train and J_val per epoch, with a dotted line at the kept epoch (best val macro-F1).

    history: one row per epoch, indexed by epoch, with the columns train_loss, val_loss
    and macro_f1 (as returned by fit() or read by load_history()).
    """
    ax.clear()
    ax.plot(history.index, history["train_loss"], marker="o", label="$J_{train}$")
    ax.plot(history.index, history["val_loss"], marker="o", label="$J_{val}$")
    ax.axvline(history["macro_f1"].idxmax(), color="grey", linestyle=":", label="kept epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()


class LivePlot:
    """A loss plot in the notebook output, redrawn after every epoch."""

    def __init__(self, title, figsize=(7, 3.6)):
        self.title = title
        self.fig = Figure(figsize=figsize)   # not a pyplot figure: shown only by update()
        self.ax = self.fig.subplots()
        self.handle = None

    def update(self, history):
        """Draw the epochs of `history` (a list of rows with an "epoch" key, or a DataFrame)."""
        history = pd.DataFrame(history)
        if "epoch" in history:
            history = history.set_index("epoch")
        if len(history):
            plot_losses(self.ax, history, self.title)
        else:
            self.ax.set_title(f"{self.title}: waiting for epoch 1")
        self.fig.tight_layout()
        # Shown as a PNG image: a notebook only draws figures by itself once pyplot has
        # made one, which is not yet the case when training starts.
        png = io.BytesIO()
        self.fig.savefig(png, format="png")
        image = Image(data=png.getvalue(), format="png")
        if self.handle is None:
            self.handle = display(image, display_id=True)
        else:
            self.handle.update(image)
