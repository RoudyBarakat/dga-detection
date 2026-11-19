"""Plots of the training, shared by the notebooks.

plot_losses() draws J_train and J_val per epoch on a matplotlib axis. LivePlot shows
that plot in the notebook output and redraws it after every epoch (fit() in train.py
uses it while a model trains). plot_latent() and plot_conflicts() compare the latent
spaces of trained models (latent.py).
"""
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from matplotlib import patheffects
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter


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


# --- Latent spaces (latent.py) ---------------------------------------------------------
# Colors: the first three categorical slots, validated together for scatter plots (blue
# and orange for the models, aqua for DGA domains), gray for legitimate domains, and ink
# for the domains in conflict, so they stand out whatever the model.
MODEL_COLORS = {"Model 2b": "#2a78d6", "Model 2": "#eb6834"}   # blue, orange
DGA_COLOR, LEGIT_COLOR = "#1baf7a", "#c3c2b7"                  # aqua, gray
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def _model_colors(names):
    spare = iter(c for c in ("#2a78d6", "#eb6834", "#1baf7a") if c not in MODEL_COLORS.values())
    return {n: MODEL_COLORS[n] if n in MODEL_COLORS else next(spare) for n in names}


def plot_latent(spaces, title, families=True):
    """The 2-D maps of the latent spaces of `spaces` ({model name: latent.LatentSpace}),
    one column per model.

    Row 1, DGA vs legitimate: the DGA domains whose neighbours are mostly legitimate
    (missed) and the legitimate domains whose neighbours are mostly DGA (false alarms)
    are in ink, and the 3 DGA families most mixed with legitimate domains are named.
    Row 2 (if `families`), DGA families: the DGA domains whose neighbours are mostly of
    other DGA families are in ink, and the 3 pairs of families most mixed are named.
    """
    names = list(spaces)
    colors = _model_colors(names)
    rows = ["label", "family"] if families else ["label"]
    fig = plt.figure(figsize=(5.6 * len(names) + 3.2, 6.4 * len(rows) + 1.7), facecolor=SURFACE)
    top = 1 - 1.85 / fig.get_figheight()   # room for the header and the panel titles
    grid = fig.add_gridspec(len(rows), len(names) + 1, width_ratios=[1] * len(names) + [0.5],
                            left=0.02, right=0.99, top=top, bottom=0.02, wspace=0.05,
                            hspace=0.17)
    n = len(next(iter(spaces.values())).points)
    fig.text(0.02, 1 - 0.25 / fig.get_figheight(), title, fontsize=15, fontweight="bold",
             color=INK, va="top")
    fig.text(0.02, 1 - 0.62 / fig.get_figheight(),
             f"The same {n:,} validation domains for every model (up to 100 per DGA family, as many "
             "legitimate). A domain's neighbours are its 15 nearest in the full latent space.\n"
             "The maps are t-SNE pictures of that space: who is next to whom is kept, positions and "
             "distances are not. The percentages, not the picture, are the evidence.",
             fontsize=9.5, color=INK_2, va="top")
    for r, kind in enumerate(rows):
        for c, name in enumerate(names):
            ax = fig.add_subplot(grid[r, c])
            handles = _latent_map(ax, spaces[name], kind)
            s = spaces[name].scores
            if kind == "label":
                head = f"{name}: DGA vs legitimate"
                sub = f"{s['label_separation']:.1%} of neighbours have the same label"
            else:
                head = f"{name}: DGA families"
                sub = f"{s['family_separation']:.1%} of neighbours of the same family"
            _panel_title(ax, head, sub + " (higher is better)", colors[name])
        legend = fig.add_subplot(grid[r, -1])
        legend.axis("off")
        legend.legend(handles=handles, loc="center left", fontsize=9, frameon=False,
                      labelcolor=INK_2, handletextpad=0.4, labelspacing=1.1,
                      title="DGA vs legitimate" if kind == "label" else "DGA families",
                      title_fontproperties={"weight": "bold", "size": 9.5}, alignment="left")
    return fig


def plot_conflicts(spaces, families=True, top=12):
    """The `top` DGA families most mixed with legitimate domains and (if `families`) the
    `top` pairs of DGA families most mixed with each other, in the latent space of each
    model of `spaces`: one marker per model, worst first, lower is better."""
    colors = _model_colors(spaces)
    legit = pd.DataFrame({n: s.families["legit_share"] for n, s in spaces.items()})
    panels = [(legit, "DGA families most mixed with legitimate domains",
               "share of the family's neighbours that are legitimate")]
    if families:
        pairs = pd.DataFrame({n: s.pairs.set_index("pair")["share"]
                              for n, s in spaces.items()}).fillna(0)
        panels.append((pairs, "DGA families most mixed with each other",
                       "share of neighbours from the other family of the pair"))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.4 * len(panels), 0.36 * top + 1.6),
                             layout="constrained", facecolor=SURFACE, squeeze=False)
    for ax, (table, title, xlabel) in zip(axes[0], panels):
        _dumbbell(ax, table, colors, top, title, xlabel)
    return fig


def _latent_map(ax, space, kind):
    """One 2-D map: legitimate gray, DGA aqua, the domains in conflict in ink. Returns
    the legend entries."""
    p, xy = space.points, space.xy
    legit = (p["label"] == "legit").to_numpy()
    family = p["family"].to_numpy()
    if kind == "label":
        dga_in_legit = ~legit & (p["legit_share"] > 0.5).to_numpy()
        legit_in_dga = legit & (p["legit_share"] < 0.5).to_numpy()
        quiet = ~(dga_in_legit | legit_in_dga)
        ax.scatter(*xy[quiet & legit].T, s=3, c=LEGIT_COLOR, lw=0, rasterized=True)
        ax.scatter(*xy[quiet & ~legit].T, s=3, c=DGA_COLOR, lw=0, alpha=0.7, rasterized=True)
        ax.scatter(*xy[dga_in_legit].T, s=9, c=INK, lw=0, rasterized=True)
        ax.scatter(*xy[legit_in_dga].T, s=18, facecolors="none", edgecolors=INK, lw=0.7,
                   rasterized=True)
        handles = [_dot(LEGIT_COLOR, "legitimate"), _dot(DGA_COLOR, "DGA"),
                   _dot(INK, "DGA among legitimate:\nlikely missed"),
                   _dot(INK, "legitimate among DGA:\nlikely false alarm", hollow=True),
                   _dot(None, "named: the 3 DGA families\nmost mixed with legitimate", text=True)]
        labels = [(xy[(family == f) & dga_in_legit] if ((family == f) & dga_in_legit).sum() >= 3
                   else xy[family == f], f) for f in space.families.index[:3]]
    else:
        mixed = ~legit & (p["other_dga_share"] > 0.5).to_numpy()
        ax.scatter(*xy[legit].T, s=3, c=LEGIT_COLOR, lw=0, alpha=0.5, rasterized=True)
        ax.scatter(*xy[~legit & ~mixed].T, s=3, c=DGA_COLOR, lw=0, alpha=0.7, rasterized=True)
        ax.scatter(*xy[mixed].T, s=9, c=INK, lw=0, rasterized=True)
        handles = [_dot(LEGIT_COLOR, "legitimate"), _dot(DGA_COLOR, "DGA, among its own family"),
                   _dot(INK, "DGA, among other\nDGA families"),
                   _dot(None, "named: the 3 pairs of DGA\nfamilies most mixed together", text=True)]
        other = p["other_family"].to_numpy()
        labels = []
        for a, b in space.pairs[["family_a", "family_b"]].head(3).itertuples(index=False):
            where = mixed & (((family == a) & (other == b)) | ((family == b) & (other == a)))
            labels.append((xy[where] if where.sum() >= 3 else xy[(family == a) | (family == b)],
                           f"{a} ↔ {b}"))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_facecolor(SURFACE)
    ax.autoscale_view()
    _name_on_map(ax, labels)
    return handles


def _panel_title(ax, title, subtitle, color):
    """A bold title, a dot of the model's color before it, and a subtitle under it."""
    ax.annotate("●", (0, 1), xycoords="axes fraction", xytext=(0, 27), textcoords="offset points",
                fontsize=13, color=color, va="center")
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(17, 27),
                textcoords="offset points", fontsize=12, fontweight="bold", color=INK,
                va="center")
    ax.annotate(subtitle, (0, 1), xycoords="axes fraction", xytext=(0, 9),
                textcoords="offset points", fontsize=9.5, color=INK_2, va="center")


def _name_on_map(ax, labels, fontsize=9):
    """Write each (points, text) of `labels` in ink near the middle of its points, moved
    (with a leader line) where it would cover a label already written."""
    fig = ax.figure
    box = ax.get_position()
    width = box.width * fig.get_figwidth() * 72     # the axes, in points
    height = box.height * fig.get_figheight() * 72
    placed = []
    to_axes = ax.transData + ax.transAxes.inverted()
    for points, text in labels:
        if not len(points):
            continue
        anchor = to_axes.transform(np.median(points, 0))
        w, h = 0.58 * fontsize * len(text) / width, 1.5 * fontsize / height
        moves = [(0, 0.03), (0, -0.07), (0, 0.11), (0, -0.13), (0, 0.17), (0, -0.19),
                 (0.22, 0.03), (-0.22, 0.03), (0.22, -0.07), (-0.22, -0.07)]
        for i, (dx, dy) in enumerate(moves):
            x = min(max(anchor[0] + dx, w / 2 + 0.01), 0.99 - w / 2)
            y = min(max(anchor[1] + dy, h / 2 + 0.01), 0.99 - h / 2)
            if all(abs(x - px) > (w + pw) / 2 or abs(y - py) > (h + ph) / 2
                   for px, py, pw, ph in placed):
                break
        placed.append((x, y, w, h))
        line = None if i == 0 else {"arrowstyle": "-", "color": INK_2, "lw": 0.8, "shrinkA": 5,
                                    "shrinkB": 1}   # a leader line only when moved away
        ax.annotate(text, anchor, xycoords="axes fraction", xytext=(x, y),
                    textcoords="axes fraction", ha="center", va="center", fontsize=fontsize,
                    fontweight="bold", color=INK, zorder=5, arrowprops=line,
                    # a light box: the points under the text do not show through, and the
                    # leader line ends at its edge
                    bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.8},
                    path_effects=[patheffects.withStroke(linewidth=3, foreground=SURFACE)])


def _dot(color, label, hollow=False, text=False):
    """A legend entry: a dot of `color`, an ink ring if `hollow`, a bold "A" if `text`."""
    if text:
        return Line2D([], [], ls="", marker="$\\mathbf{A}$", ms=8, color=INK, label=label)
    return Line2D([], [], ls="", marker="o", ms=6, label=label, color=INK if hollow else color,
                  markerfacecolor="none" if hollow else color, markeredgewidth=0.9)


def _dumbbell(ax, table, colors, top, title, xlabel):
    """The `top` rows of `table` (rows: items, columns: models, values: shares), worst
    first: one marker per model on a line, so the gap between the models reads directly."""
    table = table.loc[table.max(1).sort_values(ascending=False).index[:top]]
    y = np.arange(len(table))
    ax.hlines(y, table.min(1), table.max(1), color=GRID, lw=2.5, zorder=1)
    for name in table.columns:
        ax.scatter(table[name], y, s=64, color=colors[name], edgecolor=SURFACE, lw=2, zorder=3,
                   label=name)
    ax.set_yticks(y, table.index, fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, min(1, table.to_numpy().max() * 1.12 + 0.02))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.tick_params(axis="y", labelcolor=INK)   # the names are read, the % axis is chrome
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(f"{xlabel} (lower is better)", color=INK_2, fontsize=9.5)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.9, edgecolor=GRID,
              labelcolor=INK_2)
