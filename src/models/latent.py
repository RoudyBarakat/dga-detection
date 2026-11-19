"""The latent space of the trained CNNs: each domain as their output layer sees it.

Both CNNs end with a dense layer (`model.dense`). Its output is the latent vector of a
domain: 15 numbers in CharCNN (Model 2), 128 in CharCNN2 (Model 2b). The output layer
is linear and reads only this vector, so two domains with close latent vectors get
close predictions: if a DGA domain sits among legitimate ones, or among another DGA
family, the model cannot tell them apart.

    sample = sample_domains(val)              # the same domains for every model
    space = LatentSpace(embed(model, encode(sample["domain"])), sample)
    space.scores, space.families, space.pairs, space.xy

Who is next to whom is measured in the full latent space, by the k nearest neighbours
of each domain (Euclidean distance, as the output layer sees the vector). The 2-D map
(t-SNE) is only a picture of it: it keeps the neighbours but not the distances, so the
numbers, not the map, are the evidence. plots.py draws them (plot_latent,
plot_conflicts); verdict() and actions() turn them into tables.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

K = 15   # neighbours per domain


def sample_domains(df, per_family=100, seed=0):
    """Up to `per_family` domains of every DGA family, and as many legitimate domains.

    Every family is seen, even the smallest, and DGA and legitimate weigh the same, so
    a share of legitimate neighbours means the same for every family.
    """
    dga = df[df["label"] == "dga"].sample(frac=1, random_state=seed)
    dga = dga.groupby("family").head(per_family)   # shuffled: a random sample of each family
    legit = df[df["label"] == "legit"].sample(len(dga), random_state=seed)
    return pd.concat([dga, legit]).reset_index(drop=True)[["domain", "label", "family"]]


@torch.no_grad()
def embed(model, x, batch_size=1024):
    """The latent vector of every encoded domain in `x`: the output of `model.dense`."""
    device = next(model.parameters()).device
    model.eval()
    out = []
    hook = model.dense.register_forward_hook(lambda module, inputs, z: out.append(z.cpu().numpy()))
    try:
        for start in range(0, len(x), batch_size):
            model(torch.as_tensor(x[start:start + batch_size]).to(device))
    finally:
        hook.remove()
    return np.concatenate(out)


def project(z, seed=0):
    """A 2-D map of the latent vectors (t-SNE, after PCA to at most 50 dimensions)."""
    if z.shape[1] > 50:
        z = PCA(50, random_state=seed).fit_transform(z)
    return TSNE(2, init="pca", perplexity=30, random_state=seed).fit_transform(z)


def _neighbours(z, k):
    """Indices of the k nearest neighbours of every row of z, the row itself left out."""
    idx = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z, return_distance=False)
    is_self = idx == np.arange(len(z))[:, None]
    is_self[~is_self.any(1), -1] = True   # ties put the row elsewhere: drop the farthest
    return idx[~is_self].reshape(len(z), k)


class LatentSpace:
    """The latent vectors `z` of the domains of `sample`, and who is next to whom.

    points:   per domain, the share of its k neighbours that are legitimate, of its own
              family and of other DGA families, and the DGA family most of its DGA
              neighbours come from.
    families: per DGA family, the mean of those shares over its domains, sorted by the
              share of legitimate neighbours.
    pairs:    per pair of DGA families A, B: the share of A's neighbours that are of
              family B and of B's that are of family A, averaged; sorted.
    scores:   label separation (share of neighbours with the same label, DGA or
              legitimate) and family separation (share of a DGA domain's neighbours of
              its own family, averaged per family).
    xy:       the 2-D map, for the plots.
    """

    def __init__(self, z, sample, k=K, seed=0):
        self.z, self.sample, self.k = z, sample.reset_index(drop=True), k
        idx = _neighbours(z, k)
        label = self.sample["label"].to_numpy()
        family = self.sample["family"].to_numpy()
        is_dga = label == "dga"
        nb_family, nb_dga = family[idx], label[idx] == "dga"
        same_label = (label[idx] == label[:, None]).mean(1)
        legit_share = (~nb_dga).mean(1)
        own_share = (nb_family == family[:, None]).mean(1)
        # The DGA family most of a domain's DGA neighbours (of another family) come from.
        other = np.where(nb_dga & (nb_family != family[:, None]), nb_family, "")
        top_other = []
        for row in other:
            names, counts = np.unique(row[row != ""], return_counts=True)
            top_other.append(names[counts.argmax()] if len(names) else "")
        self.points = self.sample.assign(
            same_label_share=same_label, legit_share=legit_share, own_family_share=own_share,
            other_dga_share=(other != "").mean(1), other_family=top_other)

        dga = self.points[is_dga]
        self.families = (dga.groupby("family")
                         .agg(n=("domain", "size"), legit_share=("legit_share", "mean"),
                              own_family_share=("own_family_share", "mean"))
                         .sort_values("legit_share", ascending=False))

        # Share of family A's neighbours that are of DGA family B, for every A != B.
        rows, cols = np.repeat(family[is_dga], k), nb_family[is_dga].ravel()
        keep = nb_dga[is_dga].ravel() & (rows != cols)
        share = pd.crosstab(rows[keep], cols[keep])
        share = share.reindex(index=self.families.index, columns=self.families.index,
                              fill_value=0).div(self.families["n"] * k, axis=0)
        names = np.array(sorted(share.index))
        sym = ((share + share.T) / 2).loc[names, names].to_numpy()
        a, b = np.triu_indices(len(names), 1)
        self.pairs = (pd.DataFrame({"family_a": names[a], "family_b": names[b], "share": sym[a, b]})
                      .query("share > 0").sort_values("share", ascending=False)
                      .reset_index(drop=True))
        self.pairs["pair"] = self.pairs["family_a"] + "  ↔  " + self.pairs["family_b"]

        self.scores = {"label_separation": same_label.mean(),
                       "family_separation": self.families["own_family_share"].mean()}
        self.xy = project(z, seed)


def verdict(spaces, families=True):
    """The scores of every model of `spaces` ({name: LatentSpace}), and the better model."""
    columns = {"label_separation": "DGA vs legitimate: neighbours with the same label"}
    if families:
        columns["family_separation"] = "DGA families: neighbours of the same family"
    table = pd.DataFrame({name: s.scores for name, s in spaces.items()}).T[list(columns)]
    best = table.idxmax()
    table = table.map(lambda v: f"{v:.1%}")
    table.loc["better"] = best
    return table.rename(columns=columns)


def _name_kind(domains):
    """How the names of a family look: made of words, short, or random-looking."""
    names = domains.str.split(".").str[0]
    letters = names.str.replace(r"[^a-z]", "", regex=True)
    vowels = letters.str.count(r"[aeiouy]").sum() / max(letters.str.len().sum(), 1)
    if names.str.len().median() <= 8:
        return "short"
    return "words" if vowels >= 0.3 else "random"


def actions(spaces, families=True, top=5, threshold=0.2):
    """What the latent spaces show, and what to do about it: one row per finding.

    The findings are those of the model that separates DGA from legitimate domains
    best: the DGA families with more than `threshold` of legitimate neighbours, the
    legitimate domains that sit among DGA domains and (if `families`) the family pairs
    sharing more than `threshold` of their neighbours; at most `top` of each kind, with
    the most mixed domains as examples. Returns (that model's name, the table).
    """
    name = max(spaces, key=lambda n: spaces[n].scores["label_separation"])
    p = spaces[name].points
    rows = []
    for family, row in spaces[name].families.head(top).iterrows():
        if row["legit_share"] <= threshold:
            break
        mine = p[p["family"] == family].sort_values("legit_share", ascending=False)
        kind = _name_kind(mine["domain"])
        median = mine["domain"].str.split(".").str[0].str.len().median()
        others = "; ".join(f"{n}: {s.families.loc[family, 'legit_share']:.0%}"
                           for n, s in spaces.items() if n != name)
        rows.append({
            "conflict": "DGA family among legitimate domains",
            "who": family,
            "how much": f"{row['legit_share']:.0%} of its neighbours are legitimate"
                        + (f" ({others})" if others else ""),
            "examples": ", ".join(mine["domain"].head(3)),
            "what to do": {
                "words": "Its names are made of real words, as legitimate names are: the "
                         "characters cannot tell them apart. Add word-level features "
                         "(dictionary words, word n-grams) or more of its examples; expect "
                         "low recall for it until then.",
                "short": f"Its names are short (median {median:.0f} characters), as short as "
                         "many legitimate names: too few characters to tell them apart. A "
                         "character model alone will keep missing some: add features beyond "
                         "the name (e.g. DNS behaviour), or accept the lower recall.",
                "random": "Its names have the character patterns of legitimate names: look at "
                          "the examples; more of its examples in training may help.",
            }[kind],
        })
    is_legit = p["label"] == "legit"
    lost = p[is_legit & (p["legit_share"] < 0.5)].sort_values("legit_share")
    if len(lost):
        near = lost["other_family"].replace("", np.nan).value_counts().head(3).index
        rows.append({
            "conflict": "legitimate domains among DGA domains",
            "who": f"{len(lost) / is_legit.sum():.1%} of legitimate domains",
            "how much": "most of their neighbours are DGA, mostly " + ", ".join(near),
            "examples": ", ".join(lost["domain"].head(3)),
            "what to do": "Likely false alarms: legitimate names that look generated. Check "
                          "them against an allow-list of known domains, or look at the "
                          "precision/recall trade-off of the 0.5 threshold.",
        })
    if families:
        for a, b, share in spaces[name].pairs[["family_a", "family_b", "share"]].head(top) \
                .itertuples(index=False):
            if share <= threshold:
                break
            ex = [p.loc[(p["family"] == f) & (p["other_family"] == g)].sort_values(
                "other_dga_share", ascending=False)["domain"].head(1) for f, g in ((a, b), (b, a))]
            rows.append({
                "conflict": "two DGA families mixed together",
                "who": f"{a} ↔ {b}",
                "how much": f"{share:.0%} of their neighbours are of the other family",
                "examples": " | ".join(e.iloc[0] for e in ex if len(e)),
                "what to do": "Both are DGA, so DGA vs legitimate is not affected. For "
                              "family attribution: merge the two classes, or check whether "
                              "they are the same generator under two names.",
            })
    return name, pd.DataFrame(rows)
