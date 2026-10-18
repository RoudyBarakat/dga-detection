# DGA Detection from Domain Names: Family-First or Direct Binary?

Course project, Introduction to Machine Learning, American University of Beirut.

**Team:** Roudy Barakat (rtb07@mail.aub.edu), Abdo Sebailli (ams86@mail.aub.edu)

Malware uses Domain Generation Algorithms (DGAs) to create many domains so infected
hosts can reach their attacker. This project detects DGA domains from the domain name
alone and asks: when the final decision is DGA vs. legitimate, is it better to first
learn the individual DGA families (then collapse the prediction), or to learn
DGA vs. legitimate directly?

## Repository layout

```
data/
  raw/          source datasets (not committed, see data/raw/README.md)
  processed/    built datasets (not committed, rebuilt by the scripts below)
src/
  data/         build, balance and explore the dataset; dataset.py loads and splits it
  models/       baseline.py: majority class (Model 0) and logistic regression (Model 1)
proposal/       LaTeX proposal (proposal.tex, refs.bib, figures/, proposal.pdf)
```

## Setup

Requires Python 3.10 or newer.

```
git clone https://github.com/RoudyBarakat/dga-detection.git
cd dga-detection
python -m venv .venv
.venv\Scripts\activate        # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Building the data

The datasets are not committed (their licenses differ). Download the three raw files
listed in [data/raw/README.md](data/raw/README.md) into `data/raw/`, then run from the
repository root:

```
python src/data/build_dataset.py     # raw sources -> data/processed/domains_merged.csv.gz
python src/data/balance_dataset.py   # merged -> data/processed/domains_balanced.csv.gz
python src/data/explore_data.py      # balanced -> proposal/figures/ (figure + stats.json)
```

`build_dataset.py` takes a few minutes. Every step uses a fixed seed (42), so the
output is reproducible.

`data/processed/domains_balanced.csv.gz` has three columns:

| column | values |
|---|---|
| `domain` | the full domain name, e.g. `kneemanualgirl.com` |
| `label` | `legit` or `dga` |
| `family` | `legit`, `other_dga` (the 59 families with fewer than 300 domains), or one of 102 DGA families (e.g. `cryptolocker`) |

It holds 1,999,335 domains: all 999,664 legitimate domains (Tranco top 1M) and
999,671 DGA domains (Mendeley 2025 + Kaggle 2021), with 300 to 29,120 domains per
DGA family. The cap of 29,120 is chosen so that DGA and legitimate domains are equal
in number.

## Loading the data

```python
from src.data.dataset import load_splits
train, val, test = load_splits()   # 80/10/10, stratified by family, fixed seed
```

Each split has the columns `domain`, `label` and `family`; `train` also has `weight`
(legitimate and DGA weigh half each; within DGA, proportional to 1/sqrt(family size)).

## Baseline

```
python -m src.models.baseline                     # binary setup, validation set
python -m src.models.baseline --setup family      # family-first, collapsed to DGA/legit
python -m src.models.baseline --no-tld            # ablation without the TLD
python -m src.models.baseline --sample 100000     # quick run on 100k training rows
```

It prints macro-F1, DGA precision and recall, ROC-AUC and recall per DGA family for
Model 0 and Model 1. Add `--test` only for the final results. On the full training
set, the family setup takes much longer than the binary one.

## Contributing

All changes go through a branch and a pull request; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Proposal

```
cd proposal
pdflatex proposal && bibtex proposal && pdflatex proposal && pdflatex proposal
```
