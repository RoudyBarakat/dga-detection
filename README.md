# DGA Detection from Domain Names: Family-First or Direct Binary?

Course project, Introduction to Machine Learning, American University of Beirut.

**Team:** Roudy Barakat (rtb07@mail.aub.edu), Abdo Sebaali (ams86@mail.aub.edu)

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
notebooks/
  01_data_loading.ipynb   load, split and weight the dataset
  02_baseline.ipynb       majority class (Model 0) and logistic regression (Model 1)
proposal/       LaTeX proposal (proposal.tex, refs.bib, figures/, proposal.pdf)
```

## Setup

We use a conda environment with Python 3.12:

```
git clone https://github.com/RoudyBarakat/dga-detection.git
cd dga-detection
conda create -n ml_aub python=3.12
conda activate ml_aub
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

## Notebooks

After building the data, start Jupyter from the repository root and open the
notebooks in `notebooks/`:

```
jupyter notebook
```

- `01_data_loading.ipynb` loads the dataset with `src/data/dataset.py`: an 80/10/10
  train/validation/test split, stratified by family with a fixed seed, and sample
  weights (legitimate and DGA weigh half each; within DGA, proportional to
  1/sqrt(family size)). All models use this loader, so they see the same rows.
- `02_baseline.ipynb` trains Model 0 (majority class) and Model 1 (logistic regression
  on character 1-3-grams) in the binary and family-first setups, with and without
  the TLD, and reports macro-F1, DGA precision and recall, ROC-AUC and recall per DGA
  family on the validation set. It trains on a 100k-row sample by default; set
  `SAMPLE = None` for the full training set (the family setup then takes much longer).

## Contributing

All changes go through a branch and a pull request; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Proposal

```
cd proposal
pdflatex proposal && bibtex proposal && pdflatex proposal && pdflatex proposal
```
