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
  processed/    domains_balanced.csv.gz: the dataset used in experiments
src/
  data/         scripts that build, balance and explore the dataset
proposal/       LaTeX proposal (proposal.tex, refs.bib, figures/, proposal.pdf)
papers/         related papers (PDF)
```

## Dataset

`data/processed/domains_balanced.csv.gz` has three columns:

| column | values |
|---|---|
| `domain` | the domain name, e.g. `kneemanualgirl.com` |
| `label` | `legit` or `dga` |
| `family` | `legit`, or one of 102 DGA families (e.g. `cryptolocker`) |

It holds 439,974 domains: 219,987 legitimate (Tranco top 1M) and 219,987 DGA
(Mendeley 2025 + Kaggle 2021), with 300 to 3,000 domains per DGA family.

## Setup

```
pip install -r requirements.txt
```

## Rebuilding the data

Put the raw files in `data/raw/` (see `data/raw/README.md`), then run from the
repository root:

```
python src/data/build_dataset.py     # raw sources -> data/processed/domains_merged.csv.gz
python src/data/balance_dataset.py   # merged -> data/processed/domains_balanced.csv.gz
python src/data/explore_data.py      # balanced -> proposal/figures/ (figure + stats.json)
```

`build_dataset.py` takes a few minutes. The balancing uses a fixed seed (42), so the
output is reproducible.

## Proposal

```
cd proposal
pdflatex proposal && bibtex proposal && pdflatex proposal && pdflatex proposal
```
