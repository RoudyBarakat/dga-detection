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

`data/processed/domains_balanced.csv.gz` has five columns:

| column | values |
|---|---|
| `domain` | the full domain name, e.g. `kneemanualgirl.com` |
| `label` | `legit` or `dga` |
| `family` | `legit`, `other_dga` (the 59 families with fewer than 300 domains), or one of 102 DGA families (e.g. `cryptolocker`) |
| `split` | `train`, `val` or `test` (80/10/10, stratified by family) |
| `weight` | sample weight: legitimate and DGA weigh half each; within DGA, proportional to 1/sqrt(family size); mean 1 |

It holds 1,999,335 domains: all 999,664 legitimate domains (Tranco top 1M) and
999,671 DGA domains (Mendeley 2025 + Kaggle 2021), with 300 to 29,120 domains per
DGA family. The cap of 29,120 is chosen so that DGA and legitimate domains are equal
in number.

## Contributing

All changes go through a branch and a pull request; see [CONTRIBUTING.md](CONTRIBUTING.md).

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
