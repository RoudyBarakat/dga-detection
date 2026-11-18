# Report draft: how to add the Model 1 (logistic regression) part

Build:  latexmk -pdf report.tex   (or upload this folder to Overleaf)

Every place to edit is marked in report.tex with a comment starting
  % >>> TEAMMATE (Model 1, logistic regression):
Comments do not print. Search for "TEAMMATE" to jump between them.

| Where | What to add |
|---|---|
| Abstract | one sentence with the Model 1 result |
| 1 Introduction, Contributions so far | one bullet with the Model 1 result |
| 2 Related Work, Table 1 ("This work" row) | "LR on features" in the Model column |
| 4 Methods, right after 4.1 | a new subsection for Model 1: the 74 features, out-of-fold n-gram/Markov fitting, standardisation, loss (weighted cross-entropy + L2, lambda = 1e-4), L-BFGS, stacking and PCA |
| 5 Experimental Setup, Training | one or two sentences on how Model 1 is trained |
| 6 Results, Table 3 | Model 1 rows (binary, family-first, Delta, stacking, PCA; with and without TLD) + one paragraph |
| 7 Discussion | the gain shrinks with flexibility (LR +0.019 vs CNN +0.0004); why stacking and PCA did not help |
| 8 Limitations | Model 1 failure case if it applies (dictionary DGAs, e.g. matsnu) |
| 10 Code | your notebook names |

Rules
1. Validation numbers only: Table 3 says all results are on the validation set.
2. Do not edit the shared parts: 3 Data, 4.1 Problem formulation, and the CNN subsections.
3. Every paper, dataset, library and model you use goes in refs.bib AND is cited in the text
   (scikit-learn is already there as pedregosa2011sklearn; add L-BFGS, stacking and PCA).
