# Blood Fate Lab

An educational research baseline for predicting the **observed day-6 neutrophil fraction among neutrophils plus monocytes within a clone**, using that clone's day-2 RNA expression. Mouse progenitor data, not human therapy predictions. No novelty or intervention-effect claims.

## Quick start (Windows / VS Code)

Install Python 3.11 or 3.12. Open this folder in VS Code. In PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python train.py --download --audit-only
.\.venv\Scripts\python train.py
.\.venv\Scripts\python -m unittest -v
```

Mac/Linux: use `python3 -m venv .venv`, then `.venv/bin/python` instead.
The expression download alone is about 2.1 GB compressed; allow several GB of free disk space and RAM. The loader streams the file and retains only early expression. Downloads may be large and need a working internet connection. Existing files are reused; failed partial downloads are discarded. Never add data or generated models to GitHub. The script does not run CoSpar or require a GPU.

## What happens

1. Download four original processed Experiment 1 files: normalized RNA counts, genes, clone memberships, metadata.
2. Validate alignment and report time points, labels, eligible clones and file SHA256 checksums.
3. Exclude all clone columns involved in any multiply-assigned cell, preventing shared-cell leakage.
4. Require day-2 cells and at least five sampled day-6 neutrophils/monocytes per clone.
5. Average log1p expression across day-2 cells of each eligible clone. No later expression enters the inputs.
6. Hold out 25% of clones using fixed seed 42. Fit variance filtering and selection of up to 500 genes on training clones only.
7. Fit regularized fractional logistic regression. Each clone contributes total weight one; its two class weights equal the observed fractions. C=0.1 is fixed, not tuned on test outcomes.
8. Compare clone-macro MAE, RMSE, and fractional cross-entropy against the mean training-clone proportion.

`--min-late 5`, `--seed 42`, `--data-dir data`, `--output-dir results` are configurable. Do not repeatedly choose parameters by looking at test performance.

## Outputs

- `audit.json`: dimensions, labels, eligibility and provenance.
- `clone_outcomes.csv`: observed targets, early/late sample sizes, coverage and train/test assignment.
- `test_predictions.csv`: held-out forecasts and baseline.
- `metrics.json`: evaluation; lower is better for every reported metric.
- `model.joblib`: transforms, classifier and gene ordering. Only load trusted joblib files.

## Biological interpretation and limits

A barcode links relatives, not repeated measurements of the same living cell. A sampled day-2 cell is destroyed. Later siblings provide an imperfect proxy for lineage outcomes.

The target is conditional on the two selected cell types. Other outcomes are not errors and are excluded from the denominator; `target_lineage_coverage` exposes this selection. A 0.7 prediction means approximately 70% neutrophils among sampled neutrophils/monocytes, not a 70% chance of producing a neutrophil among ALL descendants. Fractions reflect fate, proliferation, survival and sampling together.

Each clone is one statistical example. Thousands of cells are not thousands of independent training examples. Small clones have noisy targets, and requiring observed descendants creates selection bias. The model is exploratory, not a causal predictor or a validated estimate of individual-cell fate.

The split tests unseen clones within pooled experiments. It does NOT test unseen donors, laboratories or culture conditions. Library, starting population and well can confound results. Before research claims: recover independent experiment identifiers, perform experiment-held-out validation, add clone bootstrap confidence intervals, assess sample-size sensitivity, compare published methods, and keep preprocessing within training folds. Gene associations are not causal drivers.

## Learning roadmap

1. Explain the audit and trace one clone manually.
2. Understand the conditional target and why mixed outcomes are retained.
3. Reproduce the baseline and inspect its errors.
4. Evaluate donor/experiment transfer and novelty before adding complexity.
5. Investigate the separately published cytokine perturbation experiment for intervention research; do not mix it silently into this baseline.

## Sources

Weinreb et al. (2020), *Lineage tracing on transcriptional landscapes links state to fate during differentiation*, Science 367, eaaw3381. https://doi.org/10.1126/science.aaw3381

Original data description and downloads:
https://github.com/AllonKleinLab/paper-data/tree/master/Lineage_tracing_on_transcriptional_landscapes_links_state_to_fate_during_differentiation

GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140802

CoSpar tutorial (separate processed subset, not the exact full matrix used here):
https://cospar.readthedocs.io/en/latest/20210121_all_hematopoietic_data_v3.html

Respect original dataset terms and cite its authors. Data are not redistributed here.

## Publish under Fainahehe

Sign in to GitHub as Fainahehe. Create an empty **private** repository named `blood-fate-lab` at https://github.com/new (do not initialize README/license/gitignore). Then in this folder:

```powershell
git init -b main
git add README.md requirements.txt train.py test_baseline.py .gitignore VALIDATION.md
git commit -m "Add clone-level blood fate prediction baseline"
git remote add origin https://github.com/Fainahehe/blood-fate-lab.git
git push -u origin main
```

If Git asks for identity, configure your own GitHub name and email. Authenticate through Git's browser flow; never paste access tokens into chat. Share the repository URL and grant the connected GitHub account access to continue collaborating.
