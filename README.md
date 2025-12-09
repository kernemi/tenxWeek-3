# tenxWeek-3: End-to-End Insurance Risk Analytics & Predictive Modeling

## Business objective (explicit):

Help AlphaCare Insurance Solutions (ACIS) optimize marketing and product strategy by identifying low-risk customer segments and providing a reproducible, auditable analytics pipeline that supports risk-based premium adjustments (i.e., reduce premiums for verified low-risk segments and target marketing to those segments).

This repository contains everything for Task 1 (environment, data collection & preprocessing) and Task 2 (DVC-based reproducible data pipeline). It also documents how Task 3 (hypothesis testing) and Task 4 (modeling & interpretability) are integrated and reproduced.

### Contents / high-level overview
```
tenxweek-3/
│
├── README.md                        # <-- you are here
├── requirements.txt                 # Python dependencies
├── environment.yml                  # optional conda environment
├── .gitignore
├── .dvc/                            # DVC metadata (created by dvc init)
├── dvc.yaml                         # DVC pipeline stages (preprocess, eda, etc.)
├── data_raw/                        # raw datasets (git-ignored, DVC-tracked)
├── data/                            # small reference or processed files (DVC outputs)
├── notebooks/                       # interactive EDA and hypothesis notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Hypothesis_Tests.ipynb
│   └── 04_Modeling.ipynb
├── src/                             # production-style scripts & modules
│   ├── data/
│   │   └── load_data.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── stat_tests.py
│   ├── data_pipeline.py
│   ├── train_models.py
│   └── interpretability.py
├── visualizations/                  # png / svg outputs from notebooks/scripts
├── models/                          # saved model artifacts (.joblib)
├── reports/
│   ├── interim/
│   │   └── hypothesis_results.md
│   └── final/
│       └── final_blogpost.md
├── experiments/                     # DVC/experiment outputs
└── .github/
    └── workflows/
        └── ci.yml                   # CI workflow for linting/tests/dvc checks
```

### Quick start (reproduce locally)

Requirements: Python 3.10+ (recommended). This assumes you have cloned the repository and have DVC installed if you plan to pull data.
```
python -m venv .venv
source .venv/bin/activate        # mac/linux
.venv\Scripts\activate           # windows
pip install -r requirements.txt
```
DVC: fetch versioned data (if remote configured)
```
# initialize only if not already done (repo already has .dvc)
# dvc init

# pull data files tracked by DVC (local or remote must be configured)
dvc pull
```
Run preprocessing (recommended using DVC pipeline)
```
# reproduce all pipeline stages defined in dvc.yaml (preprocess -> eda -> etc.)
dvc repro
```
Run EDA notebook (interactive)
```
Open notebooks/01_EDA.ipynb in Jupyter / JupyterLab and run cells to view descriptive statistics, missing-value summaries, boxplots, creative plots (loss ratio by province, month × province heatmap, scatter+marginal hist).
```
Run hypothesis tests (Task 3)
```
python src/stat_tests.py
# outputs:
# - reports/interim/hypothesis_results.md
# - reports/interim/hypothesis_results.csv
```
Train models & evaluate (Task 4)
```
python src/train_models.py       # trains classifier + severity models and stores in models/
python src/evaluate_models.py    # evaluates on holdout & writes reports/final metrics
python src/interpretability.py   # produces SHAP summary plot and top features csv
```
Generate premium predictions
```
python src/predict_premium.py    # uses saved models to compute risk-based premiums
```
## Task 1: Setup and Data Collection 
-Created repo structure and README.md.

-Installed and validated Python environment (requirements.txt, environment.yml).

-Implemented modular code in src/:

    load_data.py (safe CSV loading & preview)
    
    preprocess.py (missing-value handling, outlier treatment, log/winsorize)

    eda.py (descriptive stats, missing-value summary, visual outputs)

- Configured CI (GitHub Actions) to:

    install dependencies

    run linting and unit tests

    run lightweight DVC checks (exists / dvc status) on PRs

- Documented reproducible development & commit policy.

## Task 2: DVC — Reproducible & Auditable Data Pipeline
Goal: reproducible, auditable pipeline for data ingestion & preprocessing.

### What I implemented

dvc init (DVC metadata in .dvc/)

Local remote example configured (dvc remote add -d local_storage data/.dvc_storage) — replaceable with S3/GDrive/Azure
```
dvc add data_raw/historical_insurance_data.csv
git add data_raw/historical_insurance_data.csv.dvc .gitignore
git commit -m "Track raw insurance data with DVC"
dvc push
```
### Created dvc.yaml pipeline stages for:

preprocess → src/preprocess.py → data/processed_data.csv

eda → src/eda.py → visualizations/*, reports/interim/*

(optional) train → src/train_models.py → models/* (may be large, consider DVC remote)

### How to reproduce

- dvc pull to fetch inputs

- dvc repro to re-run pipeline

- dvc exp run / dvc exp show for experiment tracking

### Outcome

Raw data is DVC-tracked (not in Git), processing steps are versioned and reproducible, and experiments can be compared.

## Task 3 (Hypothesis testing) — pointer

-Notebook: notebooks/02_Hypothesis_Tests.ipynb

-Script: src/stat_tests.py — runs:

    Claim frequency (has_claim) tests across Province and top PostalCodes (Chi-square)

    Claim severity tests (Kruskal-Wallis / Mann-Whitney)

    Margin differences (ANOVA/Kruskal-Wallis)

    Gender two-proportion z-test

- Outputs: reports/interim/hypothesis_results.md with p-values, effect sizes, sample sizes and business recommendations (e.g., provinces that may warrant premium adjustment or targeted marketing).

## Task 4 (Modeling & interpretability) — pointer

- Data pipeline: src/data_pipeline.py (load/clean/feature_engineer)

- Modeling scripts:

    src/train_models.py — training classifier(s) for has_claim and regressor(s) for TotalClaims (severity)

    src/evaluate_models.py — compute AUC, Precision@k, Recall, F1, RMSE, MAE, R² and save CSV reports

    src/interpretability.py — SHAP summary, top 10 features exported to reports/final/top10_shap.csv and reports/final/shap_summary.png

- Premium formula implemented in src/predict_premium.py:
```
predicted_premium = predicted_prob_claim * predicted_claim_severity + expense_loading + profit_margin
```

- All models saved under models/ (joblib) and model metadata persisted (feature list, preprocessor).

Model interpretability is an explicit deliverable — SHAP plots, feature importance, and plain-language explanations for the top 5–10 features are included in the final report.

## Branching & submission workflow

- Feature branches for each task:

    task-1 — Setup & EDA

    task-2 — DVC/data pipeline

    task-3 — Hypothesis testing

    task-4 — Modeling & interpretability

- Commit frequently (recommended: 3 commits/day for active work).

- Create Pull Requests to main for merges; CI will run and check linting/tests/DVC status.

- Final submission: ensure main contains merged results for Task 1–4 and reports/final/final_blogpost.md is complete.

## CI / Quality checks

.github/workflows/ci.yml runs on PRs and will:

    install dependencies

    run unit tests (pytest)

    run lightweight DVC checks (dvc status)

    optionally run notebook execution in headless mode (nbconvert) for reproducibility

## Next steps & optional improvements

- Add automated model monitoring (drift detection) and continuous training DVC stage for regular re-training.

- Add A/B test design for proposed premium changes and implement power calculations before rollout.

- Expand experiment tracking with DVC or MLflow to save hyperparameters and model metrics more formally.
