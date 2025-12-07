# tenxWeek-3: End-to-End Insurance Risk Analytics & Predictive Modeling

## Overview
This repository contains all code, data, and configuration files for:

Task 1: Environment Setup, Data Collection, and Preprocessing
Task 2: Establishing a reproducible, auditable data pipeline using Data Version Control (DVC), including data tracking, pipeline management, and experiment versioning

The project supports scalable analytics and predictive modeling for insurance risk analysis.

## Task 1: Setup and Data Collection 

### Folder Structure
```
.dvc/                # DVC internal metadata (automatically created)
experiments/         # Experiment artifacts (DVC or MLflow)
data/                # Small reference files
data_raw/            # Large raw files (git-ignored, tracked with DVC)
notebooks/           # Jupyter notebooks for EDA and profiling
src/                 # Python source code (models, preprocessing, helpers)
visualizations/      # Saved plots + EDA output

```
### Requirements
Python 3.10+, install dependencies:
```
pip install -r requirements.txt
```

### file/folder structure for Task 1
```
tenxweek-3/
│
├── README.md                  
├── requirements.txt           # Python dependencies
├── .gitignore
├── visualizations/
├── data_raw/
├── notebooks/
├── src/
├── experiments/
├── .dvc/
├── environment.yml
├── dvc.yml
```
### How to run
Open and run the EDA notebook:
```
notebooks/1_EDA.ipynb
```
This notebook contains descriptive statistics, missing value summaries, outlier detection, visualizations, and insights.

## Task 2: DVC — Reproducible & Auditable Data Pipeline
Below is the added section describing Task 2 contributions.
Establishing a Reproducible DVC Pipeline

The goal of Task 2 was to build a fully reproducible, trackable, and auditable data workflow using Data Version Control (DVC).

## What Was Done

1. DVC Initialization
Initialized DVC in the project:
```
dvc init
```
This created the .dvc/ directory and prepared the repository for data tracking.

2. Configure Data Remote Storage
Configured a local remote directory for storing dataset versions:
```
dvc remote add -d local_storage data/.dvc_storage
```
You may replace this later with AWS S3, Google Drive, or Azure Blob Storage.

3. Track Raw Data with DVC and commit the tracking file
Raw CSV files stored in data_raw/ are not committed to Git.
Instead, they are versioned through DVC:
```
dvc add data_raw/historical_insurance_data.csv
git add data_raw/historical_insurance_data.csv.dvc .gitignore
git commit -m "Track raw insurance data with DVC"
dvc push
```
4. Experiment Tracking
DVC experiments allow versioning of multiple modeling runs:
```
dvc exp run
dvc exp show
dvc exp diff
```
All hyperparameters, evaluation metrics, and outputs are tracked automatically.

## Outcome

- Data is securely versioned and not stored in Git
- Pipeline stages are reproducible using dvc repro
- Experiments are version-controlled and comparable
- Raw and processed datasets are auditable, supporting enterprise-grade workflows

