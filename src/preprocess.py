"""
Preprocessing Script for ACIS Insurance Risk Analytics
Handles:
- Missing values
- Outlier treatment
- Skew correction
- Feature formatting
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore

def load_data(path):
    return pd.read_csv(path)

def handle_missing(df):
    df["CustomValueEstimate"] = df["CustomValueEstimate"].fillna(df["CustomValueEstimate"].median())
    df["Cylinders"] = df["Cylinders"].fillna(df["Cylinders"].median())
    return df

def treat_outliers(df):
    numeric_cols = ["TotalClaims", "TotalPremium", "CustomValueEstimate"]
    for col in numeric_cols:
        z_scores = zscore(df[col])
        df[col] = np.where(np.abs(z_scores) > 3,
                           df[col].median(),
                           df[col])
    return df

def fix_skew(df):
    skew_cols = ["TotalClaims", "CustomValueEstimate"]
    for col in skew_cols:
        df[col] = np.log1p(df[col])
    return df

def preprocess(path_in, path_out):
    df = load_data(path_in)
    df = handle_missing(df)
    df = treat_outliers(df)
    df = fix_skew(df)
    df.to_csv(path_out, index=False)
    print(f"Saved cleaned data to {path_out}")

if __name__ == "__main__":
    preprocess("data_raw/historical_insurance_data.csv",
               "data/processed_data.csv")
