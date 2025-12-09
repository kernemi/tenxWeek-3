# src/data_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path="data/processed_data.csv"):
    df = pd.read_csv(path, parse_dates=["TransactionMonth"], infer_datetime_format=True)
    return df

def clean_data(df):
    """Basic cleaning: consistent dtypes and missing value handling used earlier."""
    df = df.copy()
    # ensure numeric
    df["TotalClaims"] = pd.to_numeric(df["TotalClaims"], errors="coerce").fillna(0.0)
    df["TotalPremium"] = pd.to_numeric(df["TotalPremium"], errors="coerce").fillna(0.0)
    # fill small missing for categorical if any
    cat_cols = ["Province","VehicleType","Gender","PostalCode","Make","Model"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN").astype(str)
    # Create binary target
    df["has_claim"] = (df["TotalClaims"] > 0).astype(int)
    # margin
    df["margin"] = df["TotalPremium"] - df["TotalClaims"]
    # vehicle age (example; use current year or transaction year)
    df["VehicleAge"] = df["TransactionMonth"].dt.year - df["RegistrationYear"]
    df["VehicleAge"] = df["VehicleAge"].clip(lower=0).fillna(df["VehicleAge"].median())
    return df

def feature_engineer(df):
    df = df.copy()
    # numeric features to keep
    numeric = ["CustomValueEstimate","Kilowatts","Cubiccapacity","VehicleAge","TotalPremium"]
    for col in numeric:
        if col not in df.columns:
            df[col] = 0
    # interaction example
    df["value_per_kw"] = df["CustomValueEstimate"] / (df["Kilowatts"].replace(0, np.nan))
    df["value_per_kw"] = df["value_per_kw"].fillna(df["value_per_kw"].median())
    # target and features
    # Example categorical features we'd encode later
    categorical = ["Province","VehicleType","Gender","PostalCode","Make"]
    features = numeric + ["value_per_kw"] + categorical
    return df, features, categorical, numeric

def train_test_split_stratified(df, features, test_size=0.2, random_state=42):
    X = df[features]
    y = df["has_claim"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
