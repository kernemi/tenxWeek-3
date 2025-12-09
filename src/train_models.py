# src/train_models.py
import joblib
import json
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
from sklearn.metrics import roc_auc_score, mean_squared_error

from src.data_pipeline import load_data, clean_data, feature_engineer, train_test_split_stratified
from sklearn.pipeline import Pipeline

MODEL_DIR = "models"

def train_classification(X_train, y_train, preprocessor):
    models = {
        "logreg": Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000))]),
        "rf": Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1))]),
        "xgb": Pipeline([("pre", preprocessor), ("clf", xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))]),
    }
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
        results[name] = {"cv_auc_mean": float(np.mean(scores)), "cv_auc_std": float(np.std(scores))}
    best_name = max(results, key=lambda k: results[k]["cv_auc_mean"])
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, f"{MODEL_DIR}/best_classifier_{best_name}.joblib")
    with open(f"{MODEL_DIR}/classification_results.json","w") as f:
        json.dump(results, f, indent=2)
    return best_name, best_model, results

def train_regression_severity(df, preprocessor):
    # only where claim occurred
    df_claims = df[df["has_claim"]==1].copy()
    if df_claims.shape[0] < 50:
        raise ValueError("Not enough claim rows to train severity model reliably.")
    X = df_claims.drop(columns=["TotalClaims"])
    y = df_claims["TotalClaims"]
    # select features consistently (we'll do preprocessor on similar columns)
    # For simplicity reuse X.to_dict() workflow; assume preprocessor works with same columns
    rf = Pipeline([("pre", preprocessor), ("reg", RandomForestRegressor(n_estimators=200, n_jobs=-1))])
    xgb_reg = Pipeline([("pre", preprocessor), ("reg", xgb.XGBRegressor())])
    models = {"rf": rf, "xgb": xgb_reg, "lin": Pipeline([("pre", preprocessor), ("reg", LinearRegression())])}
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        # use neg_root_mean_squared_error for scoring; sklearn has neg_root_mean_squared_error in newer versions
        scores = cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
        results[name] = {"cv_rmse_mean": float(-np.mean(scores)), "cv_rmse_std": float(np.std(scores))}
    best_name = min(results, key=lambda k: results[k]["cv_rmse_mean"])
    best_model = models[best_name]
    best_model.fit(X, y)
    joblib.dump(best_model, f"{MODEL_DIR}/best_severity_{best_name}.joblib")
    with open(f"{MODEL_DIR}/severity_results.json","w") as f:
        json.dump(results, f, indent=2)
    return best_name, best_model, results

def run_all():
    df = load_data("data/processed_data.csv")
    df = clean_data(df)
    df, features, categorical, numeric = feature_engineer(df)
    # choose high-card columns
    high_card = ["PostalCode"]
    preprocessor = build_preprocessor(categorical, numeric, high_card)  # include function from earlier
    X_train, X_test, y_train, y_test = train_test_split_stratified(df, features)
    clf_name, clf, clf_res = train_classification(X_train, y_train, preprocessor)
    sev_name, sev_model, sev_res = train_regression_severity(df, preprocessor)
    return {"classification": (clf_name, clf_res), "severity": (sev_name, sev_res)}

if __name__ == "__main__":
    print("Training models...")
    res = run_all()
    print(res)
