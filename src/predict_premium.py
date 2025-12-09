# src/predict_premium.py
import joblib
import pandas as pd
import numpy as np

clf = joblib.load("models/best_classifier.joblib")
sev = joblib.load("models/best_severity.joblib")

def predict_risk_premium(df, expense_loading=50.0, profit_margin=0.1):
    X = df[feature_cols]  # load features consistent with training
    prob_claim = clf.predict_proba(X)[:,1]
    predicted_severity = sev.predict(X)  # for all rows; if severity model trained on claim-only, it will still produce preds
    risk_based_premium = prob_claim * predicted_severity + expense_loading + profit_margin * (prob_claim * predicted_severity)
    df["predicted_prob_claim"] = prob_claim
    df["predicted_severity"] = predicted_severity
    df["risk_based_premium"] = risk_based_premium
    return df
