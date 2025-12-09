# src/evaluate_models.py
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.calibration import calibration_curve

def evaluate_classifier(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # Precision@k (top 5%): select top p% by predicted probability and measure precision
    k = 0.05
    cutoff = np.percentile(y_proba, 100*(1-k))
    top_idx = y_proba >= cutoff
    precision_at_k = y_test.iloc[top_idx].mean() if top_idx.sum() > 0 else np.nan
    return {"auc":auc, "precision":prec, "recall":rec, "f1":f1, "precision_at_5pct": precision_at_k}

def evaluate_regressor(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}
