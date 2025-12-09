# src/interpretability.py
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import os

def explain_model(model_path, X_train, X_test):
    model = joblib.load(model_path)
    # If model is a Pipeline with preprocessor, we need to extract the final estimator and preprocessed X
    # For tree-based models, KernelExplainer on raw features can be slow; prefer TreeExplainer for xgboost/rf
    # We will attempt TreeExplainer if model.named_steps present and final estimator is tree-based
    preprocessor = None
    if hasattr(model, "named_steps"):
        pre = model.named_steps.get("pre", None)
        est = model.named_steps.get(list(model.named_steps.keys())[-1])
    else:
        est = model

    # transform X_test using preprocessor if available
    if pre is not None:
        X_test_tr = pre.transform(X_test)
        try:
            explainer = shap.Explainer(est)
            shap_values = explainer(X_test_tr)
        except Exception:
            explainer = shap.Explainer(est, X_test_tr)
            shap_values = explainer(X_test_tr)
    else:
        explainer = shap.Explainer(est, X_test)
        shap_values = explainer(X_test)

    os.makedirs("reports/final", exist_ok=True)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("reports/final/shap_summary.png", dpi=200)
    # compute mean(|shap|)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    # map back feature names: shap_values.feature_names or X_test.columns
    feature_names = shap_values.feature_names if hasattr(shap_values, "feature_names") else X_test.columns
    feat_imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    feat_imp = feat_imp.sort_values("mean_abs_shap", ascending=False).head(10)
    feat_imp.to_csv("reports/final/top10_shap.csv", index=False)
    return feat_imp
