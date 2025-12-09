# src/stat_tests.py
"""
Statistical tests for Task 3 (Hypothesis testing) - ACIS
Saves results to reports/interim/hypothesis_results.md and .csv
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, f_oneway
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttest_ind
import statsmodels.api as sm
import os
from typing import Tuple
from math import sqrt

OUTPUT_MD = "reports/interim/hypothesis_results.md"
OUTPUT_CSV = "reports/interim/hypothesis_results.csv"

def load_data(path="../data_raw/MachineLearningRating_v3.txt"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=['TransactionMonth'], infer_datetime_format=True)
    return df

def prepare_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['has_claim'] = (df['TotalClaims'] > 0).astype(int)
    df['claim_severity'] = df.loc[df['has_claim'] == 1, 'TotalClaims']
    df['margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

# ---------- Utility functions ----------
def cohen_d(x, y):
    """Cohen's d for two samples (unequal n)."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

def risk_ratio(a_success, a_total, b_success, b_total):
    """Risk ratio for binary outcome (A vs B)."""
    p_a = a_success / a_total
    p_b = b_success / b_total
    rr = p_a / p_b if p_b != 0 else np.nan
    return rr, p_a, p_b

def prop_conf_int(count, n, alpha=0.05):
    """Wald approximate CI for proportion."""
    p = count / n
    z = 1.96  # ~95%
    se = np.sqrt(p*(1-p)/n)
    return p, p - z*se, p + z*se

# ---------- Tests ----------
def chi2_test_by_province_claimfreq(df):
    ct = pd.crosstab(df['Province'], df['has_claim'])
    chi2, p, dof, expected = chi2_contingency(ct)
    return dict(test="chi2_province_claimfreq", chi2=chi2, p=p, dof=dof, shape=ct.shape)

def kruskal_test_claim_severity_by_province(df):
    groups = [g['TotalClaims'].values for _, g in df[df['has_claim'] == 1].groupby('Province')]
    if len(groups) < 2:
        return dict(test="kruskal_province_severity", stat=np.nan, p=np.nan)
    stat, p = kruskal(*groups)
    return dict(test="kruskal_province_severity", stat=stat, p=p, n_groups=len(groups))

def chi2_topk_postalcode(df, k=10):
    top_zips = df['PostalCode'].value_counts().nlargest(k).index
    sub = df[df['PostalCode'].isin(top_zips)]
    ct = pd.crosstab(sub['PostalCode'], sub['has_claim'])
    chi2, p, dof, _ = chi2_contingency(ct)
    return dict(test=f"chi2_top{str(k)}_postalcode", chi2=chi2, p=p, dof=dof)

def margin_anova_by_postalcode(df, k=10):
    top_zips = df['PostalCode'].value_counts().nlargest(k).index
    groups = [g['margin'].dropna().values for _, g in df[df['PostalCode'].isin(top_zips)].groupby('PostalCode')]
    if len(groups) < 2:
        return dict(test="anova_postalcode_margin", stat=np.nan, p=np.nan)
    stat, p = f_oneway(*groups)
    return dict(test="anova_postalcode_margin", stat=stat, p=p, n_groups=len(groups))

def gender_two_proportion_test(df):
    counts = df.groupby('Gender')['has_claim'].sum()
    nobs = df.groupby('Gender')['has_claim'].count()
    # ensure there are exactly two groups (e.g., 'F' and 'M') or pick two most frequent categories
    if len(counts) != 2:
        # pick top two
        order = nobs.sort_values(ascending=False).index[:2]
        counts = counts.reindex(order)
        nobs = nobs.reindex(order)
    stat, pval = proportions_ztest(count=counts.values, nobs=nobs.values)
    # effect: risk ratio
    rr, p_a, p_b = risk_ratio(counts.values[0], nobs.values[0], counts.values[1], nobs.values[1])
    return dict(test="gender_proportion_ztest", stat=float(stat), p=float(pval), rr=rr, p1=p_a, p2=p_b)

def gender_severity_test(df):
    # severity among those who claimed
    sub = df[df['has_claim'] == 1]
    groups = [g['TotalClaims'].values for _, g in sub.groupby('Gender')]
    if len(groups) < 2:
        return dict(test="gender_severity_mannwhitney", stat=np.nan, p=np.nan)
    stat, p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
    d = cohen_d(groups[0], groups[1])
    return dict(test="gender_severity_mannwhitney", stat=float(stat), p=float(p), cohen_d=float(d))

# ---------- Runner ----------
def run_all(path="data/processed_data.csv", save_csv=True, save_md=True):
    df = load_data(path)
    df = prepare_metrics(df)

    results = []
    # 1. provinces - frequency
    results.append(chi2_test_by_province_claimfreq(df))
    # 2. provinces - severity
    results.append(kruskal_test_claim_severity_by_province(df))
    # 3. postal codes top-10 - frequency
    results.append(chi2_topk_postalcode(df, k=10))
    # 4. postal codes top-10 - margin
    results.append(margin_anova_by_postalcode(df, k=10))
    # 5. gender - proportion
    results.append(gender_two_proportion_test(df))
    # 6. gender - severity
    results.append(gender_severity_test(df))

    df_res = pd.DataFrame(results)
    if save_csv:
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df_res.to_csv(OUTPUT_CSV, index=False)
    if save_md:
        write_md_report(df_res, OUTPUT_MD)
    return df_res

def write_md_report(df_res: pd.DataFrame, path):
    md = []
    md.append("# Hypothesis Testing Results — Interim\n")
    md.append("**Alpha (significance)** = 0.05\n\n")
    for _, row in df_res.iterrows():
        md.append(f"## {row['test']}\n")
        for k, v in row.items():
            if k == 'test': continue
            md.append(f"- **{k}**: {v}\n")
        # short interpretation
        p = row.get('p', None) or row.get('pval', None)
        if p is not None and not pd.isna(p):
            if float(p) < 0.05:
                md.append(f"> **Interpretation**: p = {p:.4f} < 0.05 → reject H0 (statistically significant)\n")
            else:
                md.append(f"> **Interpretation**: p = {p:.4f} >= 0.05 → fail to reject H0 (not statistically significant)\n")
        md.append("\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(md))
    print(f"Wrote markdown report to {path}")

if __name__ == "__main__":
    print("Running hypothesis tests...")
    res = run_all()
    print(res)
