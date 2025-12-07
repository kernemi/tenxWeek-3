"""
EDA Script for ACIS Insurance Risk Analytics
Outputs:
- Descriptive statistics
- Missing value summary
- Correlation analysis
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda(data_path):
    df = pd.read_csv(data_path)

    # Descriptive stats
    desc = df.describe()
    desc.to_csv("visualizations/descriptive_stats.csv")

    # Missing values
    missing = df.isna().sum()
    missing.to_csv("visualizations/missing_values.csv")

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("visualizations/correlation_heatmap.png")

if __name__ == "__main__":
    run_eda("data/processed_data.csv")
