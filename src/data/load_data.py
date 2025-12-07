"""
Module for loading, validating, and inspecting insurance datasets
for ACIS Insurance Risk Analytics Challenge.

"""
import pandas as pd
import os

def load_csv(path: str) -> pd.DataFrame:
 
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path)
    return df

def preview_data(df: pd.DataFrame, n: int = 5):
    
    print("\n=== DATA PREVIEW ===")
    print(df.head(n))
    
    print("\n=== DATA INFO ===")
    print(df.info())
    
    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(df.describe())
    
    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())

def main():
  
    data_path = "data_raw/MachineLearningRating_v3.csv"
    df = load_csv(data_path)
    preview_data(df)

if __name__ == "__main__":
    main()
