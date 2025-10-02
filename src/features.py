from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PIMA_COLS = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
    "BMI","DiabetesPedigreeFunction","Age"
]

BIO_ZERO_MISSING = ["BloodPressure","SkinThickness","Insulin"]  # 0 is implausible/rare → treat as missing

class ZeroToNaN(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            if c in X:
                X.loc[X[c] == 0, c] = np.nan
        return X

def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ratios & simple transforms
    # (Pima lacks lipids, so we keep it lean)
    df["BMI_over_Age"] = df["BMI"] / (df["Age"].replace(0, np.nan))
    df["HighGlucoseFlag"] = (df["Glucose"] >= 126).astype(int)
    # missingness indicators for BIO_ZERO_MISSING
    for c in BIO_ZERO_MISSING:
        if c in df:
            df[f"Missing_{c}"] = df[c].isna().astype(int)
    return df

def risk_tier(p: float) -> str:
    if p < 0.05: return "Low"
    if p < 0.20: return "Moderate"
    return "High"
