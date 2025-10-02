import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Lab-based Diabetes Risk", layout="centered")

st.title("Predict Diabetes Risk from Routine Labs (Pima MVP)")
st.caption("Educational demo – not a diagnostic tool.")

model_path = Path("artifacts/calibrator.joblib")
feat_path = Path("artifacts/feature_list.joblib")

if not model_path.exists():
    st.warning("No trained model found. Train first: `python -m src.train --csv data/pima_sample.csv --outdir artifacts`")
else:
    st.success("Model found. Upload a CSV to get risk predictions.")

uploaded = st.file_uploader("Upload CSV with Pima schema (Outcome optional):", type=["csv"])

if uploaded and model_path.exists():
    df = pd.read_csv(uploaded)
    calib = joblib.load(model_path)
    features = joblib.load(feat_path)

    # Minimal feature engineering to match training pipeline:
    # replicate ZeroToNaN and add_feature_columns behavior inline for app
    for c in ["BloodPressure","SkinThickness","Insulin"]:
        if c in df:
            df.loc[df[c]==0, c] = np.nan
    # engineered
    df["BMI_over_Age"] = df["BMI"] / (df["Age"].replace(0, np.nan))
    df["HighGlucoseFlag"] = (df["Glucose"] >= 126).astype(int)
    for c in ["BloodPressure","SkinThickness","Insulin"]:
        if c in df:
            df[f"Missing_{c}"] = df[c].isna().astype(int)

    X = df[[c for c in features if c in df.columns]].copy()
    prob = calib.predict_proba(X)[:,1]
    out = df.copy()
    out["risk"] = prob
    out["risk_tier"] = pd.cut(out["risk"], bins=[-1,0.05,0.20,1.0], labels=["Low","Moderate","High"])

    st.subheader("Predictions")
    st.dataframe(out.head(50))

    st.download_button("Download predictions CSV", out.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
