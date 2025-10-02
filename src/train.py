from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import joblib

from .features import PIMA_COLS, ZeroToNaN, add_feature_columns, risk_tier
from .models import build_preprocess, build_baseline_model, build_xgb_model

def load_csv(path):
    df = pd.read_csv(path)
    # Treat biologically implausible zeros as missing
    df = ZeroToNaN(cols=["BloodPressure","SkinThickness","Insulin"]).transform(df)
    df = add_feature_columns(df)
    return df

def train(csv, outdir, model_type="xgb", test_size=0.2, seed=42, save_shap=False):
    os.makedirs(outdir, exist_ok=True)
    df = load_csv(csv)
    y = df["Outcome"].astype(int)
    X = df.drop(columns=["Outcome"])

    features = [c for c in X.columns]  # keep all engineered columns
    pre = build_preprocess()
    if model_type == "lr":
        clf = build_baseline_model()
    else:
        clf = build_xgb_model(seed)

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    X_tr, X_te, y_tr, y_te = train_test_split(X[features], y, test_size=test_size, stratify=y, random_state=seed)

    pipe.fit(X_tr, y_tr)

    # Calibration
    calib = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    calib.fit(X_tr, y_tr)
    proba = calib.predict_proba(X_te)[:,1]

    metrics = {
        "auroc": float(roc_auc_score(y_te, proba)),
        "prauc": float(average_precision_score(y_te, proba)),
        "brier": float(brier_score_loss(y_te, proba))
    }

    joblib.dump(pipe, os.path.join(outdir, "model.joblib"))
    joblib.dump(calib, os.path.join(outdir, "calibrator.joblib"))
    joblib.dump(features, os.path.join(outdir, "feature_list.joblib"))

    # Save example predictions
    pred_df = X_te.copy()
    pred_df["y_true"] = y_te.values
    pred_df["risk"] = proba
    pred_df["risk_tier"] = pred_df["risk"].apply(risk_tier)
    pred_df.to_csv(os.path.join(outdir, "example_predictions.csv"), index=False)

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics:", metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/pima_sample.csv")
    ap.add_argument("--outdir", type=str, default="artifacts")
    ap.add_argument("--model_type", type=str, default="xgb", choices=["xgb","lr"])
    ap.add_argument("--save_shap", action="store_true")
    args = ap.parse_args()
    train(args.csv, args.outdir, model_type=args.model_type, save_shap=args.save_shap)
