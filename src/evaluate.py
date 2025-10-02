from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
import joblib

def evaluate_curves(outdir):
    pred_path = os.path.join(outdir, "example_predictions.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError("Run training first to create example_predictions.csv")
    df = pd.read_csv(pred_path)
    y = df["y_true"].values
    p = df["risk"].values
    fpr, tpr, thr = roc_curve(y, p)
    pr, rc, thr2 = precision_recall_curve(y, p)
    curves = {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thr": thr.tolist()},
        "pr": {"precision": pr.tolist(), "recall": rc.tolist()}
    }
    with open(os.path.join(outdir, "curves.json"), "w") as f:
        json.dump(curves, f, indent=2)
    print("Saved curves.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="artifacts")
    args = ap.parse_args()
    evaluate_curves(args.outdir)
