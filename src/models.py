from __future__ import annotations
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from .features import PIMA_COLS, ZeroToNaN, add_feature_columns

NUM_COLS = PIMA_COLS + ["BMI_over_Age","HighGlucoseFlag","Missing_BloodPressure","Missing_SkinThickness","Missing_Insulin"]
CAT_COLS = []

def build_preprocess():
    numeric = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("sc", RobustScaler())
    ])
    categorical = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", numeric, NUM_COLS),
        ("cat", categorical, CAT_COLS)
    ])
    return pre

def build_baseline_model():
    lr = LogisticRegression(penalty="elasticnet", l1_ratio=0.3, C=1.0, solver="saga", max_iter=2000, n_jobs=-1)
    return lr

def build_xgb_model(random_state=42):
    xgb = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )
    return xgb
