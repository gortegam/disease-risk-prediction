# Disease Risk Prediction  
*A machine learning pipeline for predicting disease risk from laboratory tests and demographics*  

[![Streamlit App](https://img.shields.io/badge/demo-streamlit-brightgreen)](#)  
[![Power BI Dashboard](https://img.shields.io/badge/dashboard-PowerBI-blue)](#)  

---

## 📌 Overview  
This project develops an **end-to-end pipeline** for predicting chronic disease risk using routine lab values and basic patient demographics. It demonstrates how data science can be applied in healthcare analytics for early screening and decision support.  

- **Core features:** exploratory data analysis, feature engineering, model training (logistic regression, XGBoost), probability calibration, and explainability (SHAP).  
- **Deliverables:**  
  - ⚙️ **Pipeline** (`src/`) — clean, reproducible codebase  
  - 📊 **Streamlit app** — patient-level risk estimation  
  - 📈 **Power BI dashboards** — population-level insights  
- **MVP:** Type-2 Diabetes risk prediction using the Pima Indians Diabetes dataset  
- **Future extensions:** NHANES dataset, Chronic Kidney Disease (CKD), and Atherosclerotic Cardiovascular Disease (ASCVD).  

---

## 🚀 Quickstart  

```bash
# (1) clone repo
git clone https://github.com/<your-username>/disease-risk-prediction.git
cd disease-risk-prediction

# (2) install dependencies
pip install -r requirements.txt

# (3) train a calibrated model on the sample (or your dataset)
python -m src.train --csv data/pima_sample.csv --outdir artifacts

# (4) run the app (upload a CSV with the schema below)
streamlit run app/streamlit_app.py
```
## 🧪 Data Schema (Pima MVP)

Required columns (case-sensitive):  
- `Pregnancies` (int)  
- `Glucose` (mg/dL)  
- `BloodPressure` (mmHg)  
- `SkinThickness` (mm)  
- `Insulin` (IU/mL)  
- `BMI` (kg/m²)  
- `DiabetesPedigreeFunction`  
- `Age` (years)  
- `Outcome` (0/1) ← **training only**; omit for inference  

**Notes:**  
- In the original dataset, zeros can indicate **missing** for some labs (e.g., `SkinThickness`, `Insulin`, `BloodPressure`).  
- The pipeline treats biologically implausible zeros as missing, imputes them, and adds missing-indicator flags.

---

## 📂 Artifacts Produced

- `artifacts/model.joblib` — fitted pipeline (preprocess + classifier)  
- `artifacts/calibrator.joblib` — probability calibration model (isotonic)  
- `artifacts/feature_list.joblib` — list of input feature names used at train time  
- `artifacts/metrics.json` — AUROC, PR-AUC, Brier score  
- `artifacts/example_predictions.csv` — demo predictions with `risk` and `risk_tier`  
- *(optional via `src/evaluate.py`)* `artifacts/curves.json` — ROC/PR curve points

---

## 📊 Power BI Binding Guide

Use `artifacts/example_predictions.csv` (or the Streamlit app’s exported `predictions.csv`). Recommended visuals:

- Histogram of predicted risk (`risk`)  
- Metrics cards for AUROC, PR-AUC, Brier (enter from `metrics.json`)  
- Confusion matrix at adjustable threshold (What-If parameter)  
- Risk tier distribution by demographics (e.g., Age, BMI bands)  

See **`powerbi/README.md`** for suggested measures (Sensitivity, Specificity, PPV, NPV) and setup steps.

---

## ⚖️ Ethics & Clinical Caveats

- This project is **for educational and portfolio purposes only**.  
- Outputs are **not diagnostic** and must not be used for clinical decision-making.  
- Assess fairness: compare performance across age/sex/BMI groups and report gaps.  
- Prevent leakage: ensure labs that define labels aren’t simultaneously used as predictors from the same timepoint.

---

## 🔮 Roadmap

- ✅ MVP: Pima Indians Diabetes dataset  
- 🔄 Next: NHANES ingestion + Diabetes risk replication  
- 🔜 Additional conditions:  
  - Chronic Kidney Disease (CKD) via eGFR thresholds  
  - ASCVD 10-year risk proxy (lipids + vitals + demographics)

---

## 📖 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

