# Project 3 — Predicting Disease Risk from Lab Tests (Pima first)

**MVP target:** Type‑2 Diabetes (binary) using the Pima “Indians Diabetes” dataset.  
**Deliverables:** reproducible pipeline (EDA → Train/Calibrate → Explain), Streamlit app for CSV upload, and Power BI binding instructions.

## Quickstart
```bash
# (1) create env and install
pip install -r requirements.txt

# (2) train a calibrated model on the sample (or your full Pima CSV)
python -m src.train --csv data/pima_sample.csv --outdir artifacts

# (3) run the app (upload a CSV with the schema below)
streamlit run app/streamlit_app.py
```

## Data schema (Pima)
Required columns (case-sensitive):
- `Pregnancies` (int)
- `Glucose` (mg/dL)
- `BloodPressure` (mmHg)
- `SkinThickness` (mm)
- `Insulin` (IU/mL)
- `BMI` (kg/m^2)
- `DiabetesPedigreeFunction`
- `Age` (years)
- `Outcome` (0/1)  ← **training only**; omit for inference

### Notes
- In the original dataset, zeros can indicate **missing** for some labs (e.g., SkinThickness, Insulin, BloodPressure). The pipeline treats biologically impossible zeros as missing, imputes them, and adds missing-indicator flags.

## Artifacts produced
- `artifacts/model.joblib` — trained classifier
- `artifacts/calibrator.joblib` — probability calibration
- `artifacts/preprocess.joblib` — imputer/scaler/encoder pipeline
- `artifacts/metrics.json` — AUROC, PR‑AUC, Brier, etc.
- `artifacts/example_predictions.csv` — per‑row predicted risk, risk tier, and SHAP values (when run with `--save_shap`)

## Ethics & clinical caveats
- This is **for educational/demo purposes only**. Not a diagnostic device. Risk estimates are sensitive to population and measurement context.
- Address fairness: compare performance across age/sex/BMI bands before any real deployment.
- Avoid leakage when you move beyond Pima (e.g., NHANES cohort splits by cycle, feature timestamping vs. outcome timing).

## Power BI quick bind
1. Train the model and export `artifacts/example_predictions.csv` (or app predictions).  
2. In Power BI Desktop: **Get Data → Text/CSV** and load the predictions file.  
3. Use the included `powerbi/README.md` for visuals and DAX snippets.

---

Made with ❤️ for a portable, clinician‑friendly demo.
