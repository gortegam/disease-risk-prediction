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

