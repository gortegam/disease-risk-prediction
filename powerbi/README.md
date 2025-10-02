# Power BI Binding Guide

**Data source:** `artifacts/example_predictions.csv` or Streamlit `predictions.csv`.

### Fields
- `risk` (decimal) — model probability
- `risk_tier` (Low/Moderate/High)
- `y_true` (0/1, only in example_predictions)
- Any original columns you included (age, BMI, etc.)

### Visuals
- Histogram of `risk` (bin width 0.02)
- Gauge or Cards for AUROC/PR-AUC (from metrics.json, enter manually or via Power Query JSON parse)
- Matrix by `risk_tier` × demographic slicers
- Threshold tuning (What‑If parameter `thresh` ∈ [0,1] step .01), add measures:
```
PredPos := SUMX( 'data', IF( 'data'[risk] >= [thresh], 1, 0 ) )
TP := SUMX( 'data', IF( 'data'[risk] >= [thresh] && 'data'[y_true]=1, 1, 0 ) )
FP := [PredPos] - [TP]
FN := SUMX( 'data', IF( 'data'[risk] < [thresh] && 'data'[y_true]=1, 1, 0 ) )
TN := COUNTROWS('data') - [TP] - [FP] - [FN]
Sensitivity := DIVIDE([TP], [TP]+[FN])
Specificity := DIVIDE([TN], [TN]+[FP])
PPV := DIVIDE([TP], [TP]+[FP])
NPV := DIVIDE([TN], [TN]+[FN])
```
