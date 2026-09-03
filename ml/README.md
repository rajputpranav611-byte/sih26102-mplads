# ML Pipeline README

This folder contains the synthetic MP work-data pipeline for anomaly detection and fraud classification.

## 1) Folder structure

- `data/raw/`  
  Raw source MP allocation files used to build the synthetic work records.

- `data/processed/`  
  Generated and cleaned datasets, including:
  - `works.csv`
  - `works_features.csv`
  - `eda_summary.png`

- `src/`  
  Python scripts for generation, feature engineering, training, duplicate checking, and prediction.
  Key files:
  - `generate_data.py`
  - `features.py`
  - `train_anomaly.py`
  - `train_fraud_classifier.py`
  - `duplicate_detection.py`
  - `predict.py`
  - `test_predict.py`

- `models/`  
  Persisted trained models:
  - `isolation_forest.pkl`
  - `xgboost_classifier.pkl`

- `notebooks/`  
  EDA and exploration notebooks.

---

## 2) Reproduce everything from scratch

Run the pipeline in this exact order:

1. `src/generate_data.py`  
   Builds the synthetic work dataset and injects realistic anomaly patterns into `data/processed/works.csv`.

2. `src/features.py`  
   Creates the engineered feature set and saves it to `data/processed/works_features.csv`.

3. `src/train_anomaly.py`  
   Trains the Isolation Forest model and saves `models/isolation_forest.pkl`.

4. `src/train_fraud_classifier.py`  
   Trains the XGBoost fraud classifier and saves `models/xgboost_classifier.pkl`.

Example command sequence:

```powershell
Set-Location -LiteralPath 'F:\SIH 26102 mpland\sih26102-mplads\ml'
& 'C:/Users/User/AppData/Local/Python/pythoncore-3.14-64/python.exe' src/generate_data.py
& 'C:/Users/User/AppData/Local/Python/pythoncore-3.14-64/python.exe' src/features.py
& 'C:/Users/User/AppData/Local/Python/pythoncore-3.14-64/python.exe' src/train_anomaly.py
& 'C:/Users/User/AppData/Local/Python/pythoncore-3.14-64/python.exe' src/train_fraud_classifier.py
```

---

## 3) How the backend teammate should import and call predict()

The backend should import the function from `src/predict.py` and pass a single work record dictionary.

### Import

```python
from src.predict import predict
```

If running from the `ml/` directory, make sure the project root is on the Python path or use a package-style import depending on the environment.

### Exact input dict shape

The function expects a dictionary with these fields:

```python
{
    "work_id": "W-123",
    "sanctioned_amount": 3200000,
    "expenditure_amount": 2100000,
    "vendor_id": "V-STRAIGHT-001",
    "beneficiary_ids": "B-UNIQUE-001",
    "recommended_date": "2026-06-01",
    "sanction_date": "2026-06-30",
    "completion_percent": 74,
    "work_category": "Roads",
    "house": "Lok Sabha",
    "district": "Nashik",
    "constituency": "Nashik",
}
```

Field list:
- `work_id`
- `sanctioned_amount`
- `expenditure_amount`
- `vendor_id`
- `beneficiary_ids`
- `recommended_date`
- `sanction_date`
- `completion_percent`
- `work_category`
- `house`
- `district`
- `constituency`

### Exact output dict shape

```python
{
    "work_id": "W-123",
    "risk_score": 0.2504,
    "is_anomaly": False,
    "flags": [],
    "explanation": "No major anomaly signals were detected; cost, timing, and duplicate checks remain within expected ranges."
}
```

Required output keys:
- `work_id` : str
- `risk_score` : float
- `is_anomaly` : bool
- `flags` : list[str]
- `explanation` : str

### Example call

```python
record = {
    "work_id": "W-123",
    "sanctioned_amount": 3200000,
    "expenditure_amount": 2100000,
    "vendor_id": "V-STRAIGHT-001",
    "beneficiary_ids": "B-UNIQUE-001",
    "recommended_date": "2026-06-01",
    "sanction_date": "2026-06-30",
    "completion_percent": 74,
    "work_category": "Roads",
    "house": "Lok Sabha",
    "district": "Nashik",
    "constituency": "Nashik",
}

result = predict(record)
print(result)
```

---

## 4) Model availability requirement

The trained model files must already exist before `predict()` can be called:

- `models/isolation_forest.pkl`
- `models/xgboost_classifier.pkl`

These are created by running:
- `src/train_anomaly.py`
- `src/train_fraud_classifier.py`

If the models do not exist yet, `predict()` will fail to load them.

---

## 5) Final verified metrics

These were the validated results on the regenerated five-type dataset.

### XGBoost classifier
- ROC-AUC: 0.84
- Precision: 0.66
- Recall: 0.63
- F1-score: 0.65

### Isolation Forest
- Recall (true anomalies caught / actual anomalies): 0.48
- Precision (true positives / predicted anomalies): 0.48

### Per anomaly-type breakdown

#### Isolation Forest
- cost_overrun: 29/84 caught
- delayed_completion: 50/84 caught
- sanction_delay: 18/84 caught
- duplicate_vendor: 71/83 caught
- duplicate_beneficiary: 34/83 caught

#### XGBoost classifier (test set)
- cost_overrun: 10/13 caught
- delayed_completion: 15/19 caught
- sanction_delay: 6/16 caught
- duplicate_vendor: 12/16 caught
- duplicate_beneficiary: 10/20 caught

---

## 6) Duplicate vendor scoring note

The duplicate-vendor logic is not based on a rule alone. The combined score uses the trained model output and then adds a rule-based cluster boost when a vendor matches a known duplicate vendor pattern.

In practice:
- XGBoost probability is combined with the Isolation Forest normalized anomaly score.
- If the work matches a duplicate vendor cluster, the risk score is raised to a minimum threshold (`0.55`) to reflect the strong repeated-pattern signal.
- Similarly, if the beneficiary matches a repeated duplicate-beneficiary pattern, the risk score is raised to a minimum threshold (`0.60`).

This means the final `risk_score` is not purely a raw model score; it is the model score plus a small explicit duplicate-pattern adjustment so these fraud patterns are not silently missed when the underlying model is uncertain.

This is intentionally documented here so future readers understand why duplicate cluster flags can push the combined risk above the normal threshold.
