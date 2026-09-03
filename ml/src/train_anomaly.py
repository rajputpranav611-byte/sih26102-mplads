from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "works_features.csv"
MODEL_PATH = BASE_DIR / "models" / "isolation_forest.pkl"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    if "is_anomaly" not in df.columns:
        raise ValueError("The data file does not contain the 'is_anomaly' label column.")

    base_features = [
        "expenditure_ratio",
        "days_since_sanction",
        "days_since_recommendation",
        "sanction_gap_days",
        "vendor_work_count",
        "beneficiary_work_count",
        "house_Lok Sabha",
        "house_Rajya Sabha",
    ]
    work_category_cols = [col for col in df.columns if col.startswith("work_category_")]
    feature_cols = base_features + work_category_cols

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[feature_cols].copy().fillna(0)

    model = IsolationForest(
        contamination=0.10,
        random_state=42,
        n_estimators=300,
    )
    model.fit(X)

    anomaly_scores = model.decision_function(X)
    anomaly_labels = model.predict(X)

    df["iforest_score"] = anomaly_scores
    df["iforest_pred"] = anomaly_labels

    actual_anomalies = df["is_anomaly"].astype(bool)
    predicted_anomalies = (anomaly_labels == -1)

    true_positives = int((predicted_anomalies & actual_anomalies).sum())
    actual_total = int(actual_anomalies.sum())
    predicted_total = int(predicted_anomalies.sum())
    recall = (true_positives / actual_total) if actual_total else 0.0
    precision = (true_positives / predicted_total) if predicted_total else 0.0

    print(f"True anomaly count: {actual_total}")
    print(f"Predicted anomalies: {predicted_total}")
    print(f"True anomalies caught: {true_positives}")
    print(f"Recall (true anomalies caught / actual anomalies): {recall:.4f}")
    print(f"Precision (true positives / predicted anomalies): {precision:.4f}")

    anomaly_types = [
        "cost_overrun",
        "delayed_completion",
        "sanction_delay",
        "duplicate_vendor",
        "duplicate_beneficiary",
    ]
    print("\nAnomaly type breakdown (true labels vs Isolation Forest):")
    for anomaly_type in anomaly_types:
        true_mask = (df["anomaly_type"] == anomaly_type)
        if not true_mask.any():
            caught = 0
            total = 0
        else:
            total = int(true_mask.sum())
            caught = int(((predicted_anomalies) & true_mask).sum())
        print(f"- {anomaly_type}: {caught}/{total} caught")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
