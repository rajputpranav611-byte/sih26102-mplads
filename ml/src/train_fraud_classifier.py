from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "works_features.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_classifier.pkl"


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
    y = df["is_anomaly"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pos_ratio = y.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Positive rate in test set: {y_test.mean():.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\nAnomaly type breakdown on the test set:")
    test_index = X_test.index
    test_df = df.loc[test_index].copy()
    test_df["predicted_anomaly"] = y_pred

    anomaly_types = [
        "cost_overrun",
        "delayed_completion",
        "sanction_delay",
        "duplicate_vendor",
        "duplicate_beneficiary",
    ]
    for anomaly_type in anomaly_types:
        true_mask = test_df["anomaly_type"] == anomaly_type
        total = int(true_mask.sum())
        caught = int(((test_df["predicted_anomaly"] == 1) & true_mask).sum())
        print(f"- {anomaly_type}: {caught}/{total} caught")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
