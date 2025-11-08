"""
Training script for the churn prediction model.

Usage:
    python -m src.train
"""

from __future__ import annotations

import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from .config import TARGET_COL, MODEL_PATH, PROCESSED_DATA_DIR
from .data import load_raw_data, clean_target, clean_features, train_val_test_split


from .model import build_model
from pathlib import Path


def main() -> None:
    """
    Train the churn model, evaluate on validation set, and save the model.
    """
    print("📥 Loading data...")
    df = load_raw_data()
    df = clean_target(df)
    df = clean_features(df)

    print("🔀 Splitting into train/val/test...")
    train_df, val_df, _ = train_val_test_split(df)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_val = val_df.drop(columns=[TARGET_COL])
    y_val = val_df[TARGET_COL]

    print("🏗️ Building model pipeline...")
    model = build_model()

    print("🚀 Training model...")
    model.fit(X_train, y_train)

    print("📊 Evaluating on validation set...")
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "roc_auc": roc_auc_score(y_val, y_val_proba),
        "precision": precision_score(y_val, y_val_pred),
        "recall": recall_score(y_val, y_val_pred),
        "f1": f1_score(y_val, y_val_pred),
    }

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    # Save metrics to file for documentation
    metrics_path = PROCESSED_DATA_DIR / "val_metrics.txt"
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"📝 Validation metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
