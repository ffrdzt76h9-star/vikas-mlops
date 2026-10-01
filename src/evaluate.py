import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
)

from target import add_target, TARGET_COL, RAW_PATH


def run_evaluation():
    print("Starting Model Evaluation...")

    # 1. Load test data and the best model
    X_test_final = np.load("data/processed/X_test_final.npy")
    y_test = np.load("data/processed/y_test.npy")
    model = joblib.load("models/best_model.pkl")
    try:
        with open("models/best_model_name.txt") as f:
            best_name = f.read().strip()
    except FileNotFoundError:
        best_name = type(model).__name__

    # 2. Predictions
    y_pred = model.predict(X_test_final)

    # 3. Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n--- Model Evaluation Report ({best_name}) ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("-------------------------------\n")

    # 4. Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Budget", "Premium"]
    )
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig("artifacts/confusion_matrix.png", dpi=120)
    plt.close()

    # 5. Error analysis export (rebuild the same split to recover the raw rows)
    df = add_target(pd.read_csv(RAW_PATH))
    _, X_test_raw = train_test_split(
        df.drop(TARGET_COL, axis=1), test_size=0.2, random_state=42, stratify=df[TARGET_COL]
    )

    errors_df = X_test_raw.copy()
    errors_df["Actual_Class"] = y_test
    errors_df["Predicted_Class"] = y_pred

    # False negative = Premium phone predicted as Budget; false positive = the reverse
    false_negatives = errors_df[(errors_df["Actual_Class"] == 1) & (errors_df["Predicted_Class"] == 0)]
    false_positives = errors_df[(errors_df["Actual_Class"] == 0) & (errors_df["Predicted_Class"] == 1)]

    false_negatives.to_csv("outputs/false_negatives.csv", index=False)
    false_positives.to_csv("outputs/false_positives.csv", index=False)

    print("Evaluation complete and error analysis files saved!")


if __name__ == "__main__":
    run_evaluation()
