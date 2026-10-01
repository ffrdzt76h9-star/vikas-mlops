import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from target import add_target, TARGET_COL, ID_COL, RAW_PATH


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")

    # 1. Load the raw data
    df = pd.read_csv(RAW_PATH)

    # 2. Create target + basic cleaning
    df = add_target(df)
    if ID_COL in df.columns:
        df = df.drop(ID_COL, axis=1)

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Fill missing values with the TRAIN median, then scale
    train_median = X_train.median()
    X_train = X_train.fillna(train_median)
    X_test = X_test.fillna(train_median)

    num_cols = X_train.columns.tolist()  # every phone feature is numeric / binary

    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train)
    X_test_final = scaler.transform(X_test)

    # 5. Save artifacts
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.int64))

    joblib.dump(scaler, "models/scaler.pkl")

    metadata = {
        "dataset_name": "Mobile Phone Price-Tier Classification",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": [],
    }
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()
