"""Shared helper: builds the Budget/Premium target used across the whole project.

The raw phone dataset has no label, so (exactly as in the original notebook) the
target is derived from a standardised "spec score" of RAM, battery, resolution
and internal memory, plus Gaussian noise. Phones scoring at/above the median are
labelled Premium (1), the rest Budget (0).
"""
import numpy as np
import pandas as pd

TARGET_COL = "target"
ID_COL = "id"
RAW_PATH = "data/raw/mobile.csv"


def add_target(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Return a copy of df with a binary `target` column (0 = Budget, 1 = Premium)."""
    df = df.copy()
    rng = np.random.RandomState(seed)  # same stream as np.random.seed(42) in the notebook

    resolution = df["px_height"] * df["px_width"]
    score = (
        (df["ram"] - df["ram"].mean()) / df["ram"].std()
        + (df["battery_power"] - df["battery_power"].mean()) / df["battery_power"].std()
        + (resolution - resolution.mean()) / resolution.std()
        + (df["int_memory"] - df["int_memory"].mean()) / df["int_memory"].std()
    )
    noisy_score = score + rng.normal(0, 1.0, size=len(score))
    df[TARGET_COL] = (noisy_score >= np.median(noisy_score)).astype(int)
    return df
