"""Loading the Kaggle Playground S6E4 data (Predicting Irrigation Need).

The competition data can't be redistributed, so it isn't in the repo. Put
train.csv in ./data, point GROUNDWORK_DATA at a folder that has it, or run on
Kaggle with the competition attached.
"""

import os
from pathlib import Path

import pandas as pd

TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]

NUMERIC = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity", "Temperature_C",
    "Humidity", "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh", "Field_Area_hectare",
    "Previous_Irrigation_mm",
]
CATEGORICAL = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season", "Irrigation_Type",
    "Water_Source", "Mulching_Used", "Region",
]

_SEARCH = [
    Path(os.environ.get("GROUNDWORK_DATA", "data")),
    Path(__file__).resolve().parents[2] / "data",
    Path("/kaggle/input/competitions/playground-series-s6e4"),
    Path("/kaggle/input/playground-series-s6e4"),
    Path.home() / ".mldata" / "s6e4",
]


def data_dir() -> Path:
    for folder in _SEARCH:
        if (folder / "train.csv").exists():
            return folder
    raise FileNotFoundError(
        "train.csv not found. Download it with "
        "`kaggle competitions download -c playground-series-s6e4` into ./data"
    )


def load(split: str = "train") -> pd.DataFrame:
    df = pd.read_csv(data_dir() / f"{split}.csv")
    if TARGET in df:
        df[TARGET] = pd.Categorical(df[TARGET], categories=CLASSES, ordered=True)
    return df


def sample(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """A stratified random sample, small enough for plots and slow algorithms."""
    df = load()
    frac = n / len(df)
    return (
        df.groupby(TARGET, observed=True, group_keys=False)
        .sample(frac=frac, random_state=seed)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def xy(df: pd.DataFrame, numeric_only: bool = False):
    """Features as a one-hot encoded float matrix and the target as class indices."""
    cols = NUMERIC if numeric_only else NUMERIC + CATEGORICAL
    X = pd.get_dummies(df[cols], columns=[] if numeric_only else CATEGORICAL, dtype=float)
    y = df[TARGET].cat.codes.to_numpy()
    return X, y
