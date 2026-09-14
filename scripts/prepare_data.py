import ast
import json
import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Paths
INPUT_CSV = "data/anchors_features.csv"
OUTPUT_CSV = "data/processed/anchors_clean.csv"
MODELS_DIR = "models"

os.makedirs("data/processed", exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Base DSP Feature Columns (excluding raw MFCC column which gets expanded below)
BASE_DSP_COLS = [
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_flatness",
    "active_rms",
    "crest_factor",
    "rms_frames_mean",
    "stomp_score",
    "mod_score",
    "reverb_score",
    "delay_score",
    "delay_time_ms",
]


def parse_mfcc_entry(val) -> list[float]:
    """Safely converts string representation of MFCC list into list of floats."""
    if isinstance(val, (list, np.ndarray)):
        return [float(x) for x in val]
    if pd.isna(val):
        return []
    val_str = str(val).strip()
    try:
        return [float(x) for x in json.loads(val_str)]
    except Exception:
        try:
            return [float(x) for x in ast.literal_eval(val_str)]
        except Exception:
            cleaned = re.sub(r"[^\d\.\,\-eE\s]", "", val_str)
            tokens = [t for t in re.split(r"[\s,]+", cleaned) if t]
            return [float(t) for t in tokens]


def parse_truncated_array_mean(val) -> float:
    """Extracts mean value from a numpy string representation containing '...'."""
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).replace("...", "").strip()
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", val_str)
    if not matches:
        return 0.0
    floats = [float(m) for m in matches]
    return float(np.mean(floats))


def prepare_feature_dataframe(df: pd.DataFrame,) -> tuple[pd.DataFrame, list[str]]:
    """Expands MFCCs into individual float columns and cleans rms_frames."""
    if "mfccs" in df.columns:
        mfcc_parsed = df["mfccs"].apply(parse_mfcc_entry)
        mfcc_arr = np.array(mfcc_parsed.tolist(), dtype=np.float32)

        n_mfccs = mfcc_arr.shape[1]
        mfcc_cols = [f"mfcc_{i+1}" for i in range(n_mfccs)]

        for idx, col in enumerate(mfcc_cols):
            df[col] = mfcc_arr[:, idx]
        df.drop(columns=["mfccs"], inplace=True)
    else:
        mfcc_cols = [c for c in df.columns if c.startswith("mfcc_")]

    # 2. Extract scalar mean from rms_frames and DROP raw string column
    if "rms_frames" in df.columns:
        df["rms_frames_mean"] = df["rms_frames"].apply(parse_truncated_array_mean)
        df.drop(columns=["rms_frames"], inplace=True)

    return df, mfcc_cols


def prepare_knn_data():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Anchor database not found at {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} anchor samples from {INPUT_CSV}")

    df, mfcc_cols = prepare_feature_dataframe(df)
    feature_cols = BASE_DSP_COLS + mfcc_cols

    # Ensure numerical feature types
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Fit StandardScaler on the full anchor set
    X = df[feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save fitted scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Fitted StandardScaler saved to {scaler_path}")

    # Attach scaled values to DataFrame
    for idx, col in enumerate(feature_cols):
        df[f"{col}_scaled"] = X_scaled[:, idx]

    # Save single processed CSV (no splits)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Processed anchor database saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    prepare_knn_data()