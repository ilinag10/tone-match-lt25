import ast
import json
import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths
INPUT_CSV = "data/anchors_features_augmented.csv"
OUTPUT_DIR = "data/processed"
MODELS_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Gain Family Mapping (Grouping 20 Amp Models into 4 Core Families)
GAIN_FAMILY_MAP = {
    # Clean (8 Models)
    "Super Clean": "Clean",
    "Champ": "Clean",
    "50s Twin": "Clean",
    "Princeton": "Clean",
    "Deluxe Cln": "Clean",
    "Twin Clean": "Clean",
    "Excelsior": "Clean",
    "60s Uk Cln": "Clean",
    
    # Crunch (5 Models)
    "Deluxe Dirt": "Crunch",
    "Bassman": "Crunch",
    "Smalltone": "Crunch",
    "70s Uk Cln": "Crunch",
    "70s Rock": "Crunch",
    
    # High Gain (4 Models)
    "80s Rock": "High Gain",
    "90s Rock": "High Gain",
    "Alt Metal": "High Gain",
    "Burn": "High Gain",
    
    # Metal (3 Models)
    "Metal 2000": "Metal",
    "Doom Metal": "Metal",
    "Super Heavy": "Metal",
}

# 2. Base DSP Feature Columns (excluding raw MFCC column which gets expanded below)
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

# 3. Target Columns
KNOB_COLS = ["gain", "treble", "middle", "bass", "volume"]
FX_COLS = ["stomp_fx", "mod_fx", "delay_fx", "reverb_fx"]


def map_gain_family(amp_name: str) -> str:
    """Maps a specific amp model name to its overarching gain family."""
    if pd.isna(amp_name):
        return "Clean"
    clean_name = str(amp_name).strip()
    return GAIN_FAMILY_MAP.get(clean_name, "Crunch")


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
    mfcc_parsed = df["mfccs"].apply(parse_mfcc_entry)
    mfcc_arr = np.array(mfcc_parsed.tolist(), dtype=np.float32)

    n_mfccs = mfcc_arr.shape[1]
    mfcc_cols = [f"mfcc_{i+1}" for i in range(n_mfccs)]

    for idx, col in enumerate(mfcc_cols):
        df[col] = mfcc_arr[:, idx]

    # 2. Extract scalar mean from rms_frames and DROP raw string column
    if "rms_frames" in df.columns:
        df["rms_frames_mean"] = df["rms_frames"].apply(
            parse_truncated_array_mean
        )
        df.drop(columns=["rms_frames"], inplace=True)

    # 3. Drop raw mfccs string column as well
    if "mfccs" in df.columns:
        df.drop(columns=["mfccs"], inplace=True)

    return df, mfcc_cols


def prepare_data():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Augmented features file not found at {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} samples from {INPUT_CSV}")

    df, mfcc_cols = prepare_feature_dataframe(df)
    feature_cols = BASE_DSP_COLS + mfcc_cols

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Assign Targets & Clean Missing Values
    df["gain_family"] = df["amp_model"].apply(map_gain_family)

    for col in FX_COLS:
        df[col] = df[col].fillna("None").astype(str)

    for col in KNOB_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(5.0)

    # Fit StandardScaler on Input Features
    X = df[feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Fitted StandardScaler saved to {scaler_path}")

    # Attach scaled feature values back to DataFrame
    for idx, col in enumerate(feature_cols):
        df[f"{col}_scaled"] = X_scaled[:, idx]

    # Perform 80/20 Stratified Split
    y_stratify = df["gain_family"].values
    df_train, df_val = train_test_split(
        df, test_size=0.20, random_state=42, stratify=y_stratify
    )

    # Save to disk
    train_path = os.path.join(OUTPUT_DIR, "train_dataset.csv")
    val_path = os.path.join(OUTPUT_DIR, "val_dataset.csv")
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)

    print(
        f"Data prep complete: Train ({len(df_train)} rows), Val ({len(df_val)} rows) saved to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    prepare_data()