import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Paths
ANCHOR_CSV = "data/anchors_features.csv"
CLEAN_ANCHORS_CSV = "data/processed/anchors_clean.csv"
INDEX_PATH = "models/knn_index.pkl"
SCALER_PATH = "models/scaler.pkl"

os.makedirs("models", exist_ok=True)

# 15 DSP Features matching your pipeline
FEATURE_COLS = [
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_flatness",
    "active_rms",
    "crest_factor",
    "rms_frames",
    "stomp_score",
    "mod_score",
    "reverb_score",
    "delay_score",
    "delay_time_ms",
]

PRESET_METADATA_COLS = [
    "preset_name",
    "amp_model",
    "gain",
    "treble",
    "middle",
    "bass",
    "volume",
    "stomp_fx",
    "mod_fx",
    "delay_fx",
    "reverb_fx",
]

def clean_fx_val(val, default="None"):
    """Helper to catch NaN or null FX block values."""
    if pd.isna(val) or val is None or str(val).strip().lower() == "nan":
        return default
    return val

class ToneMatcher:
    def __init__(self, n_neighbors: int = 1):
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")

        # Auto-load anchors if dataset exists
        csv_target = CLEAN_ANCHORS_CSV if os.path.exists(CLEAN_ANCHORS_CSV) else ANCHOR_CSV
        self.anchors_df = pd.read_csv(csv_target) if os.path.exists(csv_target) else None


    def fit(self, csv_path: str = ANCHOR_CSV):
        """Loads anchor dataset, expands features, fits scaler and k-NN index."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Anchor database not found at {csv_path}")

        self.anchors_df = pd.read_csv(csv_path)

        # Handle MFCC columns dynamically if present
        mfcc_cols = [c for c in self.anchors_df.columns if c.startswith("mfcc_")]
        all_features = FEATURE_COLS + sorted(mfcc_cols)

        # Clean numerical values
        for col in all_features:
            self.anchors_df[col] = pd.to_numeric(self.anchors_df[col], errors="coerce").fillna(0.0)

        X = self.anchors_df[all_features].values.astype(np.float32)

        # Fit and save StandardScaler
        X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, SCALER_PATH)

        # Fit and save NearestNeighbors Index
        self.knn.fit(X_scaled)
        joblib.dump(self.knn, INDEX_PATH)

        print(f" Successfully fitted k-NN index on {len(self.anchors_df)} anchor presets.")


    def match_features(self, target_features: dict) -> dict:
        """Matches a single audio feature dictionary to the closest LT25 anchor preset."""
        # Ensure scaler and knn are loaded
        if not hasattr(self.scaler, "mean_"):
            self.scaler = joblib.load(SCALER_PATH)
            self.knn = joblib.load(INDEX_PATH)

        # Expand MFCC list/array if stored under a single key in target_features
        clean_features = target_features.copy()
        if "mfccs" in clean_features:
            raw_mfccs = clean_features.pop("mfccs")
            if isinstance(raw_mfccs, (list, np.ndarray)):
                for idx, val in enumerate(raw_mfccs):
                    clean_features[f"mfcc_{idx+1}"] = float(val)

        # Convert input dictionary to array matching feature order
        mfcc_cols = [c for c in target_features.keys() if c.startswith("mfcc_")]
        all_features = FEATURE_COLS + sorted(mfcc_cols)

        # Build scalar float array cleanly
        feature_list = []
        for col in all_features:
            val = clean_features.get(col, 0.0)
            # Unpack single-element arrays or scalars
            if isinstance(val, (list, np.ndarray)):
                val = float(val[0]) if len(val) > 0 else 0.0
            feature_list.append(float(val))

        feature_vector = np.array([feature_list], dtype=np.float32)

        # Scale features using fitted scaler
        feature_scaled = self.scaler.transform(feature_vector)

        # Query Nearest Neighbors
        distances, indices = self.knn.kneighbors(feature_scaled)
        matched_idx = indices[0][0]
        distance_score = float(distances[0][0])

        matched_preset = self.anchors_df.iloc[matched_idx]

        # Structure LT25 Preset Recipe Output
        recipe = {
            "match_distance": round(distance_score, 4),
            "preset_name": matched_preset.get("song_title", "Custom Preset"),
            "amp_model": matched_preset.get("amp_model", "Clean"),
            "knob_settings": {
                "gain": matched_preset.get("gain", 5.0),
                "volume": matched_preset.get("volume", 5.0),
                "treble": matched_preset.get("treble", 5.0),
                "middle": matched_preset.get("middle", 5.0),
                "bass": matched_preset.get("bass", 5.0),
            },
            "fx_blocks": {
                "stomp": clean_fx_val(matched_preset.get("stomp_fx")),
                "mod": clean_fx_val(matched_preset.get("mod_fx")),
                "delay": clean_fx_val(matched_preset.get("delay_fx")),
                "reverb": clean_fx_val(matched_preset.get("reverb_fx")),
            },
        }

        return recipe


def print_recipe(recipe: dict):
    """Formats and prints the matched preset recipe."""
    print("\n==================================================")
    print("      MUSTANG LT25 MATCHED TONE RECIPE           ")
    print("==================================================")
    print(f" Closest Match Preset : {recipe['preset_name']}")
    print(f" Amp Model            : {recipe['amp_model']}")
    print(f" Distance Metric      : {recipe['match_distance']:.4f}")
    print("\n --- Knob Settings ---")
    for knob, val in recipe["knob_settings"].items():
        print(f"   * {knob.capitalize():8s}: {val}")
    print("\n --- FX Slots ---")
    for slot, fx in recipe["fx_blocks"].items():
        print(f"   * {slot.capitalize():8s}: {fx}")
    print("==================================================\n")

if __name__ == "__main__":
    matcher = ToneMatcher(n_neighbors=1)
    matcher.fit(ANCHOR_CSV)