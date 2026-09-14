import glob
import os
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.feature_extractor import extract_raw_blueprint_features

# Paths
AUDIO_DIR = "data/audio_stems"  
METADATA_CSV = "data/anchors.csv"
OUTPUT_CSV = "data/anchors_features.csv"

SAMPLE_RATE = 22050


def process_anchor_dataset():
    if not os.path.exists(METADATA_CSV):
        raise FileNotFoundError(f"Missing anchor metadata CSV at {METADATA_CSV}")

    df_meta = pd.read_csv(METADATA_CSV)
    print(f"Loaded {len(df_meta)} metadata records from {METADATA_CSV}")

    features_list = []

    for idx, row in df_meta.iterrows():
        # Identify audio file using song_title or artist
        identifier = row.get("song_title", row.get("artist", f"anchor_{idx}"))

        # Search for matching audio file
        audio_files = glob.glob(f"{AUDIO_DIR}/{identifier}*")

        if not audio_files:
            # Fallback to positional audio matching if specific file name is not found
            all_audio = glob.glob(f"{AUDIO_DIR}/*")
            if idx < len(all_audio):
                audio_path = all_audio[idx]
            else:
                print(
                    f"Warning: No audio file found for row {idx} ({identifier}). Skipping."
                )
                continue
        else:
            audio_path = audio_files[0]

        print(f"Processing: {os.path.basename(audio_path)}...")

        # Call existing feature extraction function directly
        dsp_features = extract_raw_blueprint_features(audio_path, sample_rate=SAMPLE_RATE)

        # Merge metadata attributes with extracted DSP features
        combined_record = {**row.to_dict(), **dsp_features}
        features_list.append(combined_record)

    df_out = pd.DataFrame(features_list)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSuccessfully generated {OUTPUT_CSV} with {len(df_out)} records.")


if __name__ == "__main__":
    process_anchor_dataset()