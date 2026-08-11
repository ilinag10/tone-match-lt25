import os
import sys
import librosa
import numpy as np
import src.feature_extractor

def test_feature_extraction():
    sample_file = "data/processed/htdemucs_6s/I Want You to Want Me (Live at Nippon Budokan, Tokyo, JPN - April 1978)/guitar.wav"

    if not os.path.exists(sample_file):
        print(f"[Error] Test file '{sample_file}' not found.")
        sys.exit(1)

    print(f"--- Starting Feature Mapping Test on: {sample_file} ---")

    features = src.feature_extractor.extract_raw_blueprint_features(sample_file)

    print("\n=== EXTRACTED BLUEPRINT FEATURES ===")
    for key, value in features.items():
        if key == "rms_frames":
            print(f"rms_frames        : Array with {len(value)} frames (min={value.min():.4f}, max={value.max():.4f})")
        elif key == "mfccs":
            rounded_mfccs = [round(v, 2) for v in value]
            print(f"mfccs (1-4)       : {rounded_mfccs}")
        elif isinstance(value, float):
            print(f"{key:<18}: {value:.4f}")
        else:
            print(f"{key:<18}: {value}")
    print("====================================\n")


if __name__ == "__main__": 
    test_feature_extraction()