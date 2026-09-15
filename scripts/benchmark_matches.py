import os
import shutil
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.downloader import run_downloader_cli
from src.separator import run_separator_cli
from src.feature_extractor import extract_raw_blueprint_features
from src.tone_matcher import ToneMatcher

# Temporary workspace for evaluation
BENCHMARK_TEMP_DIR = "data/processed/benchmark_temp"

def cleanup_benchmark_cache(stem_folder_path: str, raw_wav_path: str):
    """Cleans up temporary files generated during benchmark run."""
    for path in [stem_folder_path, raw_wav_path, BENCHMARK_TEMP_DIR]:
        if path and os.path.exists(path):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except Exception as e:
                print(f" Warning: Failed to cleanup {path}: {e}")


def print_extracted_features(features: dict):
    """Prints extracted DSP blueprint features cleanly to stdout."""
    print("\n--------------------------------------------------")
    print("         EXTRACTED DSP BLUEPRINT FEATURES         ")
    print("--------------------------------------------------")

    # Separate base DSP features from MFCC array/keys
    base_features = {
        k: v for k, v in features.items() if not k.startswith("mfcc")
    }
    mfcc_features = {
        k: v for k, v in features.items() if k.startswith("mfcc")
    }

    print("--- Core Audio Metrics ---")
    for key, val in base_features.items():
        if isinstance(val, float):
            print(f"  * {key:20s}: {val:.4f}")
        else:
            print(f"  * {key:20s}: {val}")

    if mfcc_features:
        print("\n--- MFCC Coefficients ---")
        for key in sorted(mfcc_features.keys()):
            val = mfcc_features[key]
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            print(f"  * {key:20s}: {val_str}")
    elif "mfccs" in features:
        print("\n--- MFCC Array ---")
        mfcc_list = features["mfccs"]
        formatted_mfccs = [
            f"{v:.4f}" if isinstance(v, float) else str(v) for v in mfcc_list
        ]
        print(f"  * mfccs               : {formatted_mfccs}")

    print("--------------------------------------------------\n")



def run_multi_match_benchmark(k_values=[1, 3, 5]):
    """
    Downloads and separates audio ONCE, then evaluates feature matching 
    across multiple k-NN configurations for comparative analysis.
    """
    os.makedirs(BENCHMARK_TEMP_DIR, exist_ok=True)
    wav_path = None
    guitar_stem = None
    stem_folder_path = None

    try:
        # Step 1: Run Downloader ONCE
        print("\n==================================================")
        print("   STEP 1: DOWNLOAD AUDIO (SINGLE EXECUTION)      ")
        print("==================================================")
        wav_path = run_downloader_cli()
        if not wav_path or not os.path.exists(wav_path):
            print("Download failed or was cancelled. Exiting benchmark.")
            return

        # Step 2: Run Demucs Stem Separation ONCE
        print("\n==================================================")
        print("   STEP 2: STEM SEPARATION (SINGLE EXECUTION)    ")
        print("==================================================")
        guitar_stem = run_separator_cli(audio_path=wav_path)
        if isinstance(guitar_stem, (tuple, list)):
            guitar_stem = guitar_stem[0]

        if guitar_stem and os.path.exists(guitar_stem):
            stem_folder_path = os.path.dirname(guitar_stem)

        # Step 3: Run Feature Extraction ONCE on the generated guitar stem
        print("\n==================================================")
        print("   STEP 3: FEATURE EXTRACTION (SINGLE EXECUTION)  ")
        print("==================================================")
        print(f"Extracting DSP blueprint features from: {os.path.basename(guitar_stem)}")
        target_features = extract_raw_blueprint_features(guitar_stem, sample_rate=22050)

        print_extracted_features(target_features)


        # Step 4: Iterative Matching Benchmarks (MULTIPLE RUNS)
        print("\n==================================================")
        print("   STEP 4: MULTI-CONFIG k-NN MATCH COMPARISON     ")
        print("==================================================")

        results = []
        for k in k_values:
            print(f"--> Evaluating ToneMatcher with k={k}...")
            matcher = ToneMatcher(n_neighbors=k)
            recipe = matcher.match_features(target_features)
            
            # Format row for comparison matrix
            results.append({
                "k_Neighbors": k,
                "Match Distance": recipe["match_distance"],
                "Preset Name": recipe["preset_name"],
                "Amp Model": recipe["amp_model"],
                "Gain": recipe["knob_settings"]["gain"],
                "Treble": recipe["knob_settings"]["treble"],
                "Middle": recipe["knob_settings"]["middle"],
                "Bass": recipe["knob_settings"]["bass"],
                "Stomp FX": recipe["fx_blocks"]["stomp"],
                "Mod FX": recipe["fx_blocks"]["mod"],
                "Delay FX": recipe["fx_blocks"]["delay"],
                "Reverb FX": recipe["fx_blocks"]["reverb"],
            })

        # Step 5: Display Side-by-Side Comparison
        df_results = pd.DataFrame(results)
        print("\n" + "=" * 80)
        print("                   MUSTANG LT25 MATCH COMPARISON SUMMARY           ")
        print("=" * 80)
        print(df_results.to_string(index=False))
        print("=" * 80 + "\n")

    finally:
        print("\n=== Cleaning Up Temporary Benchmark Audio ===")
        cleanup_benchmark_cache(stem_folder_path, wav_path)

if __name__ == "__main__":
    # Compare output across k=1, k=3, and k=5 nearest neighbors
    run_multi_match_benchmark(k_values=[1, 3, 5])