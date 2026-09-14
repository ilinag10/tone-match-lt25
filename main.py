import os
import shutil
from src.downloader import run_downloader_cli
from src.separator import run_separator_cli
from src.feature_extractor import extract_raw_blueprint_features
from src.tone_matcher import ToneMatcher, print_recipe

def cleanup_path(target_path: str):
    """Deletes a file or an entire directory tree if it exists."""
    if not target_path or not os.path.exists(target_path):
        return
    try:
        if os.path.isdir(target_path):
            shutil.rmtree(target_path)
            print(f" Cleaned up stem directory: {os.path.basename(target_path)}")
        else:
            os.remove(target_path)
            print(f" Cleaned up file: {os.path.basename(target_path)}")
    except Exception as e:
        print(f" Warning: Could not remove {target_path}: {e}")


def main():
    wav_path = None
    guitar_stem = None
    stem_folder_path = None

    try:
        print("=== Step 1: Downloader ===")
        wav_path = run_downloader_cli()
        
        if not wav_path:
            print("Download failed or was cancelled. Exiting.")
            return

        print("\n=== Step 2: Stem Separation ===")
        guitar_stem = run_separator_cli(audio_path=wav_path)

        if isinstance(guitar_stem, (tuple, list)):
            guitar_stem = guitar_stem[0]

        if guitar_stem and os.path.exists(guitar_stem):
            stem_folder_path = os.path.dirname(guitar_stem)

        print("\n=== Step 3: Feature Extraction ===")
        target_features = extract_raw_blueprint_features(guitar_stem)

        print("\n=== Step 4: k-NN Tone Matching ===")
        matcher = ToneMatcher(n_neighbors=1)
        recipe = matcher.match_features(target_features)
        print_recipe(recipe)

    finally:
        print("\n=== Step 5: Post-Processing Cleanup ===")
        # 1. Clean up downloaded raw audio (.wav/.mp3)
        cleanup_path(wav_path)

        # 2. Clean up the entire Demucs song output folder
        if stem_folder_path and os.path.exists(stem_folder_path):
            cleanup_path(stem_folder_path)

        print("Pipeline execution completed.")

        print("Pipeline execution completed.")


if __name__ == "__main__":
    main()
