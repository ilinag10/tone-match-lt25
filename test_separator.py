import os
import subprocess
import sys
from src.separator import isolate_guitar_stem

def test_separation():

    sample_file = "audio_downloads/demucs_test.wav"

    if not os.path.exists(sample_file):
        print(f"[Error] Test file '{sample_file}' not found.")
        print("Please place a sample .wav file in 'data/raw/' first, or run src/downloader.py!")
        sys.exit(1)
        
    print(f"--- Starting Separation Test on: {sample_file} ---")
    
    # 2. Run the separation function
    output_stem = isolate_guitar_stem(sample_file)
    
    # 3. Assertions and checks
    print("\n--- Test Results ---")
    print(f"Returned Stem Path: {output_stem}")
    
    if os.path.exists(output_stem):
        file_size_mb = os.path.getsize(output_stem) / (1024 * 1024)
        print(f"File exists! Size: {file_size_mb:.2f} MB")
        
        if output_stem.endswith("guitar.wav") and file_size_mb > 0:
            print("\nSUCCESS: 'guitar.wav' was successfully isolated and is non-empty!")
        else:
            print("\nWARNING: Output exists, but fallback path or empty file was returned.")
    else:
        print("\nFAIL: Output path does not exist on disk.")


if __name__ == "__main__":
    test_separation()