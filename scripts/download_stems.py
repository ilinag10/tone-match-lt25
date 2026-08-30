import os
import re
import logging
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import src.downloader
from src.separator import isolate_guitar_stem

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
INPUT_CSV = "data/anchors.csv"
OUTPUT_DIR = "data/audio_stems"
TEMP_DIR = "data/temp_raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def clean_string(s: str) -> str:
    """Standardizes strings for consistent filename creation and matching."""
    s = str(s).lower()
    s = re.sub(r"[^\w\s-]", "", s)  # Remove punctuation except letters, numbers, spaces, and hyphens
    s = re.sub(r"[\s-]+", "_", s).strip("_")  # Replace spaces and hyphens with a single underscore
    return s

def main():
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input CSV not found at {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    logging.info(f"Loaded {len(df)} anchor entries from {INPUT_CSV}")

    success_count = 0

    for idx, row in df.iterrows():
        song_title = row.get("song_title", f"track_{idx:03d}")
        artist = row.get("artist", "unknown")
        youtube_url = row.get("youtube_url")
        start_time = int(row.get("start_time", 0))

        # Target output stem filename
        safe_artist = clean_string(artist)
        safe_title = clean_string(song_title)
        final_filename = f"{idx:03d}_{safe_artist}_{safe_title}.wav"
        final_stem_path = os.path.join(OUTPUT_DIR, final_filename)

        # Temporary path for raw YouTube clip before source separation
        raw_temp_path = os.path.join(TEMP_DIR, f"temp_{idx:03d}_{safe_title}.wav")

        # Skip if isolated stem already exists
        if os.path.exists(final_stem_path):
            logging.info(f"[{idx+1}/{len(df)}] Skipping existing file: {final_filename}")
            success_count += 1
            continue

        if not isinstance(youtube_url, str) or not youtube_url.startswith("http"):
            logging.warning(f"[{idx+1}/{len(df)}] Invalid URL for '{song_title}'. Skipping...")
            continue

        logging.info(f"[{idx+1}/{len(df)}] Downloading: {artist} - {song_title}")
        raw_downloaded_path = None

        try:
            # 1. Download full audio clip to temp directory
            raw_downloaded_path = src.downloader.download_audio(youtube_url, start_time, duration=30, output_dir=TEMP_DIR)

            if not raw_downloaded_path or not os.path.exists(raw_downloaded_path):
                logging.error(f"Raw download failed for '{song_title}'")
                continue

            # 2. Isolate guitar stem using separator module
            # Passes TEMP_DIR as intermediate output directory
            isolated_path = isolate_guitar_stem(raw_downloaded_path, output_dir=TEMP_DIR)

            # 3. Rename and move isolated stem to final destination
            if isolated_path and os.path.exists(isolated_path):
                os.replace(isolated_path, final_stem_path)
                logging.info(f"Successfully isolated guitar stem: {final_filename}")
                success_count += 1
            else:
                logging.error(f"Guitar isolation produced no output for '{song_title}'")

        except Exception as e:
            logging.error(f"Error processing '{song_title}': {e}")

        finally:
            # 4. Clean up raw temporary audio download
            if raw_downloaded_path and os.path.exists(raw_downloaded_path):
                os.remove(raw_downloaded_path)

    # Remove temporary directory if empty
    if os.path.exists(TEMP_DIR) and not os.listdir(TEMP_DIR):
        os.rmdir(TEMP_DIR)

    logging.info(f"Done! {success_count}/{len(df)} isolated guitar stems saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()