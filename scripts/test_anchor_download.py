import os
import sys
from pathlib import Path
import tempfile
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.downloader import download_audio

CSV_PATH = "data/anchors.csv"


def run_download_test(csv_path: str = CSV_PATH):
    """
    Iterates through all YouTube links in anchors.csv and tests downloadability.
    Deletes temporary downloaded files immediately to conserve disk space.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found at '{csv_path}'")
        return

    df = pd.read_csv(csv_path)

    if "youtube_url" not in df.columns:
        print("❌ Error: 'youtube_url' column not found in CSV.")
        return

    total = len(df)
    successes = 0
    failures = []

    print(f"==================================================")
    print(f" Starting Download Test for {total} Anchor Links")
    print(f"==================================================\n")

    # Use a temporary directory so test artifacts are auto-cleaned
    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, row in df.iterrows():
            song = row.get("song_title", f"Track #{idx + 1}")
            artist = row.get("artist", "Unknown Artist")
            url = row.get("youtube_url")
            start_time = row.get("start_time")
            duration = 30

            track_label = f"{artist} - {song}"
            print(f"[{idx + 1}/{total}] Testing: {track_label}")
            print(f"   URL: {url}")

            # Validate URL presence and catch unfilled placeholder links
            if not url or pd.isna(url) or "EXAMPLE" in str(url):
                print("   ❌ FAILED: Missing or placeholder URL\n")
                failures.append({
                    "track": track_label,
                    "url": url,
                    "reason": "Missing or placeholder URL"
                })
                continue

            try:
                # Execute the actual downloader function
                output_path = download_audio(url, start_time, duration, output_dir=temp_dir)

                if output_path and os.path.exists(output_path):
                    print(f"   ✅ SUCCESS: Download verified.\n")
                    successes += 1
                    # Delete audio file right away to free space
                    os.remove(output_path)
                else:
                    raise FileNotFoundError("download_audio completed but returned no valid output path.")

            except Exception as e:
                print(f"   ❌ FAILED: {str(e)}\n")
                failures.append({
                    "track": track_label,
                    "url": url,
                    "reason": str(e)
                })

    # Summary Report
    print("=" * 50)
    print(" DOWNLOAD TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tested : {total}")
    print(f"Passed       : {successes}")
    print(f"Failed       : {len(failures)}")
    print("=" * 50)

    if failures:
        print("\nBroken / Unreachable Links:")
        for f in failures:
            print(f"• {f['track']}")
            print(f"  URL: {f['url']}")
            print(f"  Error: {f['reason']}\n")


if __name__ == "__main__":
    run_download_test()