import os
import sys
import yt_dlp
from yt_dlp.utils import DownloadError

def test_audio_download(youtube_url: str, output_dir: str = "data/raw", clip_duration: int = 15) -> str:
    # Ensure local directory exists dynamically
    os.makedirs(output_dir, exist_ok=True)

    # Output filename template
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    start_time = 0
    end_time = 15

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
        'force_keyframes_at_cuts': True,
    }

    try:
        print(f"Downloading clip ({clip_duration}s) from: {youtube_url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            video_id = info.get("id")
            downloaded_file = os.path.join(output_dir, f"{video_id}.wav")
            print(f"Successfully saved to: {downloaded_file}")
            return downloaded_file

    except DownloadError:
        print("\n[Error] Invalid or unreachable YouTube URL.")
        sys.exit(1)