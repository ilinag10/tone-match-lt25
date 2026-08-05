import os
import sys
import yt_dlp
from yt_dlp.utils import DownloadError
from datetime import datetime, timedelta

def test_audio_download(youtube_url: str, start_time: int, duration: int, output_dir: str = "audio_downloads"):
    # Ensure local directory exists dynamically
    os.makedirs(output_dir, exist_ok=True)

    # Output filename template
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    end_time = start_time + duration

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

    print(f"\nAttempting download for: {youtube_url}...")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        print("\nDownload and conversion complete!")
        
    except DownloadError:
        print("\n[Error] Invalid or unreachable YouTube URL. Please check the link and try again.")
        sys.exit(1)


def duration_helper():
    val = input("Enter 0 for 30 seconds, 1 for 45 seconds, or 2 for 60 seconds:").strip()
    match val:
        case "0":
            return 30
        case "1": 
            return 45
        case "2":
            return 60
        case _:
            print("Error: invalid input. Please try again.")
            duration_helper()


if __name__ == "__main__":
    user_url = input("Enter YouTube URL: ").strip()
    start_timestamp = input("Enter start timestamp in format xx:xx:xx with no spaces: ").strip()
    dt = datetime.strptime(start_timestamp, "%H:%M:%S")
    delta = timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second)
    start_seconds = int(delta.total_seconds())
    duration = duration_helper()


    if user_url:
        test_audio_download(user_url, start_seconds, duration)
    else:
        print("Error: no URL provided.")
