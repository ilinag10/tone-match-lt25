import os
import sys
import yt_dlp
from yt_dlp.utils import DownloadError
from datetime import datetime, timedelta

def download_audio(youtube_url: str, start_time: int, duration: int, output_dir: str = "data/raw") -> str:
    # Ensure local directory exists dynamically
    os.makedirs(output_dir, exist_ok=True)

    # Output filename template
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    end_time = start_time + duration

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        # Force yt-dlp to avoid ANDROID_VR and use standard web clients
        'extractor_args': {
            'youtube': {
                'player_client': ['web', 'mweb', 'android']
            }
        },
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        # Fix for FFmpeg HTTP 403 / header propagation
        'external_downloader_args': {
            'ffmpeg': ['-headers', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)']
        },
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,

        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
        'force_keyframes_at_cuts': True,
    }

    print(f"\nAttempting download for: {youtube_url}...")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            filename = ydl.prepare_filename(info)
            wav_path = os.path.splitext(filename)[0] + ".wav"
        print("\nDownload and conversion complete!")
        return wav_path

    except DownloadError:
        print("\n[Error] Invalid or unreachable YouTube URL.")
        sys.exit(1)

def get_clip_duration():
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
            get_clip_duration()

def get_video_duration(user_url: str):
    """Extracts total video duration in seconds without downloading."""
    ydl_opts = {'quiet': True, 'skip_download': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(user_url, download=False)
            return int(info.get('duration', 0))

    except DownloadError:
        print("\n[Error] Invalid or unreachable YouTube URL. Please check the link.")
        sys.exit(1)

def get_video_information(user_url: str):
    total_video_duration = get_video_duration(user_url)
    while True:
        start_timestamp = input("Enter start timestamp in format HH:MM:SS or MM:SS : ").strip()
        start_seconds = parse_timestamp(start_timestamp)

        if(start_seconds < 0):
            print("Error: Invalid format. Please use HH:MM:SS or MM:SS (e.g., 01:15).\n")
            continue
        elif(start_seconds + 30 > total_video_duration):
            print(f"Error: Start time ({start_seconds}s) exceeds total video length ({total_video_duration}s).\n")
            continue
        break
    return(total_video_duration, start_seconds)

def parse_timestamp(timestamp_str: str) -> int:
    """Helper to parse HH:MM:SS or MM:SS into total seconds."""
    formats = ["%H:%M:%S", "%M:%S"]
    for fmt in formats:
        try:
            dt = datetime.strptime(timestamp_str, fmt)
            delta = timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second)
            return int(delta.total_seconds())
        except ValueError:
            continue
    return -1

def run_downloader_cli():
    user_url = input("Enter YouTube URL: ").strip()
    if not user_url:
        print("Error: no URL provided.")
        sys.exit(1)
    

    print("Fetching video information...")
    total_video_duration, start_seconds = get_video_information(user_url)

    duration = get_clip_duration()

    wav_path = download_audio(user_url, start_seconds, duration)
    return wav_path



if __name__ == "__main__":
    run_downloader_cli()