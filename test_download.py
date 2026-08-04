import os
import yt_dlp

def test_audio_download(youtube_url: str, output_dir: str = "audio_downloads"):
    # Ensure local directory exists dynamically
    os.makedirs(output_dir, exist_ok=True)

    # Output filename template
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    start_time = 0
    end_time = 15

    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
        'force_keyframes_at_cuts': True,
    }

    print(f"\nDownloading 15-second clip from: {youtube_url}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    print("\nDownload and conversion complete!")


if __name__ == "__main__":
    user_url = input("Enter YouTube URL: ").strip()

    if user_url:
        test_audio_download(user_url)
    else:
        print("Error: no URL provided.")