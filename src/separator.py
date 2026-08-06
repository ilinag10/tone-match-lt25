import os
import subprocess
import sys

def isolate_guitar_stem(audio_path: str, output_dir: str = "data/processed") -> str:
    """
    Uses the 6-stem Demucs model (htdemucs_6s) to isolate the dedicated guitar track.
    Returns the path to the isolated guitar.wav stem.
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        "htdemucs_6s",
        "-o",
        output_dir,
        audio_path
    ]

    try:
        print(f"Running 6-stem Demucs on {audio_path}...")
        subprocess.run(cmd, check=True)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        guitar_stem_path = os.path.join(
            output_dir,
            "htdemucs_6s",
            base_name,
            "guitar.wav"
        )

        if os.path.exists(guitar_stem_path):
            print(f"Guitar stem isolated successfully: {guitar_stem_path}")
            return guitar_stem_path
        else:
            print("[Warning] 'guitar.wav' missing. Falling back to raw audio.")
            return audio_path

    except Exception as e:
        print(f"[Error] Demucs separation failed: {e}")
        return audio_path

def run_separator_cli(audio_path=None):
    if not audio_path:
        audio_path = input("Enter path to WAV file (e.g., data/raw/song.wav): ").strip()

    if not audio_path or not os.path.exists(audio_path):
        print(f"Error: File not found at '{audio_path}'")
        return None

    print(f"Starting separation for: {audio_path}")
    output_dir = isolate_guitar_stem(audio_path, output_dir="data/processed")
    return output_dir


if __name__ == "__main__":
        run_separator_cli()