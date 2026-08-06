from src.downloader import run_downloader_cli
from src.separator import run_separator_cli

def main():
    print("=== Step 1: Downloader ===")
    wav_path = run_downloader_cli()
    
    if not wav_path:
        print("Download failed or was cancelled. Exiting.")
        return

    print("\n=== Step 2: Stem Separation ===")
    run_separator_cli(audio_path=wav_path)

    print("\nPipeline complete! Check data/processed/ for stems.")

if __name__ == "__main__":
    main()
