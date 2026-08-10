import os
import librosa
import numpy as np

LT25_AMP_MODELS = [
    "super clean",
    "champ",
    "deluxe dirt",
    "50s twin",
    "bassman",
    "princeton",
    "deluxe clean",
    "twin clean",
    "excelsior",
    "smalltone",
    "70s UK clean",
    "60s UK clean",
    "70s rock",
    "80s rock",
    "doom metal",
    "burn",
    "90s rock",
    "alt metal",
    "metal 2000",
    "super heavy",
]

def extract_raw_blueprint_features(audio_path: str, sample_rate: int = 22050) -> dict:

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    audio_signal, sample_rate = librosa.load(audio_path, sr=sample_rate, mono=True)
    frame_length = 2048
    hop_length = 512

    # RMS (root mean square) energy
    # LT25 MAPPING:
    #   1. Active-Frame Masking (isolates played notes from silence).
    #   2. Main Gain Knob scaling (translates dynamic signal volume to 1.0-10.0).
    #   3. Full unmasked frame array saved for downstream Reverb/Delay decay tracking.
    rms_frames = librosa.feature.rms(y=audio_signal, frame_length=frame_length, hop_length=hop_length)[0]
    #create active playing mask (only frames > 10% of peak RMS)
    active_mask = rms_frames > (0.10 * np.max(rms_frames))
    if not np.any(active_mask):
        active_mask = np.ones_like(rms_frames, dtype=bool) #fallback if completely silent

    active_rms = float(np.mean(rms_frames[active_mask]))

    # 1. Timbre & Brightness Metrics
    # Spectral Centroid (Center of Mass / Brightness)
    # LT25 MAPPING:
    #   1. Treble Knob (70% primary weight).
    #   2. Amp Model 'brightness_score' (differentiates bright chime from dark/warm amps).
    centroid_frames = librosa.feature.spectral_centroid(y=audio_signal, sr=sample_rate)[0]
    #filter out silent frames
    centroid = float(np.mean(centroid_frames[active_mask]))

    # Spectral Rolloff (85% Frequency Energy Ceiling)
    # LT25 MAPPING:
    #   1. Treble Knob (30% secondary weight for top-end air/chime).
    #   2. Amp Model 'brightness_score' (detects high-harmonic extension in bright amps).
    rolloff_frames = librosa.feature.spectral_rolloff(y=audio_signal, sr=sample_rate, roll_percent=0.85)[0]
    rolloff = float(np.mean(rolloff_frames[active_mask]))

    # Mel-Frequency Cepstral Coefficients (MFCCs 1-13)
    # LT25 MAPPING:
    #   1. MFCC 1 & 2 -> Bass Knob (low-end cabinet body and bass response).
    #   2. MFCC 3 & 4 -> Middle Knob & Amp Model 'mid_contour' (differentiates
    #      mid-scooped metal amps like 'super heavy' from mid-boosted '80s rock').
    mfcc_frames = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
    mfccs_active = np.mean(mfcc_frames[:4, active_mask], axis=1)

    # Spectral Flatness (Noise-like Saturation vs Tonal Peaks)
    # LT25 MAPPING:
    #   1. Amp Model 'gain_score' (differentiates clean tones from high-gain clipping).
    #   2. Main Gain Knob multiplier (drives pre-amp distortion level).
    flatness_frames = librosa.feature.spectral_flatness(y=audio_signal)[0]
    flatness = np.mean(flatness_frames[active_mask])

    # Dynamic & crest factor features
    peak_val = float(np.max(np.abs(audio_signal)))
    crest_factor = float(peak_val / (active_rms + 1e-6))

    return {
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "spectral_flatness": flatness,
        "mfccs": mfccs_active.tolist(),
        "active_rms": active_rms,
        "crest_factor": crest_factor,
        "rms_frames": rms_frames,
    }

