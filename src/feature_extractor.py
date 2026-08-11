import os
import librosa
import numpy as np

def extract_raw_blueprint_features(audio_path: str, sample_rate: int = 22050) -> dict:

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    audio_signal, sample_rate = librosa.load(audio_path, sr=sample_rate, mono=True)
    frame_length = 2048
    hop_length = 512
    fps = sample_rate / hop_length

    # RMS (root mean square) energy
    rms_frames = librosa.feature.rms(y=audio_signal, frame_length=frame_length, hop_length=hop_length)[0]
    #create active playing mask (only frames > 10% of peak RMS)
    active_mask = rms_frames > (0.10 * np.max(rms_frames))
    if not np.any(active_mask):
        active_mask = np.ones_like(rms_frames, dtype=bool) #fallback if completely silent

    active_rms = float(np.mean(rms_frames[active_mask]))

    # Spectral Centroid (Center of Mass / Brightness)
    centroid_frames = librosa.feature.spectral_centroid(y=audio_signal, sr=sample_rate)[0]
    #filter out silent frames
    centroid = float(np.mean(centroid_frames[active_mask]))

    # Spectral Rolloff (85% Frequency Energy Ceiling)
    rolloff_frames = librosa.feature.spectral_rolloff(y=audio_signal, sr=sample_rate, roll_percent=0.85)[0]
    rolloff = float(np.mean(rolloff_frames[active_mask]))

    # Mel-Frequency Cepstral Coefficients (MFCCs 1-13)
    mfcc_frames = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
    mfccs_active = np.mean(mfcc_frames[:4, active_mask], axis=1)

    # Spectral Flatness (Noise-like Saturation vs Tonal Peaks)
    flatness_frames = librosa.feature.spectral_flatness(y=audio_signal)[0]
    flatness = np.mean(flatness_frames[active_mask])

    # Dynamic & crest factor features
    peak_val = float(np.max(np.abs(audio_signal)))
    crest_factor = float(peak_val / (active_rms + 1e-6))

    # FX Scores
    # Stompbox
    stomp_score = float(np.clip(flatness / (crest_factor + 1e-6), 0.0, 1.0))

    # Modulation
    N = len(rms_frames)
    rms_centered = rms_frames - np.mean(rms_frames)
    # length-normalized fast fourier transform to convert volume movements into oscillation rates
    rms_fft = np.abs(np.fft.rfft(rms_centered)) / N
    fft_freqs = np.fft.rfftfreq(N, d=1.0 / fps)

    # standard guitar mod is between 0.5 Hz - 5.0 Hz
    lfo_mask = (fft_freqs >= 0.5) & (fft_freqs <= 5.0)
    mod_score = float(np.clip(np.max(rms_fft[lfo_mask]) * 50.0, 0.0, 1.0)) if np.any(lfo_mask) else 0.0

    # Reverb
    # flip all values of active_mask -> locate frames where note stops ringing
    diff = np.diff(active_mask.astype(int))
    note_off_indices = np.where(diff == -1)[0]
    tail_frames = int(0.40 * fps) #0.40 sec = 400 ms window
    
    reverb_decays = []
    for idx in note_off_indices:
        if idx + tail_frames < len(rms_frames):
            tail = rms_frames[idx : idx + tail_frames]
            # Energy decay ratio = (mean energy in 400 ms) / (energy at note release)
            decay_ratio = np.mean(tail) / (rms_frames[idx] + 1e-6)
            reverb_decays.append(decay_ratio)

    reverb_score = float(np.clip(np.mean(reverb_decays) * 5.0, 0.0, 1.0)) if reverb_decays else 0.0

    # Delay Score and Delay Time
    autocorr = librosa.autocorrelate(rms_frames)
    # limit search window to standard delay times - 100-600 ms
    min_lag = int(0.10 * fps)  # 100 ms
    max_lag = int(0.60 * fps)  # 600 ms

    delay_score = 0.0
    delay_time_ms = 0.0

    # ensure clip is at least 600 ms long
    if len(autocorr) > max_lag:
        norm_autocorr = autocorr / (autocorr[0] + 1e-6)
        search_window = norm_autocorr[min_lag:max_lag]
        max_peak_idx = np.argmax(search_window)
        max_peak_lag = min_lag + max_peak_idx
        peak_val_corr = search_window[max_peak_idx]

        # filter out false positives
        if peak_val_corr > 0.15:
            delay_score = float(np.clip(peak_val_corr, 0.0, 1.0))
            delay_time_ms = float((max_peak_lag / fps) * 1000.0)
    

    return {
        #core
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "spectral_flatness": flatness,
        "mfccs": mfccs_active.tolist(),
        "active_rms": active_rms,
        "crest_factor": crest_factor,
        "rms_frames": rms_frames,
        #fx
        "stomp_score": stomp_score,
        "mod_score": mod_score,
        "reverb_score": reverb_score,
        "delay_score": delay_score,
        "delay_time_ms": delay_time_ms, 

    }

