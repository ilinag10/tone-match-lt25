import os
import shutil
import tempfile
import json
import pandas as pd
import streamlit as st

from src.downloader import (
    download_audio,
    get_video_duration,
    parse_timestamp,
)
from src.feature_extractor import extract_raw_blueprint_features
from src.separator import run_separator_cli
from src.tone_matcher import ToneMatcher, clean_fx_val

# Page Setup
st.set_page_config(
    page_title="Mustang LT25 Tone Matcher", page_icon="🎸", layout="centered"
)

st.title("🎸 Mustang LT25 Tone Matcher")
st.markdown(
    "Extract audio signatures from YouTube clips or local files to match Fender Mustang LT25 amp presets."
)

tab_yt, tab_file = st.tabs(["YouTube Link", "File Upload"])

audio_target_path = None
stem_folder_path = None
temp_dir = tempfile.mkdtemp()

def format_fx_block(raw_val):
    """Parses JSON/dict FX string into clean readable text: 'Type (key: val, key: val)'"""
    if not raw_val or pd.isna(raw_val) or str(raw_val).strip().lower() in ["none", "nan"]:
        return "None"

    # Convert stringified JSON to dict if necessary
    data = raw_val
    if isinstance(raw_val, str):
        try:
            data = json.loads(raw_val.replace("'", '"'))
        except Exception:
            return str(raw_val)

    if isinstance(data, dict):
        fx_type = data.get("type", "Enabled")
        settings = data.get("settings", {})
        if settings:
            settings_str = ", ".join(
                [f"{k}: {v}" for k, v in settings.items()]
            )
            return f"**{fx_type}** ({settings_str})"
        return f"**{fx_type}**"

    return str(data)


try:
    # --- TAB 1: YOUTUBE INPUT ---
    with tab_yt:
        with st.form("yt_form"):
            yt_url = st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
            )
            col1, col2 = st.columns(2)
            with col1:
                timestamp_str = st.text_input(
                    "Start Timestamp",
                    value="00:00",
                    help="Format: MM:SS or HH:MM:SS",
                )
            with col2:
                duration_opt = st.radio(
                    "Clip Duration",
                    options=[30, 45, 60],
                    format_func=lambda x: f"{x} seconds",
                    horizontal=True,
                )

            submit_yt = st.form_submit_button("Match Tone from YouTube", type="primary")

        if submit_yt:
            if not yt_url:
                st.error("Please enter a valid YouTube URL.")
            else:
                # 1. Parse timestamp
                start_seconds = parse_timestamp(timestamp_str.strip())
                if start_seconds < 0:
                    st.error(
                        "Invalid timestamp format. Use HH:MM:SS or MM:SS (e.g., 01:15)."
                    )
                else:
                    # 2. Check total video duration
                    with st.spinner("Fetching video metadata..."):
                        total_duration = get_video_duration(yt_url)

                    if start_seconds + 30 > total_duration:
                        st.error(
                            f"Start time ({start_seconds}s) is too close to end of video ({total_duration}s)."
                        )
                    else:
                        # 3. Download trimmed clip via download_audio
                        with st.spinner("Downloading audio snippet via yt-dlp..."):
                            try:
                                audio_target_path = download_audio(
                                    youtube_url=yt_url,
                                    start_time=start_seconds,
                                    duration=duration_opt,
                                    output_dir=os.path.join(temp_dir, "raw"),
                                )
                            except Exception as e:
                                st.error(f"Download failed: {e}")

    # --- TAB 2: LOCAL FILE UPLOAD ---
    with tab_file:
        uploaded_file = st.file_uploader(
            "Upload Audio File (.wav, .mp3)", type=["wav", "mp3"]
        )
        if uploaded_file and st.button("Match Tone from File", type="primary"):
            save_path = os.path.join(temp_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            audio_target_path = save_path

    # --- PIPELINE PROCESSING & UI RESULTS ---
    if audio_target_path and os.path.exists(audio_target_path):
        st.divider()

        # Step 1: Demucs Stem Separation
        with st.spinner("Separating guitar stem with Demucs..."):
            guitar_stem = run_separator_cli(audio_path=audio_target_path)
            if isinstance(guitar_stem, (tuple, list)):
                guitar_stem = guitar_stem[0]

            if guitar_stem and os.path.exists(guitar_stem):
                stem_folder_path = os.path.dirname(guitar_stem)

        # Step 2: Feature Extraction
        with st.spinner("Extracting DSP audio features..."):
            target_features = extract_raw_blueprint_features(
                guitar_stem, sample_rate=22050
            )

        # Step 3: k-NN Tone Matching
        with st.spinner("Matching against anchor presets..."):
            matcher = ToneMatcher(n_neighbors=1)
            recipe = matcher.match_features(target_features)

        st.success("Analysis Complete!")

        # UI Recipe Output Display
        st.header(f"Matched Preset: {recipe['preset_name']}")
        st.caption(
            f"Amp Model: **{recipe['amp_model']}** | Match Distance: `{recipe['match_distance']:.4f}`"
        )

        col_knobs, col_fx = st.columns(2)

        with col_knobs:
            st.subheader("🎛️ Knob Settings")
            knobs_df = pd.DataFrame(
                list(recipe["knob_settings"].items()),
                columns=["Knob", "Setting"],
            )
            st.dataframe(knobs_df, hide_index=True, use_container_width=True)

        with col_fx:
            st.subheader("🔌 FX Blocks")

            formatted_fx = []
            for slot, raw_val in recipe["fx_blocks"].items():
                formatted_fx.append(
                    {
                        "Slot": slot.capitalize(),
                        "Effect & Settings": format_fx_block(raw_val),
                    }
                )

            fx_df = pd.DataFrame(formatted_fx)

            # Display clean table using Streamlit markdown rendering
            st.markdown(fx_df.to_markdown(index=False), unsafe_allow_html=True)

finally:
    # Cleanup temporary local execution cache
    if stem_folder_path and os.path.exists(stem_folder_path):
        shutil.rmtree(stem_folder_path, ignore_errors=True)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)