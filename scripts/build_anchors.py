import json
import os
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from anchor_data import ANCHOR_PRESETS

# =============================================================================
# VALIDATION SCHEMAS & CONSTRAINTS
# =============================================================================


VALID_AMP_MODELS = {"Super Clean", "Champ", "Deluxe Dirt", "50s Twin", "Bassman", 
                    "Princeton", "Deluxe Cln", "Twin Clean", "Excelsior", "Smalltone", 
                    "70s Uk Cln", "60s Uk Cln", "70s Rock", "80s Rock", "Doom Metal", 
                    "Burn", "90s Rock", "Alt Metal", "Metal 2000", "Super Heavy"}

STOMP_SCHEMAS = {
    "None": [],
    "Overdrive": ["level", "gain", "low", "mid", "high"],
    "Blues Drive": ["level", "gain", "tone", "blend"],
    "Myth Drive": ["level", "gain", "treble"],
    "Rock Dirt": ["level", "distort", "filter"],
    "Fuzz": ["level", "gain", "vari"],  # vari: {"tight", "normal", "loose"}
    "Big Fuzz": ["level", "tone", "sustain"],
    "Octobot": ["direct", "down", "sizzle"],
    "Compressor": ["type"],  # type: {"low", "mid", "high", "max"}
    "Sustain": ["sense", "level"],
    "Metal Gate": ["thresh"],
    "5 Band EQ": ["low", "low_mid", "mid", "high_mid", "high"],  # [-12.0, 12.0], step 0.5
}

MOD_SCHEMAS = {
    "None": [],
    "Chorus": ["level", "speed", "depth"],
    "Flanger": ["level", "speed", "depth", "feedback"],
    "Vibratone": ["level", "speed", "depth", "feedback", "phase"],
    "Tremolo": ["level", "speed"],
    "Phaser": ["level", "speed", "depth"],
    "Step Filter": ["level", "speed", "resonate", "low freq", "hi freq"],
    "Touch Wah": ["level", "thresh", "mode", "filter", "peak"],
}

DELAY_SCHEMAS = {
    "None": [],
    "Delay": ["level", "time", "feedback", "tone"],
    "Reverse": ["level", "time", "feedback"],
    "Echo": ["level", "time", "feedback", "wow"],
}

REVERB_SCHEMAS = {
    "None": [],
    "Large Hall": ["level", "decay", "tone"],
    "Small Room": ["level", "decay", "tone"],
    "Spring 65": ["level", "decay", "tone"],
    "Plate": ["level", "decay", "tone"],
    "Arena": ["level", "decay", "tone"],
}




# =============================================================================
# VALIDATION ENGINE
# =============================================================================

def validate_fx_block(fx_data: Union[str, Dict[str, Any]], category_name: str, schemas: Dict[str, List[str]]) -> None:
    """Validates an FX block against its allowed types and required setting keys."""
    if fx_data == "None":
        return

    if not isinstance(fx_data, dict) or "type" not in fx_data or "settings" not in fx_data:
        raise ValueError(f"[{category_name}] Must be 'None' or dict with 'type' and 'settings'. Got: {fx_data}")

    fx_type = fx_data["type"]
    settings = fx_data["settings"]

    if fx_type not in schemas:
        raise ValueError(f"[{category_name}] Invalid type '{fx_type}'. Allowed types: {list(schemas.keys())}")

    expected_keys = set(schemas[fx_type])
    given_keys = set(settings.keys())

    if expected_keys != given_keys:
        raise ValueError(
            f"[{category_name} -> {fx_type}] Key mismatch.\n"
            f"Expected keys: {sorted(list(expected_keys))}\n"
            f"Given keys:    {sorted(list(given_keys))}"
        )

    # Specific Enum / Range Validations
    if fx_type == "Fuzz" and settings.get("vari") not in {"tight", "normal", "loose"}:
        raise ValueError(f"[Fuzz] 'vari' must be 'tight', 'normal', or 'loose'. Got: {settings.get('vari')}")

    if fx_type == "Compressor" and settings.get("type") not in {"low", "mid", "high", "max"}:
        raise ValueError(f"[Compressor] 'type' must be 'low', 'mid', 'high', or 'max'. Got: {settings.get('type')}")

    if fx_type == "5 band eq":
        for eq_key, eq_val in settings.items():
            if not (-12.0 <= float(eq_val) <= 12.0) or (float(eq_val) % 0.5 != 0):
                raise ValueError(f"[5 band eq] '{eq_key}' must be between -12.0 and 12.0 in step 0.5. Got: {eq_val}")

    if fx_type == "Touch wah":
        if settings.get("mode") not in {"LO-UP", "LO-DOWN", "HI-UP", "HI-DOWN"}:
            raise ValueError(f"[Touch wah] Invalid mode: {settings.get('mode')}")
        if settings.get("filter") not in {"LPF", "BPF", "HPF"}:
            raise ValueError(f"[Touch wah] Invalid filter: {settings.get('filter')}")


def validate_anchor_entry(entry: Dict[str, Any]) -> None:
    """Validates a single anchor tone record before adding it to the dataset."""
    required_fields = [
        "song_title", "artist", "youtube_url", "start_time", "amp_model", "gain", "volume", 
        "treble", "middle", "bass", "stomp_fx", "mod_fx", "delay_fx", "reverb_fx"
    ]

    for field in required_fields:
        if field not in entry:
            raise KeyError(f"Missing required field '{field}' in entry: {entry.get('song_title', 'Unknown')}")

    # Validate Amp Model
    if entry["amp_model"] not in VALID_AMP_MODELS:
        raise ValueError(f"[{entry['amp_model']}] Invalid amp model '{entry['amp_model']}'. Must be one of {VALID_AMP_MODELS}")

    # Validate Core Amp Knobs (1.0 - 10.0 range)
    for knob in ["gain", "treble", "middle", "bass"]:
        val = entry[knob]
        if not (1.0 <= float(val) <= 10.0):
            raise ValueError(f"[{entry['song_title']}] '{knob}' knob must be between 1.0 and 10.0. Got: {val}")

    validate_fx_block(entry["stomp_fx"], "Stomp", STOMP_SCHEMAS)
    validate_fx_block(entry["mod_fx"], "Mod", MOD_SCHEMAS)
    validate_fx_block(entry["delay_fx"], "Delay", DELAY_SCHEMAS)
    validate_fx_block(entry["reverb_fx"], "Reverb", REVERB_SCHEMAS)


# =============================================================================
# DATASET BUILDER & EXPORTER
# =============================================================================

class AnchorDatasetBuilder:
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def add_anchor(self, **kwargs: Any) -> None:
        """Validates and appends a single anchor tone to the in-memory builder."""
        validate_anchor_entry(kwargs)
        self.entries.append(kwargs)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts entries to a pandas DataFrame with serialized JSON FX columns."""
        processed_rows = []
        for row in self.entries:
            formatted_row = row.copy()
            for fx_col in ["stomp_fx", "mod_fx", "delay_fx", "reverb_fx"]:
                val = formatted_row[fx_col]
                formatted_row[fx_col] = json.dumps(val) if isinstance(val, dict) else "None"
            processed_rows.append(formatted_row)
        
        return pd.DataFrame(processed_rows)

    def export_csv(self, output_path: str = "data/anchors.csv") -> None:
        """Exports validated anchor records to CSV format."""
        df = self.to_dataframe()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Successfully validated and wrote {len(df)} anchor tones to: {output_path}")


# =============================================================================
# DEMONSTRATION / USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    builder = AnchorDatasetBuilder()


    print(f"Loading {len(ANCHOR_PRESETS)} preset definitions...")
    for preset in ANCHOR_PRESETS:
        builder.add_anchor(**preset)


    # Save to CSV
    builder.export_csv("data/anchors.csv")