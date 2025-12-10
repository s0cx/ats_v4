# core/presets.py
from dataclasses import dataclass, asdict
import json
from typing import Dict

@dataclass
class SettingsPreset:
    sample_rate: int = 44100
    duration: int = 0                 # 0 = auto
    min_freq: float = 80.0
    max_freq: float = 16000.0
    volume: float = 0.12
    stereo: bool = True
    rgb_mode: bool = False
    rgb_to_stereo: bool = False
    phase_random: bool = False
    attack: float = 0.01
    decay: float = 0.05
    sustain: float = 0.8
    release: float = 0.1
    brightness: float = 1.0
    contrast: float = 1.0
    invert: bool = False
    rotation: int = 0
    flip_h: bool = False
    flip_v: bool = False
    height: int = 256                 # vertical resolution / number of frequency bands
    freq_scale: str = "log"           # 'linear' or 'log'
    chunk_samples: int = 65536        # processing chunk size for memory management

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(d: Dict) -> "SettingsPreset":
        return SettingsPreset(**d)
