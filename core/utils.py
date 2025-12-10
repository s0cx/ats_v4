# core/utils.py
import numpy as np
from typing import Tuple

def safe_norm(audio: np.ndarray) -> np.ndarray:
    """Normalize to max abs 1.0 safely (if possible)."""
    mx = np.max(np.abs(audio))
    if mx <= 0:
        return audio
    return audio / mx

def apply_adsr(audio: np.ndarray, sr: int, attack: float, decay: float, sustain_level: float, release: float) -> np.ndarray:
    """Apply ADSR envelope to 1-D audio buffer."""
    n = audio.shape[0]
    attack_n = int(max(0.0, attack) * sr)
    decay_n = int(max(0.0, decay) * sr)
    release_n = int(max(0.0, release) * sr)
    sustain_n = max(0, n - (attack_n + decay_n + release_n))

    env = np.zeros(n, dtype=np.float32)
    pos = 0
    if attack_n > 0:
        env[pos:pos+attack_n] = np.linspace(0.0, 1.0, attack_n, endpoint=False)
        pos += attack_n
    if decay_n > 0:
        env[pos:pos+decay_n] = np.linspace(1.0, sustain_level, decay_n, endpoint=False)
        pos += decay_n
    if sustain_n > 0:
        env[pos:pos+sustain_n] = sustain_level
        pos += sustain_n
    if release_n > 0:
        env[-release_n:] = np.linspace(sustain_level, 0.0, release_n, endpoint=False)
    return audio * env

def freq_map(height: int, min_f: float, max_f: float, scale: str = "log") -> np.ndarray:
    """Return array of frequencies per row (height entries)."""
    if scale == "linear":
        return np.linspace(min_f, max_f, height, dtype=float)
    else:
        # log scale, avoid log(0)
        return np.exp(np.linspace(np.log(max(1.0, min_f)), np.log(max_f), height)).astype(float)

def nice_duration_from_width(width_px: int) -> int:
    """Heuristic mapping from image width to seconds (keeps outputs reasonable)."""
    return max(2, int(round(width_px / 160)))
