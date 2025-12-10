# audio/engine.py
import numpy as np
import threading
import os
from typing import Callable, Tuple

from scipy.io.wavfile import write as wav_write
from core.utils import safe_norm, apply_adsr, freq_map

# optional soundfile for streaming write/read
try:
    import soundfile as sf  # type: ignore
    SOUND_FILE_AVAILABLE = True
except Exception:
    SOUND_FILE_AVAILABLE = False

# optional sounddevice for playback (replaces simpleaudio)
try:
    import sounddevice as sd  # type: ignore
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False


class AudioEngine:
    """
    Audio engine with:
      - generate_from_image -> writes WAV to disk (chunked, memory-friendly)
      - generate_to_array   -> returns float32 numpy array (mono or stereo) for previews
      - play_array          -> simple blocking playback via sounddevice
      - play_streaming      -> chunked playback with frame_callback for live waterfall
    """

    def __init__(self, progress_callback: Callable[[float, str], None] = None):
        self.progress_callback = progress_callback or (lambda p, s: None)
        self._stop_flag = threading.Event()

    # ---------------- basic helpers ----------------
    def stop(self):
        self._stop_flag.set()

    def reset_stop(self):
        self._stop_flag.clear()

    def _report(self, pct: float, msg: str):
        try:
            self.progress_callback(float(pct), str(msg))
        except Exception:
            pass

    # ---------------- in-memory generation ----------------
    def generate_to_array(self, image_processor, preset) -> Tuple[np.ndarray, int]:
        """
        Generate full audio into memory (float32 -1..1). Good for quick previews and waterfall.
        Returns (audio_array, samplerate) where audio_array shape: (N,) mono or (N,2) stereo.
        """
        self.reset_stop()
        sr = int(preset.sample_rate)

        img = image_processor.transformed(
            rotation=preset.rotation,
            flip_h=preset.flip_h,
            flip_v=preset.flip_v,
            invert=preset.invert,
            brightness=preset.brightness,
            contrast=preset.contrast
        )

        # duration
        if preset.duration <= 0:
            duration = max(1, int(round(img.width / 160)))
        else:
            duration = int(preset.duration)
        total_samples = int(sr * duration)
        t = np.linspace(0, duration, total_samples, endpoint=False, dtype=np.float32)

        height = int(preset.height)
        freqs = freq_map(height, preset.min_freq, preset.max_freq, preset.freq_scale)

        width_now = max(1, img.width)
        x_src = np.linspace(0, width_now - 1, width_now)

        if preset.rgb_mode:
            r_arr, g_arr, b_arr = image_processor.to_rgb_arrays(img, height)
        else:
            img_arr = image_processor.to_grayscale_array(img, height)

        left = np.zeros(total_samples, dtype=np.float32)
        right = np.zeros(total_samples, dtype=np.float32)

        rng = np.random.default_rng(12345 if not preset.phase_random else None)
        ang = 2.0 * np.pi * freqs

        # iterate rows (height)
        interp_x = np.linspace(0, width_now - 1, total_samples)
        for y in range(height):
            if self._stop_flag.is_set():
                raise InterruptedError("Cancelled")

            if preset.rgb_mode:
                row_r = np.interp(interp_x, x_src, r_arr[y])
                row_g = np.interp(interp_x, x_src, g_arr[y])
                row_b = np.interp(interp_x, x_src, b_arr[y])

                phase = 0.0 if not preset.phase_random else float(rng.uniform(0, 2 * np.pi))
                sig = np.sin(ang[y] * t + phase).astype(np.float32)

                if preset.rgb_to_stereo:
                    left += (row_r * sig) + 0.5 * (row_g * sig)
                    right += (row_b * sig) + 0.5 * (row_g * sig)
                else:
                    mix = (row_r + row_g + row_b) / 3.0
                    left += mix * sig
                    right += mix * sig
            else:
                row = np.interp(interp_x, x_src, img_arr[y])
                phase = 0.0 if not preset.phase_random else float(rng.uniform(0, 2 * np.pi))
                sig = np.sin(ang[y] * t + phase).astype(np.float32)
                left += row * sig
                right += row * sig

            if (y % max(1, height // 6)) == 0:
                pct = (y / height) * 90.0
                self._report(pct, f"Building preview: row {y + 1}/{height}")

        # mixdown / ADSR / volume
        if not preset.stereo:
            mono = 0.5 * (left + right)
            mono = safe_norm(mono)
            mono = apply_adsr(mono, sr, preset.attack, preset.decay, preset.sustain, preset.release)
            mono *= preset.volume
            return mono.astype(np.float32), sr
        else:
            l = safe_norm(left)
            rch = safe_norm(right)
            l = apply_adsr(l, sr, preset.attack, preset.decay, preset.sustain, preset.release) * preset.volume
            rch = apply_adsr(rch, sr, preset.attack, preset.decay, preset.sustain, preset.release) * preset.volume
            stereo = np.stack((l.astype(np.float32), rch.astype(np.float32)), axis=1)
            return stereo, sr

    # ---------------- WAV-to-disk generation ----------------
    def generate_from_image(self, image_processor, out_path: str, preset) -> str:
        """
        Generate WAV to disk using chunked streaming for large outputs (memory-friendly).
        Behavior: prefer soundfile streaming; otherwise accumulate blocks and write via scipy.
        """
        self.reset_stop()
        sr = int(preset.sample_rate)
        self._report(2.0, "Preparing image...")

        img = image_processor.transformed(
            rotation=preset.rotation,
            flip_h=preset.flip_h,
            flip_v=preset.flip_v,
            invert=preset.invert,
            brightness=preset.brightness,
            contrast=preset.contrast
        )

        if preset.duration <= 0:
            duration = max(1, int(round(img.width / 160)))
        else:
            duration = int(preset.duration)
        total_samples = int(sr * duration)
        self._report(4.0, f"Duration: {duration}s ({total_samples} samples)")

        height = int(preset.height)
        freqs = freq_map(height, preset.min_freq, preset.max_freq, preset.freq_scale)
        width_now = max(1, img.width)
        x_src = np.linspace(0, width_now - 1, width_now)

        if preset.rgb_mode:
            r_arr, g_arr, b_arr = image_processor.to_rgb_arrays(img, height)
        else:
            img_arr = image_processor.to_grayscale_array(img, height)

        chunk = int(min(preset.chunk_samples, total_samples))
        channels = 2 if preset.stereo else 1

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        use_soundfile = SOUND_FILE_AVAILABLE
        if use_soundfile:
            writer = sf.SoundFile(out_path, mode="w", samplerate=sr, channels=channels, subtype="PCM_16")
        else:
            blocks = []

        rng = np.random.default_rng(12345 if not preset.phase_random else None)
        ang_freqs = 2.0 * np.pi * freqs

        samples_done = 0
        full_xtarget = np.linspace(0, width_now - 1, total_samples, dtype=np.float32)

        while samples_done < total_samples:
            if self._stop_flag.is_set():
                self._report(0.0, "Cancelled")
                if use_soundfile:
                    writer.close()
                raise InterruptedError("Generation cancelled")

            this_chunk = min(chunk, total_samples - samples_done)
            t0 = samples_done / sr
            t = np.linspace(t0, t0 + (this_chunk / sr), this_chunk, endpoint=False, dtype=np.float32)

            x_target = full_xtarget[samples_done:samples_done + this_chunk]

            left_chunk = np.zeros(this_chunk, dtype=np.float32)
            right_chunk = np.zeros(this_chunk, dtype=np.float32)

            for y in range(height):
                if preset.rgb_mode:
                    row_r = np.interp(x_target, x_src, r_arr[y])
                    row_g = np.interp(x_target, x_src, g_arr[y])
                    row_b = np.interp(x_target, x_src, b_arr[y])

                    phase = 0.0 if not preset.phase_random else float(rng.uniform(0, 2 * np.pi))
                    sig = np.sin(ang_freqs[y] * t + phase).astype(np.float32)

                    if preset.rgb_to_stereo:
                        left_chunk += (row_r * sig) + 0.5 * (row_g * sig)
                        right_chunk += (row_b * sig) + 0.5 * (row_g * sig)
                    else:
                        mix = (row_r + row_g + row_b) / 3.0
                        left_chunk += mix * sig
                        right_chunk += mix * sig
                else:
                    row = np.interp(x_target, x_src, img_arr[y])
                    phase = 0.0 if not preset.phase_random else float(rng.uniform(0, 2 * np.pi))
                    sig = np.sin(ang_freqs[y] * t + phase).astype(np.float32)
                    left_chunk += row * sig
                    right_chunk += row * sig

                if (y % max(1, height // 6)) == 0:
                    pct = 5.0 + ((samples_done + (y / height) * this_chunk) / total_samples) * 85.0
                    self._report(pct, f"Synth rows {y + 1}/{height}")

            if not preset.stereo:
                mono = 0.5 * (left_chunk + right_chunk)
                mono = safe_norm(mono)
                mono = apply_adsr(mono, sr, preset.attack, preset.decay, preset.sustain, preset.release)
                mono *= preset.volume
                int_block = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
                if use_soundfile:
                    writer.write(int_block)
                else:
                    blocks.append(int_block)
            else:
                l = safe_norm(left_chunk)
                rch = safe_norm(right_chunk)
                l = apply_adsr(l, sr, preset.attack, preset.decay, preset.sustain, preset.release) * preset.volume
                rch = apply_adsr(rch, sr, preset.attack, preset.decay, preset.sustain, preset.release) * preset.volume
                stereo_block = np.stack((l, rch), axis=1)
                int_block = (np.clip(stereo_block, -1.0, 1.0) * 32767.0).astype(np.int16)
                if use_soundfile:
                    writer.write(int_block)
                else:
                    blocks.append(int_block)

            samples_done += this_chunk
            pct = 5.0 + (samples_done / total_samples) * 90.0
            self._report(pct, f"Rendered {samples_done}/{total_samples}")

        if use_soundfile:
            writer.close()
        else:
            data = np.concatenate(blocks, axis=0)
            wav_write(out_path, sr, data)

        self._report(100.0, f"Saved {os.path.basename(out_path)}")
        return out_path

    # ---------------- simple playback using sounddevice ----------------
    def play_array(self, audio: np.ndarray, sr: int):
        """
        Real-time playback using sounddevice (blocking).
        Accepts mono (N,) or stereo (N,2) float32/-1..1 or other dtype.
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "sounddevice is not installed. Install with:\n\npip install sounddevice"
            )

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        mx = float(np.max(np.abs(audio))) if audio.size > 0 else 1.0
        if mx > 1.0:
            audio = audio / mx

        sd.play(audio, samplerate=sr, blocking=True)

    # ---------------- streaming playback for live waterfall ----------------
    def play_streaming(self, audio: np.ndarray, sr: int, chunk_size=2048, frame_callback=None):
        """
        Stream audio in chunks with a callback.

        frame_callback(chunk, sr) is called from the audio thread for each chunk.
        The callback MUST be light; heavy work (FFT, GUI updates) should be moved
        to another thread or scheduled via a queue in the GUI.
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "sounddevice is not installed. Install with:\n\npip install sounddevice"
            )

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        mx = float(np.max(np.abs(audio))) if audio.size > 0 else 1.0
        if mx > 1.0:
            audio = audio / mx

        if audio.ndim == 1:
            channels = 1
        elif audio.ndim == 2:
            channels = audio.shape[1]
        else:
            raise ValueError("Audio array must be 1D (mono) or 2D (stereo).")

        index = 0
        total = len(audio)

        def callback(outdata, frames, time_info, status):
            nonlocal index
            if status:
                print("SoundDevice status:", status)

            end = index + frames
            if end > total:
                end = total

            chunk = audio[index:end]

            if len(chunk) < frames:
                pad_shape = (frames - len(chunk),) if channels == 1 else (frames - len(chunk), channels)
                pad = np.zeros(pad_shape, dtype=np.float32)
                chunk = np.concatenate((chunk, pad), axis=0)

            outdata[:] = chunk

            if frame_callback is not None:
                try:
                    frame_callback(chunk, sr)
                except Exception as e:
                    print("Frame callback error:", e)

            index = end
            if index >= total:
                raise sd.CallbackStop

        with sd.OutputStream(
            samplerate=sr,
            channels=channels,
            blocksize=chunk_size,
            callback=callback
        ):
            sd.sleep(int(1000 * (len(audio) / sr)))
