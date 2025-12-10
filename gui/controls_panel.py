# gui/controls_panel.py
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading, os, time, sys, subprocess
import numpy as np

from core.presets import SettingsPreset
from image.processor import ImageProcessor
from audio.engine import AudioEngine
from gui.spectrogram_panel import SpectrogramPanel
from PIL import Image, ImageTk

# optional libs
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except Exception:
    SOUNDFILE_AVAILABLE = False

try:
    from scipy.signal import spectrogram
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


class ControlsPanel(ctk.CTkScrollableFrame):
    def __init__(self, master, width=580, height=760,
                 preview_panel=None,
                 spectrogram_panel: SpectrogramPanel | None = None):
        super().__init__(master, width=width, height=height)

        self.preview_panel = preview_panel
        self.spectrogram_panel = spectrogram_panel

        self.preset = SettingsPreset()
        self.processor: ImageProcessor | None = None
        self.image_path: str | None = None
        self.output_path: str | None = None
        self.engine = AudioEngine(progress_callback=self._on_progress)
        self.gen_thread: threading.Thread | None = None
        self.last_generated: str | None = None

        self._build_controls()

    # ------------------------------------------------------------
    # UI BUILD
    # ------------------------------------------------------------
    def _build_controls(self):
        pad = 8

        # FILES
        file_frame = ctk.CTkFrame(self)
        file_frame.pack(fill="x", padx=pad, pady=(pad, 4))
        ctk.CTkButton(file_frame, text="Open Image", command=self.select_image).pack(side="left", padx=6)
        ctk.CTkButton(file_frame, text="Select Output", command=self.select_output).pack(side="left", padx=6)
        ctk.CTkButton(file_frame, text="Generate (BG)", command=self.start_generation_thread).pack(side="left", padx=6)
        ctk.CTkButton(file_frame, text="Play Last", command=self.play_last).pack(side="left", padx=6)

        # SAMPLE RATE / DURATION
        io_frame = ctk.CTkFrame(self)
        io_frame.pack(fill="x", padx=pad, pady=4)
        ctk.CTkLabel(io_frame, text="Sample rate").grid(row=0, column=0, padx=6, pady=2, sticky="w")
        self.sample_rate_var = tk.IntVar(value=self.preset.sample_rate)
        ctk.CTkOptionMenu(io_frame, values=["22050", "32000", "44100", "48000", "96000"],
                          variable=self.sample_rate_var).grid(row=0, column=1, padx=6)
        ctk.CTkLabel(io_frame, text="Duration (s, 0=auto)").grid(row=1, column=0, padx=6, pady=2, sticky="w")
        self.duration_var = tk.IntVar(value=self.preset.duration)
        ctk.CTkEntry(io_frame, textvariable=self.duration_var, width=80).grid(row=1, column=1, padx=6)

        # FREQ RANGE
        freq_frame = ctk.CTkFrame(self)
        freq_frame.pack(fill="x", padx=pad, pady=4)
        ctk.CTkLabel(freq_frame, text="Min Hz").grid(row=0, column=0, padx=6, pady=2, sticky="w")
        self.min_freq_var = tk.IntVar(value=int(self.preset.min_freq))
        ctk.CTkEntry(freq_frame, textvariable=self.min_freq_var, width=80).grid(row=0, column=1, padx=6)
        ctk.CTkLabel(freq_frame, text="Max Hz").grid(row=1, column=0, padx=6, pady=2, sticky="w")
        self.max_freq_var = tk.IntVar(value=int(self.preset.max_freq))
        ctk.CTkEntry(freq_frame, textvariable=self.max_freq_var, width=80).grid(row=1, column=1, padx=6)

        # MIXING
        mix_frame = ctk.CTkFrame(self)
        mix_frame.pack(fill="x", padx=pad, pady=4)
        ctk.CTkLabel(mix_frame, text="Volume").grid(row=0, column=0, padx=6, pady=2, sticky="w")
        self.volume_var = tk.DoubleVar(value=self.preset.volume)
        ctk.CTkEntry(mix_frame, textvariable=self.volume_var, width=80).grid(row=0, column=1, padx=6)
        self.stereo_var = tk.BooleanVar(value=self.preset.stereo)
        ctk.CTkCheckBox(mix_frame, text="Stereo", variable=self.stereo_var).grid(row=0, column=2, padx=6)
        self.rgb_var = tk.BooleanVar(value=self.preset.rgb_mode)
        ctk.CTkCheckBox(mix_frame, text="RGB mode",
                        variable=self.rgb_var,
                        command=self._on_transform_change).grid(row=0, column=3, padx=6)
        self.rgb_to_stereo_var = tk.BooleanVar(value=self.preset.rgb_to_stereo)
        ctk.CTkCheckBox(mix_frame, text="RGB→Stereo", variable=self.rgb_to_stereo_var).grid(row=1, column=0, padx=6)

        # ADSR
        adsr_frame = ctk.CTkFrame(self)
        adsr_frame.pack(fill="x", padx=pad, pady=4)
        self.attack_var = tk.DoubleVar(value=self.preset.attack)
        self.decay_var = tk.DoubleVar(value=self.preset.decay)
        self.sustain_var = tk.DoubleVar(value=self.preset.sustain)
        self.release_var = tk.DoubleVar(value=self.preset.release)
        ctk.CTkLabel(adsr_frame, text="Attack").grid(row=0, column=0, padx=6, pady=2, sticky="w")
        ctk.CTkEntry(adsr_frame, textvariable=self.attack_var, width=60).grid(row=0, column=1, padx=6)
        ctk.CTkLabel(adsr_frame, text="Decay").grid(row=0, column=2, padx=6, sticky="w")
        ctk.CTkEntry(adsr_frame, textvariable=self.decay_var, width=60).grid(row=0, column=3, padx=6)
        ctk.CTkLabel(adsr_frame, text="Sustain").grid(row=1, column=0, padx=6, sticky="w")
        ctk.CTkEntry(adsr_frame, textvariable=self.sustain_var, width=60).grid(row=1, column=1, padx=6)
        ctk.CTkLabel(adsr_frame, text="Release").grid(row=1, column=2, padx=6, sticky="w")
        ctk.CTkEntry(adsr_frame, textvariable=self.release_var, width=60).grid(row=1, column=3, padx=6)

        # CREATIVE / SPECTROGRAM TOGGLES
        creative_frame = ctk.CTkFrame(self)
        creative_frame.pack(fill="x", padx=pad, pady=4)
        self.phase_var = tk.BooleanVar(value=self.preset.phase_random)
        ctk.CTkCheckBox(creative_frame, text="Phase randomization",
                        variable=self.phase_var).grid(row=0, column=0, padx=6)

        ctk.CTkLabel(creative_frame, text="Freq scale:").grid(row=0, column=1, padx=6)
        self.freq_scale_var = tk.StringVar(value=self.preset.freq_scale)
        ctk.CTkOptionMenu(creative_frame, values=["log", "linear"],
                          variable=self.freq_scale_var).grid(row=0, column=2, padx=6)

        # Spectrogram options (B + live toggle)
        self.spec_static_var = tk.BooleanVar(value=True)
        self.spec_waterfall_var = tk.BooleanVar(value=False)  # default: static only
        self.spec_play_audio_var = tk.BooleanVar(value=False)

        ctk.CTkCheckBox(creative_frame, text="Static Spectrogram",
                        variable=self.spec_static_var).grid(row=1, column=0, padx=6, pady=2, sticky="w")
        ctk.CTkCheckBox(creative_frame, text="Animated Waterfall (live)",
                        variable=self.spec_waterfall_var).grid(row=1, column=1, padx=6, pady=2, sticky="w")

        # audio during preview is disabled for stability (no simpleaudio)
        ctk.CTkCheckBox(creative_frame,
                        text="Play audio during preview (disabled)",
                        variable=self.spec_play_audio_var,
                        state="disabled").grid(row=1, column=2, padx=6, pady=2, sticky="w")

        ctk.CTkButton(creative_frame, text="Spectrogram Preview",
                      command=self.preview_waterfall).grid(row=2, column=0, padx=6, pady=4, sticky="w")
        ctk.CTkButton(creative_frame, text="Preview Waveform",
                      command=self.preview_waveform).grid(row=2, column=1, padx=6, pady=4, sticky="w")

        # TRANSFORMS
        trans_frame = ctk.CTkFrame(self)
        trans_frame.pack(fill="x", padx=pad, pady=4)
        ctk.CTkButton(trans_frame, text="Rotate ⟲",
                      command=lambda: self._mod_rotation(-90)).pack(side="left", padx=6)
        ctk.CTkButton(trans_frame, text="Rotate ⟳",
                      command=lambda: self._mod_rotation(90)).pack(side="left", padx=6)
        self.rotate_var = tk.IntVar(value=self.preset.rotation)
        self.flip_h_var = tk.BooleanVar(value=self.preset.flip_h)
        self.flip_v_var = tk.BooleanVar(value=self.preset.flip_v)
        self.invert_var = tk.BooleanVar(value=self.preset.invert)
        ctk.CTkCheckBox(trans_frame, text="Flip H", variable=self.flip_h_var,
                        command=self._on_transform_change).pack(side="left", padx=6)
        ctk.CTkCheckBox(trans_frame, text="Flip V", variable=self.flip_v_var,
                        command=self._on_transform_change).pack(side="left", padx=6)
        ctk.CTkCheckBox(trans_frame, text="Invert", variable=self.invert_var,
                        command=self._on_transform_change).pack(side="left", padx=6)

        # PROGRESS + LOG
        self.progress = ctk.CTkProgressBar(self, width=520)
        self.progress.set(0.0)
        self.progress.pack(padx=pad, pady=8)
        self.status_label = ctk.CTkLabel(self, text="Idle")
        self.status_label.pack(padx=pad, pady=(0, 8))
        self.logbox = tk.Text(self, height=8, bg="#111", fg="#ddd", state="disabled")
        self.logbox.pack(fill="both", padx=pad, pady=4)

    # ------------------------------------------------------------
    # LOGGING / PROGRESS
    # ------------------------------------------------------------
    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        try:
            self.logbox.configure(state="normal")
            self.logbox.insert("end", line + "\n")
            self.logbox.see("end")
            self.logbox.configure(state="disabled")
        except Exception:
            pass

    def _on_progress(self, pct: float, status: str):
        try:
            self.progress.set(max(0.0, min(1.0, pct / 100.0)))
            self.status_label.configure(text=status)
            self._log(status)
        except Exception:
            pass

    # ------------------------------------------------------------
    # SYNC PRESET
    # ------------------------------------------------------------
    def _sync_ui_to_preset(self):
        self.preset.sample_rate = int(self.sample_rate_var.get())
        self.preset.duration = int(self.duration_var.get())
        self.preset.min_freq = float(self.min_freq_var.get())
        self.preset.max_freq = float(self.max_freq_var.get())
        self.preset.volume = float(self.volume_var.get())
        self.preset.stereo = bool(self.stereo_var.get())
        self.preset.rgb_mode = bool(self.rgb_var.get())
        self.preset.rgb_to_stereo = bool(self.rgb_to_stereo_var.get())
        self.preset.phase_random = bool(self.phase_var.get())
        self.preset.attack = float(self.attack_var.get())
        self.preset.decay = float(self.decay_var.get())
        self.preset.sustain = float(self.sustain_var.get())
        self.preset.release = float(self.release_var.get())
        self.preset.invert = bool(self.invert_var.get())
        self.preset.rotation = int(self.rotate_var.get())
        self.preset.flip_h = bool(self.flip_h_var.get())
        self.preset.flip_v = bool(self.flip_v_var.get())
        self.preset.freq_scale = str(self.freq_scale_var.get())

    # ------------------------------------------------------------
    # FILE OPS
    # ------------------------------------------------------------
    def select_image(self):
        p = filedialog.askopenfilename(
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
        )
        if not p:
            return
        self.image_path = p
        self.processor = ImageProcessor(p)
        self._log(f"Image loaded: {p}")
        self.preview_refresh()

    def select_output(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav"), ("FLAC", "*.flac")]
        )
        if not p:
            return
        self.output_path = p
        self._log(f"Output: {p}")

    # ------------------------------------------------------------
    # PREVIEW IMAGE
    # ------------------------------------------------------------
    def preview_refresh(self):
        if not self.processor:
            return
        self._sync_ui_to_preset()
        try:
            transformed = self.processor.transformed(
                rotation=self.preset.rotation,
                flip_h=self.preset.flip_h,
                flip_v=self.preset.flip_v,
                invert=self.preset.invert,
                brightness=self.preset.brightness,
                contrast=self.preset.contrast
            )
            if self.preview_panel:
                self.preview_panel.show_image(transformed)
            self._log("Preview refreshed")
        except Exception as e:
            self._log(f"Preview error: {e}")

    def _on_transform_change(self):
        self.preview_refresh()

    def _mod_rotation(self, delta: int):
        self.rotate_var.set((self.rotate_var.get() + delta) % 360)
        self.preview_refresh()

    # ------------------------------------------------------------
    # GENERATION
    # ------------------------------------------------------------
    def start_generation_thread(self):
        if not self.processor:
            messagebox.showwarning("No image", "Open an image first.")
            return
        if not self.output_path:
            messagebox.showwarning("No output", "Select an output path first.")
            return
        if self.gen_thread and self.gen_thread.is_alive():
            messagebox.showinfo("Busy", "Generation already running.")
            return

        self._sync_ui_to_preset()
        self.progress.set(0.0)
        self.status_label.configure(text="Starting generation...")

        def worker():
            try:
                self.engine.reset_stop()
                out = self.engine.generate_from_image(self.processor, self.output_path, self.preset)
                self.last_generated = out
                self._log(f"Generated: {out}")
            except InterruptedError:
                self._log("Generation cancelled")
            except Exception as e:
                self._log(f"Generation error: {e}")
                try:
                    messagebox.showerror("Error", str(e))
                except Exception:
                    pass

        self.gen_thread = threading.Thread(target=worker, daemon=True)
        self.gen_thread.start()
        self._log("Generation thread started")

    def cancel_generation(self):
        if self.gen_thread and self.gen_thread.is_alive():
            self.engine.stop()
            self._log("Requested cancel")

    # ------------------------------------------------------------
    # WAVEFORM PREVIEW
    # ------------------------------------------------------------
    def preview_waveform(self):
        if not self.processor:
            messagebox.showwarning("No image", "Open an image first.")
            return

        self._sync_ui_to_preset()
        tmp_preset = SettingsPreset(
            sample_rate=22050,
            duration=3 if self.duration_var.get() == 0 else min(6, max(1, int(self.duration_var.get()))),
            min_freq=self.preset.min_freq,
            max_freq=self.preset.max_freq,
            volume=self.preset.volume,
            stereo=self.preset.stereo,
            rgb_mode=self.preset.rgb_mode,
            rgb_to_stereo=self.preset.rgb_to_stereo,
            phase_random=self.preset.phase_random,
            attack=self.preset.attack,
            decay=self.preset.decay,
            sustain=self.preset.sustain,
            release=self.preset.release,
            rotation=self.preset.rotation,
            flip_h=self.preset.flip_h,
            flip_v=self.preset.flip_v,
            invert=self.preset.invert,
            height=64,
            freq_scale=self.preset.freq_scale
        )

        try:
            arr, sr = self.engine.generate_to_array(self.processor, tmp_preset)
            if arr.ndim == 2:
                arr_mono = arr.mean(axis=1)
            else:
                arr_mono = arr

            import matplotlib.pyplot as plt
            import io
            from PIL import Image as PILImage

            fig, ax = plt.subplots(figsize=(9, 2.5))
            ax.plot(arr_mono[:sr * 3], color="#0aa")
            ax.set_ylim(-1.0, 1.0)
            ax.axis("off")
            buf = io.BytesIO()
            fig.savefig(buf, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            buf.seek(0)
            pil_im = PILImage.open(buf)

            win = tk.Toplevel(self)
            win.title("Waveform Preview")
            imgtk = ImageTk.PhotoImage(pil_im)
            lbl = tk.Label(win, image=imgtk)
            lbl.image = imgtk
            lbl.pack()
        except Exception as e:
            messagebox.showerror("Waveform error", str(e))

    # ------------------------------------------------------------
    # SPECTROGRAM PREVIEW (STATIC + / OR ANIMATED)
    # ------------------------------------------------------------
    def preview_waterfall(self):
        if not SCIPY_AVAILABLE:
            messagebox.showerror(
                "Missing Dependency",
                "Install scipy to enable spectrogram preview:\n\npip install scipy"
            )
            return
        if not self.processor:
            messagebox.showwarning("No image", "Open an image first.")
            return
        if not self.spectrogram_panel:
            self._log("No spectrogram panel wired.")
            return

        self._log("Generating spectrogram preview (in-memory)...")
        self._sync_ui_to_preset()
        try:
            arr, sr = self.engine.generate_to_array(self.processor, self.preset)

            # build mono for spectrogram
            if arr.ndim == 2:
                arr_mono = arr.mean(axis=1)
            else:
                arr_mono = arr

            f, t_vec, Sxx = spectrogram(arr_mono, sr, nperseg=1024, noverlap=800)
            Sdb = 10 * np.log10(Sxx + 1e-9)

            self.spectrogram_panel.reset()
            if self.spec_static_var.get():
                self.spectrogram_panel.show_full(Sdb, f, t_vec, cmap="magma")
            if self.spec_waterfall_var.get():
                self.spectrogram_panel.draw_progressive(Sdb, f, t_vec, interval_ms=60, cmap="magma")

            self._log("Spectrogram preview updated.")
        except Exception as e:
            messagebox.showerror("Waterfall error", str(e))

    # ------------------------------------------------------------
    # PLAYBACK (via system default player)
    # ------------------------------------------------------------
    def play_last(self):
        if not self.last_generated:
            messagebox.showwarning("No file", "Generate audio first.")
            return

        path = self.last_generated
        self._log(f"Opening in system player: {path}")

        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", path])
            elif os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Playback error", f"Could not open file:\n{e}")
