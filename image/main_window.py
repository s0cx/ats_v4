# gui/main_window.py
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk
import threading
import queue
import os
import numpy as np
import io
import matplotlib
matplotlib.use("Agg")  # avoid direct UI backend for rendering images for embedding
import matplotlib.pyplot as plt
from core.presets import SettingsPreset
from image.processor import ImageProcessor
from audio.engine import AudioEngine
from core.utils import nice_duration_from_width
import soundfile as sf  # optional; used for reading previews if available
import simpleaudio as sa  # optional playback

class ATSMainWindow:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("ATS Studio — v4 (CustomTkinter)")
        self.root.geometry("1180x820")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # core state
        self.preset = SettingsPreset()
        self.processor = None
        self.image_path = None
        self.output_path = None
        self.engine = AudioEngine(progress_callback=self._on_progress)
        self.log_queue = queue.Queue()
        self.gen_thread = None
        self.last_generated = None

        # Build UI
        self._build_ui()
        self._process_queue()  # start GUI update loop

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self.root)
        header.place(x=12, y=8, width=1156, height=54)
        title = ctk.CTkLabel(header, text="ATS Studio — v4", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(side="left", padx=10)
        subtitle = ctk.CTkLabel(header, text="Image → Spectrogram → WAV (modular, vectorized)", font=ctk.CTkFont(size=12))
        subtitle.pack(side="left", padx=10)

        # Left pane: preview & transforms
        left = ctk.CTkFrame(self.root)
        left.place(x=12, y=70, width=560, height=740)

        self.preview_canvas = ctk.CTkLabel(left, text="No Image", width=520, height=380, corner_radius=6)
        self.preview_canvas.pack(padx=12, pady=8)

        btn_frame = ctk.CTkFrame(left)
        btn_frame.pack(fill="x", padx=8, pady=6)
        ctk.CTkButton(btn_frame, text="Open Image", command=self.select_image).pack(side="left", padx=6)
        ctk.CTkButton(btn_frame, text="Save Preset", command=self.save_preset).pack(side="left", padx=6)
        ctk.CTkButton(btn_frame, text="Load Preset", command=self.load_preset).pack(side="left", padx=6)
        ctk.CTkButton(btn_frame, text="Export JSON", command=self.export_settings).pack(side="left", padx=6)

        # transform controls
        trans_frame = ctk.CTkFrame(left)
        trans_frame.pack(fill="x", padx=8, pady=6)

        self.rotate_var = tk.IntVar(value=self.preset.rotation)
        ctk.CTkButton(trans_frame, text="⟲", width=44, command=lambda: self._mod_rotation(-90)).grid(row=0, column=0, padx=6, pady=6)
        ctk.CTkButton(trans_frame, text="⟳", width=44, command=lambda: self._mod_rotation(90)).grid(row=0, column=1, padx=6)
        self.flip_h_var = tk.BooleanVar(value=self.preset.flip_h)
        self.flip_v_var = tk.BooleanVar(value=self.preset.flip_v)
        ctk.CTkCheckBox(trans_frame, text="Flip H", variable=self.flip_h_var, command=self.preview_refresh).grid(row=0, column=2, padx=6)
        ctk.CTkCheckBox(trans_frame, text="Flip V", variable=self.flip_v_var, command=self.preview_refresh).grid(row=0, column=3, padx=6)
        self.invert_var = tk.BooleanVar(value=self.preset.invert)
        ctk.CTkCheckBox(trans_frame, text="Invert", variable=self.invert_var, command=self.preview_refresh).grid(row=0, column=4, padx=6)

        # brightness/contrast
        bc_frame = ctk.CTkFrame(left)
        bc_frame.pack(fill="x", padx=8, pady=6)
        self.brightness_var = tk.DoubleVar(value=self.preset.brightness)
        self.contrast_var = tk.DoubleVar(value=self.preset.contrast)
        ctk.CTkLabel(bc_frame, text="Brightness").grid(row=0, column=0, sticky="w", padx=6)
        ctk.CTkSlider(bc_frame, from_=0.2, to=2.5, variable=self.brightness_var, command=lambda v: self.preview_refresh()).grid(row=0, column=1, padx=6, sticky="ew")
        ctk.CTkLabel(bc_frame, text="Contrast").grid(row=1, column=0, sticky="w", padx=6)
        ctk.CTkSlider(bc_frame, from_=0.2, to=2.5, variable=self.contrast_var, command=lambda v: self.preview_refresh()).grid(row=1, column=1, padx=6, sticky="ew")

        # Right pane: audio controls
        right = ctk.CTkFrame(self.root)
        right.place(x=584, y=70, width=584, height=740)

        file_frame = ctk.CTkFrame(right)
        file_frame.pack(fill="x", padx=8, pady=6)
        ctk.CTkButton(file_frame, text="Select Output WAV", command=self.select_output).pack(side="left", padx=6)
        ctk.CTkButton(file_frame, text="Quick Generate (BG)", command=self.start_generation_thread).pack(side="left", padx=6)
        ctk.CTkButton(file_frame, text="Play Last", command=self.play_last).pack(side="left", padx=6)

        # sample rate & duration
        io_frame = ctk.CTkFrame(right)
        io_frame.pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(io_frame, text="Sample rate:").grid(row=0, column=0, padx=6, sticky="w")
        self.sample_rate_var = tk.IntVar(value=self.preset.sample_rate)
        ctk.CTkOptionMenu(io_frame, values=["22050","32000","44100","48000","96000"], variable=self.sample_rate_var).grid(row=0, column=1, padx=6, sticky="w")
        ctk.CTkLabel(io_frame, text="Duration (s, 0=auto):").grid(row=1, column=0, padx=6, sticky="w")
        self.duration_var = tk.IntVar(value=self.preset.duration)
        ctk.CTkEntry(io_frame, textvariable=self.duration_var, width=80).grid(row=1, column=1, padx=6, sticky="w")

        # freq / mixing
        freq_frame = ctk.CTkFrame(right)
        freq_frame.pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(freq_frame, text="Min freq").grid(row=0, column=0, padx=6, sticky="w")
        self.min_freq_var = tk.IntVar(value=int(self.preset.min_freq))
        ctk.CTkEntry(freq_frame, textvariable=self.min_freq_var, width=80).grid(row=0, column=1, padx=6, sticky="w")
        ctk.CTkLabel(freq_frame, text="Max freq").grid(row=1, column=0, padx=6, sticky="w")
        self.max_freq_var = tk.IntVar(value=int(self.preset.max_freq))
        ctk.CTkEntry(freq_frame, textvariable=self.max_freq_var, width=80).grid(row=1, column=1, padx=6, sticky="w")

        mix_frame = ctk.CTkFrame(right)
        mix_frame.pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(mix_frame, text="Volume").grid(row=0, column=0, padx=6, sticky="w")
        self.volume_var = tk.DoubleVar(value=self.preset.volume)
        ctk.CTkEntry(mix_frame, textvariable=self.volume_var, width=80).grid(row=0, column=1, padx=6, sticky="w")
        self.stereo_var = tk.BooleanVar(value=self.preset.stereo)
        ctk.CTkCheckBox(mix_frame, text="Stereo", variable=self.stereo_var).grid(row=0, column=2, padx=6)
        self.rgb_var = tk.BooleanVar(value=self.preset.rgb_mode)
        ctk.CTkCheckBox(mix_frame, text="RGB mode", variable=self.rgb_var, command=self.preview_refresh).grid(row=0, column=3, padx=6)
        self.rgb_to_stereo_var = tk.BooleanVar(value=self.preset.rgb_to_stereo)
        ctk.CTkCheckBox(mix_frame, text="RGB→Stereo", variable=self.rgb_to_stereo_var).grid(row=1, column=0, padx=6)

        # ADSR
        adsr_frame = ctk.CTkFrame(right)
        adsr_frame.pack(fill="x", padx=8, pady=6)
        self.attack_var = tk.DoubleVar(value=self.preset.attack)
        self.decay_var = tk.DoubleVar(value=self.preset.decay)
        self.sustain_var = tk.DoubleVar(value=self.preset.sustain)
        self.release_var = tk.DoubleVar(value=self.preset.release)
        ctk.CTkLabel(adsr_frame, text="Attack").grid(row=0, column=0, padx=6, sticky="w")
        ctk.CTkEntry(adsr_frame, textvariable=self.attack_var, width=80).grid(row=0, column=1, padx=6, sticky="w")
        ctk.CTkLabel(adsr_frame, text="Decay").grid(row=0, column=2, padx=6, sticky="w")
        ctk.CTkEntry(adsr_frame, textvariable=self.decay_var, width=80).grid(row=0, column=3, padx=6, sticky="w")
        ctk.CTkLabel(adsr_frame, text="Sustain").grid(row=1, column=0, padx=6, sticky="w")
        ctk.CTkEntry(adsr_frame, textvariable=self.sustain_var, width=80).grid(row=1, column=1, padx=6, sticky="w")
        ctk.CTkLabel(adsr_frame, text="Release").grid(row=1, column=2, padx=6, sticky="w")
        ctk.CTkEntry(adsr_frame, textvariable=self.release_var, width=80).grid(row=1, column=3, padx=6, sticky="w")

        creative_frame = ctk.CTkFrame(right)
        creative_frame.pack(fill="x", padx=8, pady=6)
        self.phase_var = tk.BooleanVar(value=self.preset.phase_random)
        ctk.CTkCheckBox(creative_frame, text="Phase randomization", variable=self.phase_var).grid(row=0, column=0, padx=6)
        ctk.CTkLabel(creative_frame, text="Frequency scale:").grid(row=0, column=1, padx=6)
        self.freq_scale_var = tk.StringVar(value=self.preset.freq_scale)
        ctk.CTkOptionMenu(creative_frame, values=["log","linear"], variable=self.freq_scale_var, command=lambda v: None).grid(row=0, column=2, padx=6)
        ctk.CTkButton(creative_frame, text="Preview Spectrogram", command=self.preview_spectrogram).grid(row=0, column=3, padx=6)
        ctk.CTkButton(creative_frame, text="Preview Waveform", command=self.preview_waveform).grid(row=0, column=4, padx=6)

        # progress / log
        self.progress = ctk.CTkProgressBar(right, width=540)
        self.progress.set(0.0)
        self.progress.pack(padx=16, pady=6)
        self.status_label = ctk.CTkLabel(right, text="Idle")
        self.status_label.pack(padx=8, pady=4)

        self.logbox = tk.Text(right, height=8, bg="#111", fg="#ddd", state="disabled")
        self.logbox.pack(fill="both", padx=12, pady=6)

    # ---------- UI utilities ----------
    def _log(self, text: str):
        self.logbox.configure(state="normal")
        self.logbox.insert("end", f"{text}\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def _on_progress(self, percent: float, status: str):
        self.log_queue.put((percent, status))

    def _process_queue(self):
        while not self.log_queue.empty():
            pct, status = self.log_queue.get_nowait()
            try:
                self.progress.set(pct / 100.0)
                self.status_label.configure(text=status)
            except Exception:
                pass
            self._log(status)
        self.root.after(150, self._process_queue)

    # ---------- Image / file ops ----------
    def select_image(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if not p:
            return
        self.image_path = p
        self.processor = ImageProcessor(p)
        # reset transforms defaults
        self.rotate_var.set(0)
        self.flip_h_var.set(False)
        self.flip_v_var.set(False)
        self.invert_var.set(False)
        self.preview_refresh()
        self._log(f"Image loaded: {p}")

    def select_output(self):
        p = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav"), ("FLAC", "*.flac")])
        if not p:
            return
        self.output_path = p
        self._log(f"Output: {p}")

    # rotate helpers
    def _mod_rotation(self, delta):
        self.rotate_var.set((self.rotate_var.get() + delta) % 360)
        self.preview_refresh()

    def preview_refresh(self):
        if not self.processor:
            self.preview_canvas.configure(text="No Image", image=None)
            return
        try:
            img = self.processor.transformed(
                rotation=self.rotate_var.get(),
                flip_h=self.flip_h_var.get(),
                flip_v=self.flip_v_var.get(),
                invert=self.invert_var.get(),
                brightness=self.brightness_var.get(),
                contrast=self.contrast_var.get()
            )
            thumb = img.copy()
            thumb.thumbnail((520, 380))
            self._tk_thumb = ImageTk.PhotoImage(thumb.convert("RGBA"))
            self.preview_canvas.configure(image=self._tk_thumb, text="")
        except Exception as e:
            messagebox.showerror("Preview error", str(e))

    def preview_spectrogram(self):
        if not self.processor:
            messagebox.showwarning("No image", "Open an image first.")
            return
        img = self.processor.transformed(
            rotation=self.rotate_var.get(),
            flip_h=self.flip_h_var.get(),
            flip_v=self.flip_v_var.get(),
            invert=self.invert_var.get(),
            brightness=self.brightness_var.get(),
            contrast=self.contrast_var.get()
        )
        arr = np.array(img.convert("L"))
        # create matplotlib figure and render to image, then show in a Toplevel window
        fig, ax = plt.subplots(figsize=(10,3))
        ax.imshow(arr, cmap="gray", aspect="auto", origin="lower")
        ax.set_title("Spectrogram-style Preview (grayscale)")
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        imgtk = ImageTk.PhotoImage(ImageTk.Image.open(buf))
        win = ctk.CTkToplevel(self.root)
        win.geometry("1000x300")
        label = ctk.CTkLabel(win, image=imgtk)
        label.image = imgtk
        label.pack(expand=True, fill="both")

    def preview_waveform(self):
        # produce a small preview by running engine with small height and short duration in main thread (fast)
        if not self.processor:
            messagebox.showwarning("No image", "Open an image first.")
            return
        self._log("Generating quick waveform preview (low res)...")
        temp_preset = SettingsPreset(
            sample_rate=22050,
            duration=3 if self.duration_var.get() == 0 else min(6, max(1, int(self.duration_var.get()))),
            min_freq=self.min_freq_var.get(),
            max_freq=self.max_freq_var.get(),
            volume=float(self.volume_var.get()),
            stereo=self.stereo_var.get(),
            rgb_mode=self.rgb_var.get(),
            rgb_to_stereo=self.rgb_to_stereo_var.get(),
            phase_random=self.phase_var.get(),
            attack=self.attack_var.get(),
            decay=self.decay_var.get(),
            sustain=self.sustain_var.get(),
            release=self.release_var.get(),
            brightness=self.brightness_var.get(),
            contrast=self.contrast_var.get(),
            invert=self.invert_var.get(),
            rotation=self.rotate_var.get(),
            flip_h=self.flip_h_var.get(),
            flip_v=self.flip_v_var.get(),
            height=64,
            freq_scale=self.freq_scale_var.get()
        )
        # use engine to render to memory via soundfile
        try:
            tmp_out = os.path.join(os.path.expanduser("~"), "ats_temp_preview.wav")
            eng = AudioEngine(progress_callback=lambda p, s: None)
            eng.generate_from_image(self.processor, tmp_out, temp_preset)
            # read back
            data, sr = sf.read(tmp_out)
            if data.ndim == 2:
                data = data.mean(axis=1)
            N = len(data)
            step = max(1, N // 900)
            x = np.arange(0, N, step)
            y = data[x]
            # draw waveform
            fig, ax = plt.subplots(figsize=(9,2.5))
            ax.plot(np.linspace(0, len(y)/sr, y.size), y, color="#0aa")
            ax.set_xlabel("s")
            ax.set_ylabel("Amplitude")
            ax.set_title("Waveform Preview")
            ax.set_ylim(-1.0, 1.0)
            ax.axis("tight")
            buf = io.BytesIO()
            fig.savefig(buf, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            buf.seek(0)
            imgtk = ImageTk.PhotoImage(ImageTk.Image.open(buf))
            win = ctk.CTkToplevel(self.root)
            win.geometry("920x260")
            label = ctk.CTkLabel(win, image=imgtk)
            label.image = imgtk
            label.pack(expand=True, fill="both")
            try:
                os.remove(tmp_out)
            except Exception:
                pass
            self._log("Waveform preview done.")
        except Exception as e:
            messagebox.showerror("Preview error", str(e))

    # ---------- Presets ----------
    def save_preset(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not p:
            return
        self._sync_ui_to_preset()
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.preset.to_json())
        self._log(f"Preset saved: {p}")

    def load_preset(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not p:
            return
        import json
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.preset = SettingsPreset.from_dict(data)
            self._apply_preset_to_ui()
            self.preview_refresh()
            self._log(f"Preset loaded: {p}")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def export_settings(self):
        self._sync_ui_to_preset()
        win = ctk.CTkToplevel(self.root)
        win.geometry("640x420")
        text = tk.Text(win, wrap="word")
        text.pack(fill="both", expand=True)
        text.insert("1.0", self.preset.to_json())
        text.configure(state="disabled")

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
        self.preset.brightness = float(self.brightness_var.get())
        self.preset.contrast = float(self.contrast_var.get())
        self.preset.invert = bool(self.invert_var.get())
        self.preset.rotation = int(self.rotate_var.get())
        self.preset.flip_h = bool(self.flip_h_var.get())
        self.preset.flip_v = bool(self.flip_v_var.get())
        self.preset.height = int(getattr(self.preset, "height", 256))
        self.preset.freq_scale = self.freq_scale_var.get()

    def _apply_preset_to_ui(self):
        self.sample_rate_var.set(self.preset.sample_rate)
        self.duration_var.set(self.preset.duration)
        self.min_freq_var.set(int(self.preset.min_freq))
        self.max_freq_var.set(int(self.preset.max_freq))
        self.volume_var.set(self.preset.volume)
        self.stereo_var.set(self.preset.stereo)
        self.rgb_var.set(self.preset.rgb_mode)
        self.rgb_to_stereo_var.set(self.preset.rgb_to_stereo)
        self.phase_var.set(self.preset.phase_random)
        self.attack_var.set(self.preset.attack)
        self.decay_var.set(self.preset.decay)
        self.sustain_var.set(self.preset.sustain)
        self.release_var.set(self.preset.release)
        self.brightness_var.set(self.preset.brightness)
        self.contrast_var.set(self.preset.contrast)
        self.rotate_var.set(self.preset.rotation)
        self.flip_h_var.set(self.preset.flip_h)
        self.flip_v_var.set(self.preset.flip_v)
        self.invert_var.set(self.preset.invert)
        self.freq_scale_var.set(self.preset.freq_scale)

    # ---------- Generation ----------
    def start_generation_thread(self):
        if not self.processor:
            messagebox.showwarning("No image", "Open an image first.")
            return
        if not self.output_path:
            messagebox.showwarning("No output", "Choose output path.")
            return
        if self.gen_thread and self.gen_thread.is_alive():
            messagebox.showinfo("Busy", "Generation already running.")
            return

        self._sync_ui_to_preset()
        out_path = self.output_path

        self.progress.set(0.0)
        self.status_label.configure(text="Starting...")

        def worker():
            try:
                self.engine.reset_stop()
                path = self.engine.generate_from_image(self.processor, out_path, self.preset)
                self.last_generated = path
                self._log(f"Generated: {path}")
            except InterruptedError:
                self._log("Generation cancelled")
            except Exception as e:
                self._log(f"Error: {e}")
                try:
                    messagebox.showerror("Generation error", str(e))
                except Exception:
                    pass

        self.gen_thread = threading.Thread(target=worker, daemon=True)
        self.gen_thread.start()
        self._log("Generation thread started")

    def cancel_generation(self):
        if self.gen_thread and self.gen_thread.is_alive():
            self.engine.stop()
            self._log("Requested cancel")

    # ---------- Playback ----------
    def play_last(self):
        if not self.last_generated:
            messagebox.showwarning("No file", "Generate a WAV first.")
            return
        try:
            data, sr = sf.read(self.last_generated)
            if data.ndim == 1:
                channels = 1
            else:
                channels = data.shape[1]
            inter = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
            play_obj = sa.play_buffer(inter.tobytes(), num_channels=channels, bytes_per_sample=2, sample_rate=sr)
            self._log("Playback started (non-blocking)")
        except Exception as e:
            messagebox.showerror("Playback error", str(e))
