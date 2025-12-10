# gui/simple_window.py
import customtkinter as ctk
from tkinter import filedialog, messagebox
from typing import Optional

import numpy as np

from core.presets import SettingsPreset
from image.processor import ImageProcessor
from audio.engine import AudioEngine
from gui.preview_panel import PreviewPanel
from gui.spectrogram_panel import SpectrogramPanel

try:
    from scipy.signal import spectrogram
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


class SimpleSpectrogramWindow(ctk.CTk):
    """Minimal flow: upload image -> view static spectrogram."""

    def __init__(self):
        super().__init__()

        self.title("ATS Studio — Simple Spectrogram")
        self.geometry("1040x720")
        self.minsize(920, 640)

        self.processor: Optional[ImageProcessor] = None
        self.preset = SettingsPreset(height=128, volume=0.16, duration=0)
        self.engine = AudioEngine(progress_callback=self._on_progress)

        self._build_layout()

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------
    def _build_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="#0f1116")
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="Image → Spectrogram",
            font=("Segoe UI", 22, "bold"),
        )
        subtitle = ctk.CTkLabel(
            header,
            text="One button: drop an image, get a spectrogram. Choose Full Studio on launch for all controls.",
            font=("Segoe UI", 13),
        )
        title.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 0))
        subtitle.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 12))

        body = ctk.CTkFrame(self)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=1)

        upload_btn = ctk.CTkButton(
            body,
            text="Upload image → Make spectrogram",
            command=self._handle_upload,
            height=56,
            font=("Segoe UI", 18, "bold"),
        )
        upload_btn.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        panes = ctk.CTkFrame(body)
        panes.grid(row=1, column=0, sticky="nsew")
        panes.grid_columnconfigure(0, weight=1, uniform="pane")
        panes.grid_columnconfigure(1, weight=1, uniform="pane")
        panes.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(panes)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_rowconfigure(1, weight=1)

        right = ctk.CTkFrame(panes)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(left, text="Your image", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, sticky="w", padx=6, pady=(6, 4)
        )
        self.preview_panel = PreviewPanel(left, width=480, height=400)
        self.preview_panel.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)

        ctk.CTkLabel(right, text="Spectrogram", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, sticky="w", padx=6, pady=(6, 4)
        )
        self.spectrogram_panel = SpectrogramPanel(right, width=480, height=400)
        self.spectrogram_panel.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)

        footer = ctk.CTkFrame(body)
        footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        footer.grid_columnconfigure(1, weight=1)

        self.progress = ctk.CTkProgressBar(footer)
        self.progress.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self.progress.set(0.0)

        self.status_label = ctk.CTkLabel(footer, text="Waiting for an image...")
        self.status_label.grid(row=0, column=1, sticky="w", padx=8)

    # ------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------
    def _handle_upload(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")],
        )
        if not path:
            return

        self._set_status("Loading image...", 0.05)
        try:
            self.processor = ImageProcessor(path)
            self.preview_panel.show_image(self.processor.transformed())
        except Exception as exc:
            messagebox.showerror("Load error", f"Could not read the image:\n{exc}")
            self._set_status("Waiting for an image...", 0.0)
            return

        self.after(50, self._make_spectrogram)

    def _make_spectrogram(self):
        if not SCIPY_AVAILABLE:
            messagebox.showerror(
                "Missing dependency",
                "Install scipy to build the spectrogram preview:\n\n    pip install scipy",
            )
            self._set_status("Install scipy to preview spectrograms", 0.0)
            return

        if not self.processor:
            return

        self._set_status("Turning pixels into sound...", 0.12)
        try:
            audio_arr, sr = self.engine.generate_to_array(self.processor, self.preset)
            audio_mono = audio_arr.mean(axis=1) if audio_arr.ndim == 2 else audio_arr

            f, t_vec, sxx = spectrogram(audio_mono, sr, nperseg=1024, noverlap=800)
            sdb = 10 * np.log10(sxx + 1e-9)

            self.spectrogram_panel.reset()
            self.spectrogram_panel.show_full(sdb, f, t_vec, cmap="magma")
            self._set_status("Spectrogram ready", 1.0)
        except Exception as exc:
            messagebox.showerror("Spectrogram error", str(exc))
            self._set_status("Something went wrong", 0.0)

    # ------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------
    def _on_progress(self, pct: float, status: str):
        pct_safe = max(0.0, min(100.0, pct)) / 100.0
        self.progress.set(pct_safe)
        self.status_label.configure(text=status)

    def _set_status(self, text: str, pct: float):
        self.progress.set(max(0.0, min(1.0, pct)))
        self.status_label.configure(text=text)