# gui/spectrogram_panel.py
import customtkinter as ctk
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import matplotlib.cm as cm


class SpectrogramPanel(ctk.CTkFrame):
    """
    Embedded spectrogram display.

    Supports:
    - show_full(Sdb, f, t, cmap="magma") -> static spectrogram
    - draw_progressive(Sdb, f, t, interval_ms=60, cmap="magma") -> animated waterfall
    - reset() -> clear drawing / cancel animation
    """

    def __init__(self, master, width=560, height=260):
        super().__init__(master, width=width, height=height)
        self.pack_propagate(False)

        self._base_width = width
        self._base_height = height

        self.canvas = tk.Canvas(self, bg="#000", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._full_img: Image.Image | None = None
        self._tk_img: ImageTk.PhotoImage | None = None
        self._anim_after_id: str | None = None
        self._anim_progress_x: int = 0

        self.canvas.bind("<Configure>", self._on_resize)

    # --------------------------
    # Public API
    # --------------------------
    def reset(self):
        """Clear canvas and cancel any running animation."""
        if self._anim_after_id is not None:
            try:
                self.after_cancel(self._anim_after_id)
            except Exception:
                pass
            self._anim_after_id = None

        self.canvas.delete("all")
        self._full_img = None
        self._tk_img = None
        self._anim_progress_x = 0

    def show_full(self, Sdb: np.ndarray, freqs: np.ndarray, times: np.ndarray,
                  cmap: str = "magma"):
        """
        Render the full spectrogram (static image).
        """
        self.reset()
        self._full_img = self._sdb_to_image(Sdb, cmap)
        self._render_image(self._full_img)

    def draw_progressive(self, Sdb: np.ndarray, freqs: np.ndarray, times: np.ndarray,
                         interval_ms: int = 60, cmap: str = "magma"):
        """
        Animate the spectrogram growing from left to right (waterfall-style).
        """
        self.reset()
        base_img = self._sdb_to_image(Sdb, cmap)
        self._full_img = base_img

        w, h = base_img.size
        if w <= 1:
            self._render_image(base_img)
            return

        self._anim_progress_x = 0

        step_cols = max(1, w // 80)  # ~80 frames for a full animation

        def step():
            if self._full_img is None:
                return

            nonlocal w, h, step_cols

            if self._anim_progress_x >= w:
                # Ensure final full render
                self._render_image(self._full_img)
                self._anim_after_id = None
                return

            self._anim_progress_x = min(w, self._anim_progress_x + step_cols)
            cropped = self._full_img.crop((0, 0, self._anim_progress_x, h))
            self._render_image(cropped, stretch_to_canvas=True)
            self._anim_after_id = self.after(interval_ms, step)

        step()

    # --------------------------
    # Internal helpers
    # --------------------------
    def _on_resize(self, event):
        # If we have a full image, re-render it at new size
        if self._full_img is not None and self._anim_after_id is None:
            self._render_image(self._full_img)

    def _sdb_to_image(self, Sdb: np.ndarray, cmap: str) -> Image.Image:
        """
        Convert dB spectrogram -> RGB image (PIL).
        """
        # normalize Sdb
        vmin = float(np.percentile(Sdb, 5))
        vmax = float(np.percentile(Sdb, 99))
        if vmax <= vmin:
            vmax = vmin + 1.0

        S_norm = np.clip((Sdb - vmin) / (vmax - vmin), 0.0, 1.0)

        colormap = cm.get_cmap(cmap)
        rgba = colormap(S_norm)  # (F, T, 4)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)

        # Flip vertically so low freqs at bottom
        rgb = np.flipud(rgb)

        img = Image.fromarray(rgb)
        return img

    def _render_image(self, img: Image.Image, stretch_to_canvas: bool = True):
        """
        Draw an image on the canvas, scaling to the canvas size if requested.
        """
        cw = self.canvas.winfo_width() or self._base_width
        ch = self.canvas.winfo_height() or self._base_height
        if cw < 10 or ch < 10:
            return

        if stretch_to_canvas:
            img_disp = img.resize((cw, ch), Image.BILINEAR)
        else:
            img_disp = img

        self._tk_img = ImageTk.PhotoImage(img_disp)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_img)
