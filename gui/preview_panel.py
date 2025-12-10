# gui/preview_panel.py
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk


class PreviewPanel(ctk.CTkFrame):
    """
    Image preview panel.

    - Shows a single transformed image
    - Resizes automatically with window
    """

    def __init__(self, master, width=560, height=360):
        super().__init__(master, width=width, height=height)
        self.pack_propagate(False)

        self._base_width = width
        self._base_height = height

        # Use a standard tkinter Canvas inside CTkFrame
        self.canvas = tk.Canvas(self, bg="#111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.current_img: Image.Image | None = None
        self._tk_img: ImageTk.PhotoImage | None = None

        # Re-render on resize
        self.canvas.bind("<Configure>", self._on_resize)

    # --------------------------
    # Public API
    # --------------------------
    def show_image(self, img: Image.Image):
        """Store the current image and render it."""
        self.current_img = img
        self._render_current()

    # --------------------------
    # Internal
    # --------------------------
    def _on_resize(self, event):
        if self.current_img is not None:
            self._render_current()

    def _render_current(self):
        if self.current_img is None:
            return

        w = self.canvas.winfo_width() or self._base_width
        h = self.canvas.winfo_height() or self._base_height
        if w < 10 or h < 10:
            return

        img = self.current_img

        # Preserve aspect ratio
        iw, ih = img.size
        scale = min(w / iw, h / ih)
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))

        resized = img.resize((new_w, new_h), Image.LANCZOS)

        self._tk_img = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(w // 2, h // 2, image=self._tk_img)
