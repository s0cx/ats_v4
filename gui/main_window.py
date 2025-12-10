# gui/main_window.py
import customtkinter as ctk

from gui.preview_panel import PreviewPanel
from gui.spectrogram_panel import SpectrogramPanel
from gui.controls_panel import ControlsPanel


class ATSMainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ATS Studio v4 — Image → Spectrogram → Audio")
        self.geometry("1280x800")
        self.minsize(1100, 700)

        self._build_layout()

    def _build_layout(self):
        # Grid for main window: header row + content row
        self.grid_columnconfigure(0, weight=3, uniform="main")
        self.grid_columnconfigure(1, weight=2, uniform="main")
        self.grid_rowconfigure(0, weight=0)  # header
        self.grid_rowconfigure(1, weight=1)  # content

        # HEADER
        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 0))
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(header, text="ATS Studio v4",
                             font=("Segoe UI", 20, "bold"))
        subtitle = ctk.CTkLabel(header, text="Austin's Image → Spectrogram → Audio Lab",
                                font=("Segoe UI", 12))
        title.grid(row=0, column=0, sticky="w")
        subtitle.grid(row=1, column=0, sticky="w")

        # LEFT SIDE: Preview + Spectrogram stacked
        left = ctk.CTkFrame(self)
        left.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)

        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=3)  # preview
        left.grid_rowconfigure(1, weight=2)  # spectrogram

        self.preview_panel = PreviewPanel(left)
        self.preview_panel.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4, 2))

        self.spectrogram_panel = SpectrogramPanel(left)
        self.spectrogram_panel.grid(row=1, column=0, sticky="nsew", padx=4, pady=(2, 4))

        # RIGHT SIDE: Scrollable Controls
        self.controls = ControlsPanel(
            self,
            width=420,
            height=700,
            preview_panel=self.preview_panel,
            spectrogram_panel=self.spectrogram_panel,
        )
        self.controls.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=8)

        # Let right column scroll frame grow
        self.grid_columnconfigure(1, weight=2)
