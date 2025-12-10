# main.py
import sys
import customtkinter as ctk

from gui.simple_window import SimpleSpectrogramWindow
from gui.main_window import ATSMainWindow


def _ask_mode():
    """Show a small chooser that lets the user pick a mode on launch."""

    mode = {"value": None}

    chooser = ctk.CTk()
    chooser.title("ATS Studio — Choose your view")
    chooser.geometry("440x260")
    chooser.resizable(False, False)

    chooser.grid_columnconfigure(0, weight=1)
    chooser.grid_rowconfigure((1, 2), weight=1)

    title = ctk.CTkLabel(
        chooser,
        text="How would you like to open ATS Studio?",
        font=("Segoe UI", 18, "bold"),
    )
    title.grid(row=0, column=0, sticky="ew", padx=20, pady=(24, 8))

    simple_btn = ctk.CTkButton(
        chooser,
        text="Simple view (upload → spectrogram)",
        height=56,
        font=("Segoe UI", 15, "bold"),
        command=lambda: _select("simple", mode, chooser),
    )
    simple_btn.grid(row=1, column=0, sticky="ew", padx=20, pady=8)

    classic_btn = ctk.CTkButton(
        chooser,
        text="Full studio (advanced controls)",
        height=56,
        font=("Segoe UI", 15, "bold"),
        fg_color="#2c2f3a",
        hover_color="#3a3e4c",
        command=lambda: _select("classic", mode, chooser),
    )
    classic_btn.grid(row=2, column=0, sticky="ew", padx=20, pady=8)

    chooser.bind("<Escape>", lambda _event: chooser.destroy())
    chooser.mainloop()

    return mode["value"]


def _select(value: str, mode_store: dict, chooser: ctk.CTk):
    mode_store["value"] = value
    chooser.destroy()


def main():
    # Global appearance
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    app = ATSMainWindow()
    if "--classic" in sys.argv:
        chosen = "classic"
    elif "--simple" in sys.argv:
        chosen = "simple"
    else:
        chosen = _ask_mode()

    if chosen == "classic":
        app = ATSMainWindow()
    else:
        app = SimpleSpectrogramWindow()

    app.mainloop()


if __name__ == "__main__":
    main()
