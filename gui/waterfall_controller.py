# gui/waterfall_controller.py
import numpy as np


class WaterfallController:
    """Routes live audio blocks to SpectrogramPanel."""

    def __init__(self, spectrogram_panel):
        self.spec = spectrogram_panel
        self.enabled = False
        self.sample_rate = None

    def start(self, sr):
        """Begin a new real-time spectrogram session."""
        self.sample_rate = sr
        self.enabled = True
        self.spec.reset()

    def stop(self):
        """Stops waterfall mode."""
        self.enabled = False
        self.spec.reset()

    def push(self, audio_block):
        """Receive audio blocks from the player."""
        if not self.enabled:
            return
        if self.sample_rate is None:
            return
        self.spec.push_audio_block(audio_block, self.sample_rate)
