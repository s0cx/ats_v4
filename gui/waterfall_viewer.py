# gui/waterfall_viewer.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class WaterfallViewer:
    """
    Live-scrolling waterfall view that visualizes the spectrogram
    derived from the processed image rows over time.
    """
    def __init__(self, img_gray: np.ndarray):
        """
        img_gray: 2D grayscale numpy array (height × width)
        """
        self.img = img_gray
        self.h, self.w = img_gray.shape

        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.ax.set_title("Spectrogram Waterfall")
        self.ax.set_xlabel("Frequency bins")
        self.ax.set_ylabel("Time →")
        self.ax.set_ylim(0, self.h)
        self.ax.set_xlim(0, self.w)
        self.ax.invert_yaxis()  # scroll downward

        # Empty initial image (2D)
        self.waterfall_data = np.zeros((self.h, self.w))
        self.im = self.ax.imshow(
            self.waterfall_data,
            aspect="auto",
            cmap="magma",
            interpolation="nearest",
            vmin=0,
            vmax=255,
        )

        self.row_idx = 0

    def _update(self, frame):
        """
        Insert one new image row per frame.
        """
        if self.row_idx >= self.h:
            return self.im

        # shift up (scroll)
        self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)

        # insert new row at bottom
        self.waterfall_data[-1, :] = self.img[self.row_idx, :]

        self.row_idx += 1

        self.im.set_data(self.waterfall_data)
        return self.im

    def show(self):
        """
        Launch scrolling animation.
        """
        self.anim = FuncAnimation(
            self.fig, self._update, interval=30, blit=False
        )
        plt.show()
