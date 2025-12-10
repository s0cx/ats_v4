# image/processor.py
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from typing import Tuple, Optional

class ImageProcessor:
    """Load an image and provide transformed arrays at requested vertical resolution."""

    def __init__(self, path: str):
        self.path = path
        self._img = Image.open(path).convert("RGBA")

    def transformed(self,
                    rotation: int = 0,
                    flip_h: bool = False,
                    flip_v: bool = False,
                    invert: bool = False,
                    brightness: float = 1.0,
                    contrast: float = 1.0):
        img = self._img
        if rotation % 360 != 0:
            img = img.rotate(rotation, expand=True)
        if flip_h:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if flip_v:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if invert:
            r, g, b, a = img.split()
            rgb = Image.merge("RGB", (r, g, b))
            rgb = ImageOps.invert(rgb)
            img = Image.merge("RGBA", (*rgb.split(), a))
        if brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(brightness)
        if contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contrast)
        return img

    def to_grayscale_array(self, img: Image.Image, height: int):
        """Return normalized grayscale array shape (height, width) range 0..1 (float32)."""
        img_l = img.convert("L")
        w = max(1, img_l.width)
        img_small = img_l.resize((w, height), resample=Image.BILINEAR)
        arr = np.array(img_small, dtype=np.float32) / 255.0
        return arr

    def to_rgb_arrays(self, img: Image.Image, height: int):
        """Return R,G,B arrays (height,width) normalized 0..1."""
        img_rgb = img.convert("RGB")
        w = max(1, img_rgb.width)
        r, g, b = img_rgb.split()
        r_s = r.resize((w, height), resample=Image.BILINEAR)
        g_s = g.resize((w, height), resample=Image.BILINEAR)
        b_s = b.resize((w, height), resample=Image.BILINEAR)
        return (np.array(r_s, dtype=np.float32) / 255.0,
                np.array(g_s, dtype=np.float32) / 255.0,
                np.array(b_s, dtype=np.float32) / 255.0)
