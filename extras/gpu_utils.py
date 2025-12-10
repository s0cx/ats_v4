# extras/gpu_utils.py
"""
GPU helper: optional CuPy support.
If CuPy is not installed, fall back to NumPy.

Note: Installing CuPy depends on your CUDA version. Example:
  pip install cupy-cuda11x
Replace 11x with your installed CUDA (e.g. cupy-cuda11-7 -> cupy-cuda117)
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp
except Exception:
    import numpy as np
    GPU_AVAILABLE = False
    xp = np

def array(x, dtype=None):
    return xp.array(x, dtype=dtype)

def to_cpu(x):
    if GPU_AVAILABLE:
        return xp.asnumpy(x)
    return x
