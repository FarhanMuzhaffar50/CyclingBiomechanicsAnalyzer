import numpy as np
from scipy.signal import butter, filtfilt

def filter_trajectory(pose, fps=60.0, cutoff_hz=6.0, order=2):
    """Zero-phase Butterworth low-pass over time, preserving (frames,joints,3)."""
    data = np.asarray(pose, dtype=float)
    if data.ndim != 3 or data.shape[2] != 3 or not np.isfinite(data).all():
        raise ValueError("Expected finite pose with shape (frames, joints, 3)")
    if fps <= 0 or not 0 < cutoff_hz < fps / 2 or order < 1:
        raise ValueError("Require fps>0, 0<cutoff<Nyquist, order>=1")
    b, a = butter(order, cutoff_hz / (fps / 2), btype="low")
    padlen = 3 * max(len(a), len(b))
    if len(data) <= padlen:
        raise ValueError(f"At least {padlen + 1} frames are required for filtering")
    return filtfilt(b, a, data, axis=0)
