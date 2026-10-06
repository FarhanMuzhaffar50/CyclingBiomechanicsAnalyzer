import numpy as np
from scipy.signal import find_peaks

def detect_cycle_peaks(signal, fps, prominence_fraction=0.2, min_cycle_seconds=0.45):
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or not np.isfinite(signal).all() or fps <= 0:
        raise ValueError("Expected a finite 1D signal and positive FPS")
    span = np.ptp(signal)
    if span <= 0:
        raise ValueError("Cannot detect cycles in a constant signal")
    peaks, _ = find_peaks(signal, prominence=span * prominence_fraction,
                          distance=max(1, round(fps * min_cycle_seconds)))
    if len(peaks) < 2:
        raise ValueError("Fewer than two pedal-cycle boundary peaks detected")
    return peaks

def normalize_cycles(signal, peaks, points=100):
    signal = np.asarray(signal, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    if points < 2 or len(peaks) < 2 or np.any(np.diff(peaks) < 2):
        raise ValueError("Need at least one cycle of two frames and two output points")
    return np.array([np.interp(np.linspace(0, 1, points),
                              np.linspace(0, 1, b-a+1), signal[a:b+1])
                     for a, b in zip(peaks[:-1], peaks[1:])])
