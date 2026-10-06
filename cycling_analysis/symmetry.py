import numpy as np
from scipy.signal import correlate, correlation_lags

def normalized_symmetry_index(left_cycles, right_cycles):
    left, right = np.asarray(left_cycles), np.asarray(right_cycles)
    all_values = np.concatenate((left.ravel(), right.ravel()))
    denominator = all_values.max() - min(0.0, all_values.min())
    if denominator <= 0:
        raise ValueError("Zero knee trajectory range")
    return 100 * (left - right) / denominator

def cross_correlation(left, right):
    """Positive lag means the left signal occurs later than the right signal."""
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Signals must be equal-length 1D arrays")
    x, y = x-x.mean(), y-y.mean()
    denominator = np.linalg.norm(x)*np.linalg.norm(y)
    if denominator == 0:
        return {"coefficient": None, "lag_frames": None}
    corr = correlate(x, y, mode="full") / denominator
    lags = correlation_lags(len(x), len(y), mode="full")
    idx = int(np.argmax(corr))
    return {"coefficient": float(corr[idx]), "lag_frames": int(lags[idx])}
