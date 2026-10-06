import numpy as np

def derivatives(position, fps):
    p = np.asarray(position, dtype=float)
    dt = 1.0 / fps
    v = np.gradient(p, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)
    return v, a, j

def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))

def cycle_consistency(normalized_cycles):
    """RMS deviation from mean trajectory in the input coordinate units."""
    x = np.asarray(normalized_cycles, dtype=float)
    return rms(x - x.mean(axis=0))
