#!/usr/bin/env python3
"""
compute_xcorr.py

1. Load 3D keypoints (n_frames, n_kpts, 3)
2. Extract left/right knee heights (Y positive down)
3. Auto-detect cycles via BDC peaks (right knee)
4. Time-normalize each cycle to avg length
5. Compute normalized cross-correlation per cycle & overall
6. Plot and save results
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os, sys

# === USER CONFIGURATION ===
FILEPATH        = os.path.join('examples', 'ActualTest', 'Height5', 'X3Dfiltered.npy')
RIGHT_KNEE_IDX  = 2      # index of right knee keypoint
LEFT_KNEE_IDX   = 5      # index of left  knee keypoint
PROMINENCE_FRAC = 0.2    # for BDC peak detection
# ==========================

def load_knee_heights(fp):
    data = np.load(fp)  # expect shape (n_frames, n_kpts, 3)
    if data.ndim != 3 or data.shape[2] < 2:
        raise ValueError(f"Unexpected data shape {data.shape}")
    y = data[:, :, 1]  # Y coordinate (positive down)
    return y[:, LEFT_KNEE_IDX], y[:, RIGHT_KNEE_IDX]

def detect_cycles(signal, prom_frac):
    rng = signal.max() - signal.min()
    peaks, _ = find_peaks(signal, prominence=rng*prom_frac)
    if len(peaks) < 2:
        raise RuntimeError("Not enough peaks for cycle segmentation")
    return peaks

def segment_and_normalize(left, right, peaks):
    diffs    = np.diff(peaks)
    n_cycles = len(diffs)
    avg_len  = int(round(np.mean(diffs)))
    L = np.zeros((n_cycles, avg_len))
    R = np.zeros((n_cycles, avg_len))
    for i in range(n_cycles):
        s, e = peaks[i], peaks[i+1]
        seg_len = e - s
        l_seg = left[s:e]; r_seg = right[s:e]
        old_x = np.linspace(0, seg_len-1, seg_len)
        new_x = np.linspace(0, seg_len-1, avg_len)
        L[i] = np.interp(new_x, old_x, l_seg)
        R[i] = np.interp(new_x, old_x, r_seg)
    return L, R

def normalized_xcorr(x, y):
    # full cross-correlation
    corr = np.correlate(x - x.mean(), y - y.mean(), mode='full')
    # normalize by product of norms
    norm = np.linalg.norm(x - x.mean()) * np.linalg.norm(y - y.mean())
    return corr / norm

def main():
    # Load and extract
    try:
        left_h, right_h = load_knee_heights(FILEPATH)
    except Exception as e:
        print("Loading error:", e); sys.exit(1)

    # Detect cycles
    peaks = detect_cycles(right_h, PROMINENCE_FRAC)
    n_cycles = len(peaks)-1

    # Segment & normalize timebase
    Lc, Rc = segment_and_normalize(left_h, right_h, peaks)

    # Compute per-cycle and overall xcorr
    xcorr_cycles = []
    max_coeffs   = []
    max_lags     = []
    for L, R in zip(Lc, Rc):
        xc = normalized_xcorr(L, R)                                     # normalized cross-correlation :contentReference[oaicite:8]{index=8}
        lags = np.arange(-len(L)+1, len(L))
        idx  = np.argmax(xc)
        max_coeffs.append(xc[idx])                                       # peak coefficient :contentReference[oaicite:9]{index=9}
        max_lags.append(lags[idx])
        xcorr_cycles.append(xc)
    xcorr_cycles = np.array(xcorr_cycles)
    # overall
    xc_all = normalized_xcorr(left_h, right_h)
    overall_coeff = np.max(xc_all)

    # Save CSVs
    np.savetxt('examples/ActualTest/Height5/xcorr_per_cycle.csv', np.column_stack([max_coeffs, max_lags]), 
               delimiter=',', header='coeff,lag', comments='')
    np.savetxt('examples/ActualTest/Height5/xcorr_overall.csv', [overall_coeff], delimiter=',')
    
    # Scatter plot
    plt.figure()
    plt.scatter(left_h, right_h, alpha=0.1, s=5)
    mn, mx = min(left_h.min(), right_h.min()), max(left_h.max(), right_h.max())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('Left Knee Height (m)'); plt.ylabel('Right Knee Height (m)')
    plt.title(f'Scatter for Height 5 (Overall coefficient={overall_coeff:.3f})')
    plt.savefig('examples/ActualTest/Height5/xcorr_scatter.png', dpi=300)

    # Bar chart of max coeffs
    plt.figure()
    plt.bar(np.arange(1, n_cycles+1), max_coeffs)
    plt.ylim(-1,1); plt.xlabel('Cycle #'); plt.ylabel('Max cross-correlation coefficient')
    plt.title('Peak Cross-Correlation per Cycle for Height 5')
    plt.savefig('examples/ActualTest/Height5/xcorr_per_cycle.png', dpi=300)

    # Overlay xcorr waveforms
    plt.figure()
    for xc in xcorr_cycles:
        plt.plot(lags, xc, alpha=0.3)
    plt.plot(lags, xcorr_cycles.mean(axis=0), 'k-', linewidth=2)
    plt.xlabel('Lag (frames)'); plt.ylabel('Normalized cross-correlation coefficient')
    plt.title('Cross-Correlation Functions for Height 5')
    plt.savefig('examples/ActualTest/Height5/xcorr_waveforms.png', dpi=300)
    
    
    mean_coeff = np.mean(max_coeffs)
    min_coeff  = np.min(max_coeffs)
    max_coeff  = np.max(max_coeffs)
    print(f"Mean peak coefficient:       {mean_coeff:.3f}")
    print(f"Peak coefficient range:      {min_coeff:.3f}  to  {max_coeff:.3f}")
    
    



if __name__ == '__main__':
    main()
