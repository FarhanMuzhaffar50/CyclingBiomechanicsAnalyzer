#!/usr/bin/env python3
"""
compute_nsi.py

1. Load 3D keypoints (n_frames, n_kpts, 3)
2. Extract left/right knee heights (Y positive down)
3. Auto-detect cycle boundaries via right-knee BDC peaks
4. Time-normalize each cycle
5. Compute NSI = (L − R)/(max − min)*100
6. Save NSI to CSV and plot waveforms
"""

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import sys

# === USER PARAMETERS ===
FILEPATH        = os.path.join('examples', 'ActualTest',
                               'Height5', 'X3Dfiltered.npy')
RIGHT_KNEE_IDX  = 2    # right knee keypoint index
LEFT_KNEE_IDX   = 5    # left  knee keypoint index
PROMINENCE_FRAC = 0.2  # peak prominence as fraction of full range
# ========================

def load_knee_heights(fp):
    data = np.load(fp)
    if data.ndim != 3 or data.shape[2] < 2:
        raise ValueError(f"Bad shape {data.shape}; need (n_frames,n_kpts,3).")
    y = data[:, :, 1]  # Y coord (positive down)
    return y[:, LEFT_KNEE_IDX], y[:, RIGHT_KNEE_IDX]

def detect_bdc_peaks(signal, prom_frac):
    rng = signal.max() - signal.min()
    prom = rng * prom_frac
    peaks, _ = find_peaks(signal, prominence=prom)
    if len(peaks) < 2:
        raise RuntimeError("Too few peaks found for cycle detection.")
    return peaks

def segment_and_normalize(left, right, peaks):
    diffs = np.diff(peaks)
    n_cycles = len(diffs)
    avg_len  = int(np.round(np.mean(diffs)))
    print(f"Detected {n_cycles} cycles; normalizing each to {avg_len} frames.")
    L = np.zeros((n_cycles, avg_len))
    R = np.zeros((n_cycles, avg_len))
    for i in range(n_cycles):
        s, e = peaks[i], peaks[i+1]
        l_seg = left[s:e]
        r_seg = right[s:e]
        old_x = np.linspace(0, len(l_seg)-1, len(l_seg))
        new_x = np.linspace(0, len(l_seg)-1, avg_len)
        L[i] = np.interp(new_x, old_x, l_seg)
        R[i] = np.interp(new_x, old_x, r_seg)
    return L, R

def compute_nsi(L, R):
    allv = np.hstack([L.ravel(), R.ravel()])
    Xmax, Xmin = allv.max(), allv.min()
    if Xmin > 0:
        Xmin = 0.0
    denom = Xmax - Xmin
    if denom == 0:
        raise RuntimeError("Zero range in data.")
    nsi = (L - R) / denom * 100.0
    return nsi, nsi.mean()

def main():
    # 1. load
    try:
        left_h, right_h = load_knee_heights(FILEPATH)
    except Exception as e:
        print("Error loading keypoints:", e)
        sys.exit(1)
    print(f"Loaded {left_h.size} frames of knee height.")

    # 2. detect cycle peaks
    peaks = detect_bdc_peaks(right_h, PROMINENCE_FRAC)
    print(f"Found {len(peaks)} BDC peaks ➔ {len(peaks)-1} cycles.")

    # 3. segment & normalize
    Lc, Rc = segment_and_normalize(left_h, right_h, peaks)

    # 4. compute NSI
    nsi_vals, mean_nsi = compute_nsi(Lc, Rc)
    print(f"Mean NSI: {mean_nsi:.2f}%")

    # 5. save to CSV
    csv_file = 'examples/ActualTest/Height1/nsi_values.csv'
    header = ','.join(f'pt{i}' for i in range(nsi_vals.shape[1]))
    np.savetxt(csv_file, nsi_vals, delimiter=',',
               header=header, comments='')
    print(f"Saved NSI values to '{csv_file}' (rows=cycles, cols=cycle-points)")

    # 6. plotting
    x_pct = np.linspace(0, 100, nsi_vals.shape[1])
    plt.figure()  # plot individual cycles
    for cycle in nsi_vals:
        plt.plot(x_pct, cycle, alpha=0.3)
    plt.plot(x_pct, nsi_vals.mean(axis=0), linewidth=2)
    plt.xlabel('Cycle (%)')
    plt.ylabel('NSI (%)')
    plt.title('NSI Waveforms for Height 5 (thin = individual, thick = mean)')
    plt.grid(True)
    out_png = 'examples/ActualTest/Height5/nsi_waveform.png'
    plt.savefig(out_png, dpi=300)
    plt.show()
    print(f"Saved NSI plot to '{out_png}'")

if __name__ == '__main__':
    main()