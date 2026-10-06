import numpy as np
from scipy.signal import find_peaks

def main():
    # 1. Load the filtered 3D poses
    data = np.load('examples/ActualTest/Height5/X3Dfiltered.npy')  # Ensure this file exists

    # 2. Define right-knee joint index (Halpe-26 ordering)
    knee_index = 10  # adjust if needed

    # 3. Extract vertical (Y) coordinate for the right knee
    knee_y = data[:, knee_index, 1]

    # 4. Detect top-dead-center peaks (1 rev/s ~ 60 frames)
    fs = 60.0
    min_dist = int(fs * 0.9)
    peaks, _ = find_peaks(knee_y, distance=min_dist)

    # 5. Compute mid-downstroke indices (midpoint between successive peaks)
    if len(peaks) < 2:
        raise RuntimeError("Not enough peaks detected for cycle segmentation.")
    mid_idxs = ((peaks[:-1] + peaks[1:]) // 2).astype(int)

    # 6. Extract knee heights at mid-downstroke for each cycle
    mid_positions = knee_y[mid_idxs]

    # 7. Save mid-downstroke positions
    np.save('mid_positions.npy', mid_positions)
    print(f"Saved mid_positions.npy with {len(mid_positions)} cycle values.")

    # 8. Compute mean inter-cycle variability (in mm)
    diffs = np.abs(np.diff(mid_positions))
    mean_diff_mm = np.mean(diffs) * 1000
    print(f"Mean inter-cycle knee height change: {mean_diff_mm:.1f} mm")

if __name__ == "__main__":
    main()
