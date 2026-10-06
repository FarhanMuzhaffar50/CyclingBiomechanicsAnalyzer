import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def main():
    # 1. Load the filtered 3D pose data
    # Make sure 'X3Dfiltered.npy' is in the same directory
    data = np.load('examples/ActualTest/Height5/X3Dfiltered.npy')  # shape: (n_frames, n_joints, 3)

    # 2. Select the joint index for the right knee (Halpe-26 ordering)
    knee_index = 2  # adjust if your ordering differs

    # 3. Extract the vertical (Y) coordinate for the right knee across all frames
    knee_y = data[:, knee_index, 1]

    # 4. Detect top-dead-center peaks in the knee trajectory
    fs = 60.0  # sampling rate (frames per second)
    min_distance = int(fs * 0.9)  # at 60 rpm (~1 rev/s), peaks ~0.9 s apart
    peaks, _ = find_peaks(knee_y, distance=min_distance)

    # 5. Compute mid-downstroke indices (midpoint between successive peaks)
    mid_idxs = ((peaks[:-1] + peaks[1:]) // 2).astype(int)

    # 6. Gather knee Y positions at mid-downstroke for each cycle
    mid_positions = knee_y[mid_idxs]

    # 7. Plot cycle-to-cycle consistency
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(mid_positions) + 1), mid_positions, marker='o', linestyle='-')
    plt.xlabel('Cycle Number')
    plt.ylabel('Knee Vertical Position (units)')
    plt.title('Cycle-to-Cycle Knee Height Consistency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cycle_consistency.png', dpi=300)
    plt.show()

    print("Cycle consistency plot saved as 'cycle_consistency.png'")

if __name__ == "__main__":
    main()
