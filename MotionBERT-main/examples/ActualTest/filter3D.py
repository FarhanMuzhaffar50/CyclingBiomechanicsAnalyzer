import numpy as np
from scipy.signal import butter, filtfilt

def main():
    # 1. Load your raw 3D pose file
    raw = np.load('examples/ActualTest/Height5/X3D.npy')  # shape: (n_frames, n_joints, 3)

    # 2. Design a 2nd‐order Butterworth low‐pass filter at 6 Hz
    fs = 60.0       # sampling rate in frames per second
    cutoff = 6.0    # cutoff frequency in Hz
    b, a = butter(N=2, Wn=cutoff/(0.5*fs), btype='low')

    # 3. Apply zero-phase filtering to each joint coordinate
    filtered = np.empty_like(raw)
    for j in range(raw.shape[1]):       # for each joint
        for d in range(3):              # for x, y, z
            filtered[:, j, d] = filtfilt(b, a, raw[:, j, d])

    # 4. Save the filtered array
    np.save('examples/ActualTest/Height5/X3Dfiltered.npy', filtered)
    print('Filtered data saved as X3Dfiltered.npy')

if __name__ == '__main__':
    main()
