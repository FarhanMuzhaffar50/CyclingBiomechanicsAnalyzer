import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 1. Load your 3D data
#    Replace with the actual path to your .npy file
poses = np.load('examples/ActualTest/Height5/X3D.npy')  
#    poses.shape == (n_frames, n_joints, 3)

# 2. Select a single pedal revolution (here: first 100 frames)
#    and pick the joint index for the right knee.
#    If you’re using Halpe-26, the right knee is often index 10 or 11—
#    confirm from your keypoint ordering.
joint_idx = 2
y_raw = poses[:120, joint_idx, 1]  # Y‐axis (vertical) coordinate

# 3. Design a 2nd‐order Butterworth low‐pass filter at 6 Hz
fs = 60.0  # sampling rate (frames per second)
cutoff = 6.0  # cutoff frequency in Hz
b, a = butter(N=2, Wn=cutoff/(0.5*fs), btype='low')

# 4. Apply zero‐phase filtering
y_filt = filtfilt(b, a, y_raw)

# 5. Plot raw vs. filtered
plt.figure(figsize=(6,3))
plt.plot(y_raw,     label='Raw',     linewidth=1)
plt.plot(y_filt,    label='Filtered',linewidth=2)
plt.xlabel('Frame')
plt.ylabel('Vertical Position (m)')
plt.legend()
plt.title('Pelvis: Raw vs. 6 Hz Butterworth Filter')
plt.tight_layout()
plt.show()
